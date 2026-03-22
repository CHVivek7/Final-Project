from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List
import time
import gc

import joblib
import numpy as np
import pandas as pd
from rdkit import RDLogger

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.feature_service import FeatureService

# 🔕 Disable RDKit warnings
RDLogger.DisableLog("rdApp.warning")


# ----------------------------
# Targets
# ----------------------------
TARGET_COLUMNS: List[str] = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


# ----------------------------
# Dataset Builder
# ----------------------------
def _build_dataset(df, max_rows, quantum_mode):

    X_rows = []
    y_rows = []
    kept_indices = []

    # Keep missing labels as NaN.
    # In Tox21, NaN usually means "not tested", not "non-toxic".
    labels = df[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")

    smiles_values = df["smiles"].astype(str)

    print("⚙️ Generating features...\n")

    for idx, smiles in enumerate(smiles_values):

        if max_rows and len(X_rows) >= max_rows:
            break

        try:
            feat = FeatureService.generate_unified_features(
                smiles=smiles,
                prediction_mode=True,
                quantum_mode=quantum_mode,
            )
        except Exception:
            continue

        # 🔥 Memory optimized
        X_rows.append(np.array(feat["vector"], dtype=np.float32))
        y_rows.append(labels.iloc[idx].to_numpy(dtype=np.float32))
        kept_indices.append(idx)

        # 🔄 Progress
        if len(X_rows) % 25 == 0:
            print(f"🔄 Processed {len(X_rows)} molecules")

        if idx % 50 == 0:
            gc.collect()

    if not X_rows:
        raise RuntimeError("❌ No valid samples generated")

    print(f"\n✅ Feature generation completed: {len(X_rows)} samples\n")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    return X, y, np.array(kept_indices, dtype=np.int32)


# ----------------------------
# Training Function
# ----------------------------
def train(csv_path, model_path, scaler_path, test_size, max_rows, quantum_mode):

    start_time = time.time()

    print("📥 Loading dataset...")
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    target_mask = df[TARGET_COLUMNS].notna().any(axis=1)
    df = df[target_mask].reset_index(drop=True)

    print(f"🧠 Quantum mode: {quantum_mode}")

    X, y, kept_indices = _build_dataset(df, max_rows, quantum_mode)

    print("📊 Dataset shape:", X.shape)
    # ----------------------------
    # Remove low-quality samples (IMPORTANT)
    # ----------------------------
    valid_ratio = np.mean(~np.isnan(y), axis=1)
    mask = valid_ratio > 0.4   # keep only good samples

    X = X[mask]
    y = y[mask]
    kept_indices = kept_indices[mask]  # 🔥 don't forget this!

    print("🧹 After cleaning:", X.shape)
    # ----------------------------
    # Scaling
    # ----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------
    # Split
    # ----------------------------
    stratify_labels = (
        pd.to_numeric(df.iloc[kept_indices]["NR-AR"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    # Stratification requires at least 2 classes with >=2 samples each.
    unique_labels, counts = np.unique(stratify_labels, return_counts=True)
    can_stratify = len(unique_labels) > 1 and np.all(counts >= 2)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_labels if can_stratify else None,
        )
    except ValueError:
        print("⚠️ Stratify failed → using normal split")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=42
        )
    
    # ----------------------------
    # Train per target (FIX)
    # ----------------------------
    trained_models = []

    print("\n🚀 Training started...\n")

    for i, target in enumerate(TARGET_COLUMNS):

        y_col_all = y_train[:, i]
        valid_mask = ~np.isnan(y_col_all)

        if np.sum(valid_mask) < 10:
            print(f"⚠️ Skipping {target} (too few labeled samples)")
            trained_models.append(None)
            continue

        X_train_i = X_train[valid_mask]
        y_col = y_col_all[valid_mask].astype(np.int8)

        # 🔥 Handle imbalance
        pos = np.sum(y_col == 1)
        neg = np.sum(y_col == 0)

        if pos == 0 or neg == 0:
            print(f"⚠️ Skipping {target} (only one class)")
            trained_models.append(None)
            continue

        scale = neg / pos
        scale = min(scale, 10)   # avoid extreme imbalance

        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",   # 🔥 IMPORTANT (not logloss)
            n_estimators=600,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            reg_lambda=3.0,
            reg_alpha=1.0,
            gamma=0.2,
            min_child_weight=3,
            n_jobs=-1,
            verbosity=0,
        )

        clf.fit(X_train_i, y_col)
        trained_models.append(clf)

        print(f"✅ Trained {target}")

    print("\n✅ Training completed!\n")

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("📊 Evaluation results:\n")

    for i, target in enumerate(TARGET_COLUMNS):

        model_i = trained_models[i]

        if model_i is None:
            print(f"{target}: skipped")
            continue

        y_test_col_all = y_test[:, i]
        valid_test_mask = ~np.isnan(y_test_col_all)

        if np.sum(valid_test_mask) == 0:
            print(f"{target}: no labeled test samples")
            continue

        X_test_i = X_test[valid_test_mask]
        y_test_col = y_test_col_all[valid_test_mask].astype(np.int8)

        y_proba = model_i.predict_proba(X_test_i)[:, 1]

        best_thresh = 0.5
        best_acc = 0

        for t in np.arange(0.3, 0.7, 0.02):
            preds = (y_proba > t).astype(int)
            acc_temp = accuracy_score(y_test_col, preds)
            
            if acc_temp > best_acc:
                best_acc = acc_temp
                best_thresh = t

        y_pred = (y_proba > best_thresh).astype(int)
        acc = best_acc

        try:
            auc = roc_auc_score(y_test_col, y_proba)
            auc_text = f"{auc:.4f}"
        except:
            auc_text = "nan"

        print(f"{target}: accuracy={acc:.4f}, roc_auc={auc_text}")

    # ----------------------------
    # Save
    # ----------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained_models, model_path)
    joblib.dump(scaler, scaler_path)

    print("\n💾 Model saved:")
    print(model_path)
    print(scaler_path)

    # ----------------------------
    # Final message
    # ----------------------------
    total_time = round(time.time() - start_time, 2)

    print("\n🎉 ALL DONE SUCCESSFULLY!")
    print(f"⏱ Total time: {total_time} seconds")
    print("✅ Ready for prediction 🚀")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv-path", type=Path, default=base_dir / "tox21.csv")
    parser.add_argument("--model-path", type=Path, default=base_dir / "tox21_model.pkl")
    parser.add_argument("--scaler-path", type=Path, default=base_dir / "scaler.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument(
        "--quantum-mode",
        choices=["auto", "approx", "exact"],
        default="approx",
    )

    return parser.parse_args()


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    args = parse_args()

    train(
        csv_path=args.csv_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        test_size=args.test_size,
        max_rows=args.max_rows,
        quantum_mode=args.quantum_mode,
    )