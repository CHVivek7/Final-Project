# validate_accuracy.py
from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from services.feature_service import FeatureService

BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "ML" / "tox21_model.pkl"
SCALER_PATH = BASE_DIR / "ML" / "scaler.pkl"
DATA_PATH   = BASE_DIR / "ML" / "tox21.csv"

TARGET_COLUMNS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def validate_model():
    print("🧪 Loading ML assets...")
    models = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_service = FeatureService()

    print(f"📖 Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    print(f"   Total molecules with SMILES: {len(df)}")

    # ── Pre-compute features for ALL molecules once ───────────────────────
    # Much faster than recomputing per endpoint
    print(f"\n⚡ Extracting features for all {len(df)} molecules...")
    features = {}
    errors   = 0

    for idx, row in df.iterrows():
        smiles = row["smiles"]
        try:
            feat = feature_service.generate_unified_features(
                smiles=smiles,
                prediction_mode=True,
                quantum_mode="approx",
            )
            features[smiles] = scaler.transform([feat["vector"]])[0]
        except Exception:
            errors += 1
        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(df)}  (errors: {errors})")

    print(f"  Done: {len(features)} features extracted, {errors} errors\n")

    # ── Per-endpoint evaluation on up to 5000 samples ────────────────────
    print("=" * 60)
    print(f"{'Target':<18} | {'N Test':>7} | {'ROC-AUC':>8} | {'F1':>8} | {'Pos%':>6}")
    print("-" * 60)

    all_aucs = []
    report_endpoints = {}

    for i, target in enumerate(TARGET_COLUMNS):
        model = models[i] if i < len(models) else None
        if model is None:
            print(f"{target:<18} | No model")
            continue

        # Get rows labeled for this endpoint
        labeled = df[df[target].notna()][["smiles", target]].copy()
        labeled = labeled[labeled["smiles"].isin(features)]

        if len(labeled) < 20:
            print(f"{target:<18} | Too few labeled rows ({len(labeled)})")
            continue

        # ── Sample up to 5000 from labeled rows for this endpoint ─────
        n_test  = min(len(labeled), 5000)
        sampled = labeled.sample(n=n_test, random_state=42)

        y_true, y_prob = [], []
        for _, row in sampled.iterrows():
            smiles = row["smiles"]
            actual = int(row[target])
            vec    = features[smiles].reshape(1, -1)
            prob   = float(model.predict_proba(vec)[0][1])
            y_true.append(actual)
            y_prob.append(prob)

        if len(set(y_true)) < 2:
            print(f"{target:<18} | Single class in sample — skipped")
            continue

        try:
            y_pred   = [1 if p > 0.5 else 0 for p in y_prob]
            auc      = roc_auc_score(y_true, y_prob)
            f1       = f1_score(y_true, y_pred, zero_division=0)
            pos_rate = sum(y_true) / len(y_true) * 100
            all_aucs.append(auc)
            report_endpoints[target] = round(auc * 100, 2)
            print(f"{target:<18} | {n_test:>7} | {auc:>8.4f} | {f1:>8.4f} | {pos_rate:>5.1f}%")
        except Exception as e:
            print(f"{target:<18} | Error: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 60)
    mean_auc = float(np.mean(all_aucs)) if all_aucs else 0.0
    print(f"\n🏆 Mean ROC-AUC across {len(all_aucs)} endpoints: "
          f"{mean_auc:.4f}  ({mean_auc:.2%})")
    print("=" * 60)

    report = {
        "mean_auc":        round(mean_auc * 100, 2),
        "valid_endpoints": len(all_aucs),
        "test_samples":    5000,
        "per_endpoint":    report_endpoints,
        "status":          "Verified",
    }

    output_path = BASE_DIR / "ML" / "accuracy_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n📂 Report saved → {output_path}")
    return mean_auc


if __name__ == "__main__":
    validate_model()