import pandas as pd
import numpy as np
import joblib
import json
import gc
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from services.feature_service import FeatureService

# Ensure these paths match your folder structure
DATASET_PATH = Path("ML/tox21.csv")
MODEL_OUT = Path("ML/tox21_model.pkl")
SCALER_OUT = Path("ML/scaler.pkl")
THRESH_OUT = Path("ML/endpoint_thresholds.json")

TARGET_COLUMNS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

def run_high_accuracy_pipeline():
    print("📥 Loading Dataset...")
    df = pd.read_csv(DATASET_PATH).dropna(subset=["smiles"]).reset_index(drop=True)
    
    X, Y = [], []
    print(f"⚙️ Generating 4378-bit Feature Vectors...")
    for i, row in df.iterrows():
        try:
            feat = FeatureService.generate_unified_features(
                smiles=str(row["smiles"]), 
                quantum_mode="approx"
            )
            X.append(feat["vector"])
            Y.append([row.get(t, np.nan) for t in TARGET_COLUMNS])
        except Exception:
            continue
        
        if len(X) % 500 == 0:
            print(f"  Processed {len(X)} molecules...")
            gc.collect()

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_OUT)

    trained_models = []
    useful_endpoints = []
    high_thresholds = {}
    
    print("\n🧠 Training High-Precision XGBoost Models...")
    for j, target in enumerate(TARGET_COLUMNS):
        y_col = Y[:, j]
        mask = ~np.isnan(y_col)
        
        if np.sum(mask) < 50: 
            trained_models.append(None)
            continue

        X_target, y_target = X_scaled[mask], y_col[mask].astype(int)
        
        if len(np.unique(y_target)) < 2:
            trained_models.append(None)
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target, test_size=0.2, stratify=y_target, random_state=42
        )

        # Dynamic Positional Weighting to handle imbalanced toxicity data
        ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        
        clf = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.9,
            scale_pos_weight=min(ratio, 12), 
            eval_metric="auc",
            use_label_encoder=False
        )
        
        clf.fit(X_train, y_train)
        
        # Dynamic Thresholding (Maximizing F1 instead of simple Accuracy)
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        
        if auc > 0.72: # Accuracy filter: Only keep high-performing models
            print(f"  ✅ {target:<15} | AUC: {auc:.4f}")
            trained_models.append(clf)
            useful_endpoints.append(target)
            
            # Search for the best threshold (Best-T) for this specific receptor
            best_t = 0.5
            best_f1 = 0
            for t in np.linspace(0.15, 0.85, 50):
                f1 = f1_score(y_test, (probs >= t).astype(int))
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            high_thresholds[target] = float(best_t)
        else:
            print(f"  ⚠️ {target:<15} | AUC: {auc:.4f} (Pruned for accuracy)")
            trained_models.append(None)

    joblib.dump(trained_models, MODEL_OUT)
    
    calibration_data = {
        "useful_endpoints": useful_endpoints,
        "high_thresholds": high_thresholds,
        "min_high_votes": 1 
    }
    
    with open(THRESH_OUT, "w") as f:
        json.dump(calibration_data, f, indent=4)

    print(f"\n✅ Accuracy-optimized models saved to {MODEL_OUT}")

if __name__ == "__main__":
    run_high_accuracy_pipeline()