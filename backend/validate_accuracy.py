import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

from services.feature_service import FeatureService
from services.toxicity_service import TARGET_COLUMNS

# Load Assets
MODEL_PATH = Path("ML/tox21_model.pkl")
SCALER_PATH = Path("ML/scaler.pkl")
CONFIG_PATH = Path("ML/endpoint_thresholds.json")

def validate_model(test_csv_path):
    print("🧪 Starting Validation...")
    
    # 1. Load trained artifacts
    models = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    # 2. Load Test Data
    df = pd.read_csv(test_csv_path).dropna(subset=["smiles"]).head(100) # Test top 100
    
    y_true_risk = []
    y_pred_risk = []

    for _, row in df.iterrows():
        try:
            # Generate features exactly like the Service does
            feat = FeatureService.generate_unified_features(row["smiles"], quantum_mode="approx")
            X_scaled = scaler.transform([feat["vector"]])
            
            high_votes = 0
            for i, target in enumerate(TARGET_COLUMNS):
                model = models[i]
                if model is not None and target in config["useful_endpoints"]:
                    prob = model.predict_proba(X_scaled)[0][1]
                    # Apply the dynamic threshold calculated during training
                    if prob >= config["high_thresholds"][target]:
                        high_votes += 1
            
            # Predict Risk Level
            pred = "HIGH" if high_votes >= config["min_high_votes"] else "LOW"
            y_pred_risk.append(pred)
            
            # Determine Ground Truth (Simplified for validation)
            # If any target in the CSV is 1.0, it's actually HIGH
            actual = "HIGH" if (row[TARGET_COLUMNS] == 1.0).any() else "LOW"
            y_true_risk.append(actual)
            
        except Exception:
            continue

    # 3. Final Report
    print("\n📊 --- VALIDATION REPORT ---")
    print(f"Total Samples: {len(y_pred_risk)}")
    print(f"System Accuracy: {accuracy_score(y_true_risk, y_pred_risk):.2%}")
    print("\nDetailed Performance:")
    print(classification_report(y_true_risk, y_pred_risk))

if __name__ == "__main__":
    # Point this to your test CSV file
    validate_model("ML/tox21.csv")