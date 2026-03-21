from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from joblib import load

try:
    from services.feature_service import FeatureService
except ImportError:
    from .feature_service import FeatureService


# ----------------------------
# Targets
# ----------------------------
TARGET_COLUMNS: List[str] = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


# ----------------------------
# Service
# ----------------------------
class ToxicityService:
    def __init__(
        self,
        model_path: str | None = None,
        scaler_path: str | None = None,
    ) -> None:

        base_dir = Path(__file__).resolve().parent.parent

        self.model_path = model_path or str(base_dir / "ML" / "tox21_model.pkl")
        self.scaler_path = scaler_path or str(base_dir / "ML" / "scaler.pkl")

        self.model = None
        self.scaler = None

    # ----------------------------
    # Load model + scaler
    # ----------------------------
    def _ensure_loaded(self) -> None:
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise RuntimeError(f"Model file not found: {self.model_path}")
            self.model = load(self.model_path)

        if self.scaler is None:
            if not os.path.exists(self.scaler_path):
                raise RuntimeError(f"Scaler file not found: {self.scaler_path}")
            self.scaler = load(self.scaler_path)

    # ----------------------------
    # Extract probability
    # ----------------------------
    @staticmethod
    def _positive_probability(prob_matrix: np.ndarray) -> float:
        if prob_matrix.ndim == 1:
            return float(prob_matrix[0])
        if prob_matrix.shape[1] == 1:
            return float(prob_matrix[0, 0])
        return float(prob_matrix[0, 1])

    # ----------------------------
    # Predict per target
    # ----------------------------
    def _predict_probabilities(self, scaled: np.ndarray) -> List[float]:

        # Case 1: MultiOutputClassifier-like
        if hasattr(self.model, "predict_proba"):
            prob_blocks = self.model.predict_proba(scaled)
            probs = [self._positive_probability(p) for p in prob_blocks]
            return [float(p) for p in probs[: len(TARGET_COLUMNS)]]

        # Case 2: list of models
        if isinstance(self.model, list):
            probs: List[float] = []

            for estimator in self.model[: len(TARGET_COLUMNS)]:

                if estimator is None:
                    probs.append(0.5)
                    continue

                if not hasattr(estimator, "predict_proba"):
                    probs.append(0.5)
                    continue

                prob_matrix = estimator.predict_proba(scaled)
                probs.append(self._positive_probability(prob_matrix))

            # pad if needed
            if len(probs) < len(TARGET_COLUMNS):
                probs.extend([0.5] * (len(TARGET_COLUMNS) - len(probs)))

            return [float(p) for p in probs]

        raise RuntimeError("Unsupported model format")

    # ----------------------------
    # Main prediction
    # ----------------------------
    def predict_from_smiles(self, smiles: str) -> Dict[str, object]:

        self._ensure_loaded()

        # 🔬 Feature generation
        features = FeatureService.generate_unified_features(
            smiles=smiles,
            prediction_mode=True,
        )

        vector = features["vector"]

        # 🔥 Scale features
        scaled = self.scaler.transform([vector])

        probs = self._predict_probabilities(scaled)

        toxicity_probabilities = {
            target: float(prob)
            for target, prob in zip(TARGET_COLUMNS, probs)
        }

        # ----------------------------
        # 🔥 Confidence score (improved)
        # ----------------------------
        confidence_score = float(
            np.mean(np.abs(np.array(probs) - 0.5) * 2.0)
        )

        # Penalize unstable quantum
        quantum = features["quantum"]
        if quantum["delta_energy"] > 0.1:
            confidence_score *= 0.5

        return {
            "smiles": smiles,
            "vqe_energy": float(quantum["vqe_energy"]),
            "exact_energy": float(quantum["exact_energy"]),
            "delta_energy": float(quantum["delta_energy"]),
            "toxicity_probabilities": toxicity_probabilities,
            "confidence_score": confidence_score,
        }