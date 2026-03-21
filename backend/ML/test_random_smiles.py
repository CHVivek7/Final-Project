from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

from rdkit import Chem


CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.toxicity_service import ToxicityService


DEFAULT_SMILES: List[str] = [
    "CCO",  # ethanol
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
    "CC(=O)NC1=CC=C(O)C=C1",  # paracetamol
    "C1=CC=C(C=C1)N",  # aniline
    "C1=CC=C(C=C1)O",  # phenol
    "CCN(CC)CCOC(=O)C1=CC=CC=C1Cl",  # lidocaine
    "CCOC(=O)C1=CC=CC=C1",  # ethyl benzoate
    "CC(C)NCC(O)COc1ccccc1O",  # propranolol-like
]


def compute_final_summary(tox_probs: Dict[str, float]) -> Dict[str, object]:
    weights = {
        "NR-AR": 1.2,
        "NR-ER": 1.2,
        "SR-ARE": 1.0,
        "SR-MMP": 1.0,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    max_prob = 0.0
    very_high_count = 0

    for target, prob in tox_probs.items():
        p = float(min(max(prob, 0.0), 1.0))
        w = float(weights.get(target, 1.0))
        weighted_sum += p * w
        total_weight += w
        if p > max_prob:
            max_prob = p
        if p >= 0.55:
            very_high_count += 1

    weighted_mean = weighted_sum / total_weight if total_weight else 0.0
    base_score = (0.7 * weighted_mean) + (0.3 * max_prob)

    return {
        "base_score": float(base_score),
        "max_prob": float(max_prob),
        "very_high_count": int(very_high_count),
    }


def structural_alert_adjustment(smiles: str) -> Dict[str, object]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"boost": 0.0, "alerts": []}

    alerts = []
    boost = 0.0

    phenol = Chem.MolFromSmarts("[OX2H]-c1ccccc1")
    aromatic_aldehyde = Chem.MolFromSmarts("[CX3H1](=O)-c1ccccc1")
    nitro = Chem.MolFromSmarts("[NX3](=O)=O")
    aromatic_halide = Chem.MolFromSmarts("[F,Cl,Br,I]-c1ccccc1")
    aniline_like = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]-c1ccccc1")
    aromatic_amide = Chem.MolFromSmarts("c[NX3][CX3](=O)[#6]")
    simple_alcohol = Chem.MolFromSmarts("[CX4][OX2H]")

    if phenol is not None and mol.HasSubstructMatch(phenol):
        alerts.append("phenol")
        boost += 0.32

    if aromatic_aldehyde is not None and mol.HasSubstructMatch(aromatic_aldehyde):
        alerts.append("aromatic_aldehyde")
        boost += 0.18

    if nitro is not None and mol.HasSubstructMatch(nitro):
        alerts.append("nitro")
        boost += 0.15

    if aromatic_halide is not None and mol.HasSubstructMatch(aromatic_halide):
        alerts.append("aromatic_halide")
        boost += 0.08

    if aniline_like is not None and mol.HasSubstructMatch(aniline_like):
        alerts.append("aniline_like")
        boost += 0.10

    if aromatic_amide is not None and mol.HasSubstructMatch(aromatic_amide):
        alerts.append("aromatic_amide")
        boost -= 0.12

    if simple_alcohol is not None and mol.HasSubstructMatch(simple_alcohol):
        alerts.append("simple_alcohol")
        boost -= 0.05

    boost = float(min(max(boost, -0.20), 0.50))
    return {"boost": boost, "alerts": alerts}


def classify_score(base_score: float, max_prob: float, very_high_count: int, alert_boost: float) -> tuple[float, str]:
    final_score = float(min(max(base_score + alert_boost, 0.0), 1.0))

    if final_score >= 0.55 or very_high_count >= 2 or max_prob >= 0.70:
        risk = "HIGH"
    elif final_score >= 0.30 or max_prob >= 0.50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return final_score, risk


def run(smiles_count: int, seed: int) -> None:
    rng = random.Random(seed)
    smiles_count = max(1, min(smiles_count, len(DEFAULT_SMILES)))
    selected = rng.sample(DEFAULT_SMILES, smiles_count)

    service = ToxicityService()

    print(f"Testing {smiles_count} random SMILES (seed={seed})")
    print("-" * 80)

    for smiles in selected:
        result = service.predict_from_smiles(smiles)
        tox_probs = result["toxicity_probabilities"]
        core = compute_final_summary(tox_probs)
        alert = structural_alert_adjustment(smiles)
        final_score, risk = classify_score(
            base_score=float(core["base_score"]),
            max_prob=float(core["max_prob"]),
            very_high_count=int(core["very_high_count"]),
            alert_boost=float(alert["boost"]),
        )

        max_target = max(tox_probs, key=lambda k: tox_probs[k])
        summary = {
            "final_toxicity_score": round(final_score * 100.0, 2),
            "risk_level": risk,
            "highest_risk_target": max_target,
            "highest_risk_value": round(float(tox_probs[max_target]) * 100.0, 2),
            "base_score": round(float(core["base_score"]) * 100.0, 2),
            "alert_boost": round(float(alert["boost"]) * 100.0, 2),
            "structural_alerts": alert["alerts"],
        }

        top3 = sorted(tox_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_text = ", ".join([f"{k}:{v * 100.0:.1f}%" for k, v in top3])

        print(f"SMILES: {smiles}")
        print(
            f"  Score: {summary['final_toxicity_score']:.2f}% | "
            f"Risk: {summary['risk_level']} | "
            f"Highest: {summary['highest_risk_target']} ({summary['highest_risk_value']:.2f}%)"
        )
        print(
            f"  Base: {summary['base_score']:.2f}% | "
            f"Alert Boost: {summary['alert_boost']:.2f}% | "
            f"Alerts: {', '.join(summary['structural_alerts']) if summary['structural_alerts'] else 'none'}"
        )
        print(f"  Confidence: {result['confidence_score'] * 100.0:.1f}%")
        print(f"  Top targets: {top3_text}")
        print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random SMILES toxicity checks")
    parser.add_argument("--count", type=int, default=5, help="Number of random SMILES to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(smiles_count=args.count, seed=args.seed)