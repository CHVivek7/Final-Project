from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.toxicity_service import ToxicityService


BENCHMARK: List[Tuple[str, str, str]] = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "LOW"),
    ("Ethanol", "CCO", "LOW"),
    ("Paracetamol", "CC(=O)NC1=CC=C(O)C=C1", "MEDIUM"),
    ("Complex", "CC(C)NCC(O)COc1ccccc1O", "MEDIUM"),
    ("Phenol", "C1=CC=C(C=C1)O", "HIGH"),
]

WEIGHTS = {
    "NR-AR": 1.2,
    "NR-ER": 1.2,
    "SR-ARE": 1.0,
    "SR-MMP": 1.0,
}

RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def compute_features(tox_probs: Dict[str, float]) -> Tuple[float, float]:
    weighted_sum = 0.0
    total_weight = 0.0
    max_prob = 0.0

    for target, prob in tox_probs.items():
        p = float(min(max(prob, 0.0), 1.0))
        w = float(WEIGHTS.get(target, 1.0))
        weighted_sum += p * w
        total_weight += w
        if p > max_prob:
            max_prob = p

    weighted_mean = weighted_sum / total_weight if total_weight else 0.0
    return weighted_mean, max_prob


def classify(weighted_mean: float, max_prob: float, alpha: float, med_score: float, med_max: float, high_max: float) -> str:
    final_score = (alpha * weighted_mean) + ((1.0 - alpha) * max_prob)

    if max_prob >= high_max:
        return "HIGH"
    if final_score >= med_score or max_prob >= med_max:
        return "MEDIUM"
    return "LOW"


def grade(pred: List[str], exp: List[str]) -> Tuple[int, int]:
    exact = sum(1 for p, e in zip(pred, exp) if p == e)
    ordered_error = sum(abs(RANK[p] - RANK[e]) for p, e in zip(pred, exp))
    return exact, ordered_error


def main() -> None:
    svc = ToxicityService()

    records = []
    print("Benchmark predictions from current model:")
    for name, smiles, expected in BENCHMARK:
        result = svc.predict_from_smiles(smiles)
        weighted_mean, max_prob = compute_features(result["toxicity_probabilities"])
        records.append((name, expected, weighted_mean, max_prob))
        print(
            f"- {name}: weighted_mean={weighted_mean * 100:.2f}%, max_prob={max_prob * 100:.2f}%, expected={expected}"
        )

    expected_labels = [x[1] for x in records]

    alpha_grid = [i / 100.0 for i in range(50, 91, 5)]
    med_score_grid = [i / 100.0 for i in range(20, 46)]
    med_max_grid = [i / 100.0 for i in range(35, 66)]
    high_max_grid = [i / 100.0 for i in range(50, 81)]

    best = None

    for alpha, med_score, med_max, high_max in itertools.product(
        alpha_grid, med_score_grid, med_max_grid, high_max_grid
    ):
        if med_max >= high_max:
            continue

        predicted = [
            classify(wm, mp, alpha, med_score, med_max, high_max)
            for _, _, wm, mp in records
        ]

        exact, ordered_error = grade(predicted, expected_labels)

        candidate = (exact, -ordered_error, alpha, med_score, med_max, high_max, predicted)
        if best is None or candidate > best:
            best = candidate

    assert best is not None
    exact, neg_err, alpha, med_score, med_max, high_max, predicted = best
    ordered_error = -neg_err

    print("\nBest threshold configuration:")
    print(f"- alpha={alpha:.2f}")
    print(f"- med_score={med_score:.2f}")
    print(f"- med_max_prob={med_max:.2f}")
    print(f"- high_max_prob={high_max:.2f}")
    print(f"- exact_matches={exact}/{len(BENCHMARK)}")
    print(f"- ordered_error={ordered_error}")

    print("\nPer-molecule labels with best configuration:")
    for (name, expected, wm, mp), pred in zip(records, predicted):
        final_score = (alpha * wm) + ((1.0 - alpha) * mp)
        print(
            f"- {name}: predicted={pred}, expected={expected}, final_score={final_score * 100:.2f}%, max_prob={mp * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
