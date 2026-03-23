# diagnose.py  — place this file in backend/
from services.toxicity_service import ToxicityService, TARGET_COLUMNS

svc = ToxicityService()

MISMATCHES = [
    ("c1ccccc1",                     "HIGH",   "benzene"),
    ("C=O",                          "HIGH",   "formaldehyde"),
    ("[O-][N+](=O)c1ccccc1",         "HIGH",   "nitrobenzene"),
    ("ClC(Cl)=CCl",                  "HIGH",   "trichloroethylene"),
    ("O=C1NC(=O)c2ccccc21",          "MEDIUM", "phthalimide"),
    ("CC1=CC=C(C=C1)S(N)(=O)=O",     "MEDIUM", "tosylamide"),
]

print(f"\n{'Molecule':<20}", end="")
for t in TARGET_COLUMNS:
    print(f"  {t:<14}", end="")
print()
print("-" * 220)

for smiles, expected, name in MISMATCHES:
    result = svc.predict_from_smiles(smiles)
    probs  = result["toxicity_probabilities"]
    print(f"{name:<20}", end="")
    for t in TARGET_COLUMNS:
        p = probs[t]
        marker = "▲" if p > 0.4 else " "
        print(f"  {marker}{p:.3f}{'':9}", end="")
    print(f"  ← expected {expected}")