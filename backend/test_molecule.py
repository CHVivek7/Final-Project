import requests

API_URL = "http://127.0.0.1:8000/simulate"

# 🧪 Test data with EXPECTED risk
TEST_MOLECULES = [
    # ── Original 10 ───────────────────────────────────────────────────────
    ("CC(=O)Oc1ccccc1C(=O)O", "MEDIUM"),              # Aspirin
    ("CCO", "LOW"),                                    # Ethanol
    ("CC(=O)NC1=CC=C(O)C=C1", "MEDIUM"),               # Paracetamol
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "LOW"),             # Caffeine
    ("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "MEDIUM"),     # Ibuprofen
    ("CN1CCCC1C2=CN=CC=C2", "MEDIUM"),                 # Nicotine
    ("c1ccccc1", "HIGH"),                              # Benzene
    ("C=O", "HIGH"),                                   # Formaldehyde
    ("C1=CC=C(C=C1)O", "MEDIUM"),                      # Phenol
    ("Nc1ccccc1", "HIGH"),                             # Aniline

    # ── New 10 ────────────────────────────────────────────────────────────

    # LOW toxicity
    ("OC(=O)C(O)C(O)C(O)C(O)CO", "LOW"),              # Gluconic acid       — sugar acid, non-toxic
    ("OCC(O)CO", "LOW"),                               # Glycerol            — food-safe humectant
    ("CC(O)=O", "LOW"),                                # Acetic acid         — vinegar, low concern
    ("OC1CCCCC1", "LOW"),                              # Cyclohexanol        — mild solvent alcohol

    # MEDIUM toxicity
    ("O=C1NC(=O)c2ccccc21", "MEDIUM"),                 # Isatoic anhydride   — aromatic amide/ester
    ("CC1=CC=C(C=C1)S(N)(=O)=O", "MEDIUM"),            # p-Toluenesulfonamide — mild industrial irritant
    ("CCOC(=O)c1ccc(N)cc1", "MEDIUM"),                 # Benzocaine          — ester + aniline-like
    ("ClCCCl", "MEDIUM"),                              # 1,2-Dichloroethane  — aliphatic di-chloro

    # HIGH toxicity
    ("[O-][N+](=O)c1ccccc1", "HIGH"),                  # Nitrobenzene        — nitro + aromatic ring
    ("ClC(Cl)=CCl", "HIGH"),                           # Trichloroethylene   — multi-halo, carcinogen
]


def test_molecules():
    print("\n🧪 Testing Molecules\n" + "-"*80)

    correct = 0

    for smiles, expected_risk in TEST_MOLECULES:
        try:
            response = requests.post(API_URL, json={"smiles": smiles})
            data = response.json()

            summary = data.get("final_summary", {})
            molecule_info = data.get("molecule_info", {})

            name = molecule_info.get("common_name") or smiles
            score = summary.get("final_toxicity_score", 0)
            predicted_risk = summary.get("risk_level", "UNKNOWN")

            status = "OK" if predicted_risk == expected_risk else "MISMATCH"

            if status == "OK":
                correct += 1

            print(
                f"{name:<18} | {score:>6}% | "
                f"Pred: {predicted_risk:<6} | Exp: {expected_risk:<6} | {status}"
            )

        except Exception as e:
            print(f"{smiles} → ERROR: {e}")

    print("-"*80)

    total = len(TEST_MOLECULES)
    accuracy = (correct / total) * 100
    print(f"🎯 Accuracy: {accuracy:.2f}% ({correct}/{total})\n")

    # ── Per-group breakdown ────────────────────────────────────────────────
    groups = {"LOW": [0,0], "MEDIUM": [0,0], "HIGH": [0,0]}
    for smiles, expected_risk in TEST_MOLECULES:
        groups[expected_risk][1] += 1

    print("📊 Expected distribution:")
    for risk, (_, total_in_group) in groups.items():
        print(f"   {risk:<6}: {total_in_group} molecules")


if __name__ == "__main__":
    test_molecules()