import requests
import time

API_URL = "http://127.0.0.1:8000/simulate"

# 🧪 Expanded Test data (30 Molecules)
TEST_MOLECULES = [
    ("CC(=O)Oc1ccccc1C(=O)O", "MEDIUM"), # Aspirin
    ("CCO", "LOW"),                    # Ethanol
    ("CC(=O)NC1=CC=C(O)C=C1", "MEDIUM"),# Paracetamol
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "LOW"), # Caffeine
    ("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "MEDIUM"), # Ibuprofen
    ("CN1CCCC1C2=CN=CC=C2", "MEDIUM"), # Nicotine
    ("c1ccccc1", "HIGH"),              # Benzene
    ("C=O", "HIGH"),                   # Formaldehyde
    ("C1=CC=C(C=C1)O", "MEDIUM"),      # Phenol
    ("Nc1ccccc1", "HIGH"),             # Aniline
    ("OC(=O)C(O)C(O)C(O)C(O)CO", "LOW"),
    ("OCC(O)CO", "LOW"),
    ("CC(O)=O", "LOW"),
    ("OC1CCCCC1", "LOW"),
    ("O=C1NC(=O)c2ccccc21", "MEDIUM"),
    ("CC1=CC=C(C=C1)S(N)(=O)=O", "MEDIUM"),
    ("CCOC(=O)c1ccc(N)cc1", "MEDIUM"),
    ("ClCCCl", "MEDIUM"),
    ("[O-][N+](=O)c1ccccc1", "HIGH"),
    ("ClC(Cl)=CCl", "HIGH"),
    ("C(C(C(C(C(C=O)O)O)O)O)O", "LOW"), # Glucose
    ("CCCC", "LOW"),                   # Butane
    ("O=C(N)N", "LOW"),                # Urea
    ("c1ccc2c(c1)cccc2", "MEDIUM"),    # Naphthalene
    ("C1CCOC1", "MEDIUM"),             # THF
    ("CC(C)(C)OC(=O)N", "MEDIUM"),      # tert-Butyl carbamate
    ("CC(=O)Cl", "MEDIUM"),            # Acetyl chloride
    ("C#N", "HIGH"),                   # Hydrogen Cyanide
    ("ClC1=CC=C(C=C1)C(C(Cl)(Cl)Cl)C2=CC=C(Cl)C=C2", "HIGH"), # DDT
    ("CC(C)P(=O)(F)OCC", "HIGH"),      # Sarin-like
]

def run_test():
    print("\n🚀 Starting High-Accuracy Molecule Test (30 Samples)")
    print("-" * 90)
    print(f"{'Molecule/SMILES':<30} | {'Predicted':<10} | {'Expected':<10} | {'Status'}")
    print("-" * 90)

    correct = 0
    start_time = time.time()

    for smiles, expected in TEST_MOLECULES:
        try:
            resp = requests.post(API_URL, json={"smiles": smiles})
            if resp.status_code != 200:
                print(f"{smiles:<30} | Error: {resp.status_code}")
                continue
                
            data = resp.json()
            # Updated keys to match the new high-accuracy service
            predicted = data.get("final_summary", {}).get("risk_level", "ERR")
            
            status = "✅ OK" if predicted == expected else "❌ MISMATCH"
            if predicted == expected:
                correct += 1
                
            display_name = smiles[:28] + ".." if len(smiles) > 30 else smiles
            print(f"{display_name:<30} | {predicted:<10} | {expected:<10} | {status}")
            
        except Exception as e:
            print(f"{smiles:<30} | Exception: {e}")

    duration = round(time.time() - start_time, 2)
    accuracy = (correct / len(TEST_MOLECULES)) * 100

    print("-" * 90)
    print(f"🎯 FINAL ACCURACY: {accuracy:.2f}% ({correct}/{len(TEST_MOLECULES)})")
    print(f"⏱️  Total Time: {duration}s")
    print("-" * 90)

if __name__ == "__main__":
    run_test()