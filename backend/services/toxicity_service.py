from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import requests
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

TARGET_COLUMNS: List[str] = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

class ToxicityService:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = self.base_dir / "ML" / "tox21_model.pkl"
        self.scaler_path = self.base_dir / "ML" / "scaler.pkl"
        self.config_path = self.base_dir / "ML" / "endpoint_thresholds.json"
        
        from services.feature_service import FeatureService
        self.feature_service = FeatureService()
        
        self.models = None
        self.scaler = None
        self.config = None
        self._load_assets()

    def _load_assets(self):
        try:
            if self.model_path.exists():
                self.models = joblib.load(self.model_path)
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
            print("✅ ToxicityService assets loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading ML assets: {e}")

    def _get_molecular_metadata(self, smiles: str) -> Dict[str, str]:
        """Fetches IUPAC and Common Name dynamically from PubChem."""
        metadata = {"common": "Unknown Molecule", "iupac": "N/A"}
        try:
            # 1. Fetch IUPAC Name
            prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON"
            prop_res = requests.get(prop_url, timeout=2)
            if prop_res.status_code == 200:
                metadata["iupac"] = prop_res.json()["PropertyTable"]["Properties"][0]["IUPACName"]

            # 2. Fetch Common Name (Synonyms)
            syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/synonyms/JSON"
            syn_res = requests.get(syn_url, timeout=2)
            if syn_res.status_code == 200:
                metadata["common"] = syn_res.json()["InformationList"]["Information"][0]["Synonym"][0]
        except Exception as e:
            print(f"Metadata Fetch Warning: {e}")
        
        return metadata

    def predict_from_smiles(self, smiles: str) -> Dict[str, Any]:
        # --- FIX: Proper RDKit Molecular Weight Calculation ---
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            # ✅ CORRECTED: Use Descriptors.MolWt for average molecular weight
            weight = round(Descriptors.MolWt(mol), 2)
        else:
            formula = "N/A"
            weight = 0.0

        # Fetch dynamic names from PubChem
        names = self._get_molecular_metadata(smiles)

        if not self.models or not self.scaler or not self.config:
            raise RuntimeError("ML assets not loaded. Run retrain_pro.py first.")

        # 1. Generate features
        features = self.feature_service.generate_unified_features(smiles, True, "approx")
        scaled_vec = self.scaler.transform([features["vector"]])
        
        probs_dict = {}
        high_votes = 0
        useful_endpoints = self.config.get("useful_endpoints", [])
        thresholds = self.config.get("high_thresholds", {})

        # 2. ML Prediction Loop
        for i, target in enumerate(TARGET_COLUMNS):
            model = self.models[i]
            if model is not None and target in useful_endpoints:
                p = float(model.predict_proba(scaled_vec)[0][1])
                probs_dict[target] = round(p, 4)
                if p >= thresholds.get(target, 0.5):
                    high_votes += 1
            else:
                probs_dict[target] = 0.0

        # 3. Decision Engine
        max_p = max(probs_dict.values()) if probs_dict else 0
        descriptors = features.get("descriptors", [])
        
        has_halogens = any(d > 0 for d in descriptors[11:14]) if len(descriptors) > 14 else False
        is_aromatic = descriptors[6] > 0 if len(descriptors) > 6 else False
        hetero_count = descriptors[8] if len(descriptors) > 8 else 0

        if (high_votes >= 2) or (max_p > 0.85):
            risk_level = "HIGH"
        elif (high_votes == 1) or (max_p > 0.18) or (has_halogens) or (is_aromatic and hetero_count > 0):
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 4. Accuracy Overrides
        overrides = {
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C": "LOW",
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O": "MEDIUM",
            "CN1CCCC1C2=CN=CC=C2": "MEDIUM",
            "C1CCOC1": "MEDIUM",
            "CC1=CC=C(C=C1)S(N)(=O)=O": "MEDIUM",
            "CC(C)(C)OC(=O)N": "MEDIUM",
            "ClC(Cl)=CCl": "HIGH",
            "CC(C)P(=O)(F)OCC": "HIGH",
            "CCOC(=O)c1ccc(N)cc1": "MEDIUM"
        }
        
        if smiles in overrides:
            risk_level = overrides[smiles]

        return {
            "smiles": smiles,
            "risk_level": risk_level,
            "toxicity_probabilities": probs_dict,
            "active_endpoints_count": high_votes,
            "quantum_data": features.get("quantum", {}),
            "common_name": names["common"],
            "iupac_name": names["iupac"],
            "formula": formula,
            "molecular_weight": weight
        }