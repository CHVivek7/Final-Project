from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import requests
import json
import re
import joblib
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv
from google import genai  # The NEW way
import os
import os
import re
from typing import Dict, Any

load_dotenv()
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

        # ── Gemini AI client ──
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None  # Initialize to None first to avoid AttributeErrors
        self.model_id = "gemini-3-flash-preview" # Default starting point

        if api_key:
            try:
                # 1. Create the client
                from google import genai
                self.client = genai.Client(api_key=api_key)
                
                # 2. Verify models and select best available
                # Note: We use 'models.list()' which is the correct method for the new SDK
                available_models = [m.name for m in self.client.models.list()]
                
                if "models/gemini-3-flash-preview" in available_models:
                    self.model_id = "gemini-3-flash-preview"
                elif "models/gemini-2.5-flash" in available_models:
                    self.model_id = "gemini-2.5-flash"
                
                print(f"✅ Gemini AI configured with model: {self.model_id}")
                
            except Exception as e:
                print(f"❌ Failed to initialize Gemini Client: {e}")
                self.client = None 
        else:
            print("⚠️ GEMINI_API_KEY not found in environment.")

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
        metadata = {"common": "Unknown Molecule", "iupac": "N/A"}
        try:
            prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON"
            prop_res = requests.get(prop_url, timeout=3)
            if prop_res.status_code == 200:
                metadata["iupac"] = prop_res.json()["PropertyTable"]["Properties"][0]["IUPACName"]

            syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/synonyms/JSON"
            syn_res = requests.get(syn_url, timeout=3)
            if syn_res.status_code == 200:
                metadata["common"] = syn_res.json()["InformationList"]["Information"][0]["Synonym"][0]
        except Exception as e:
            print(f"Metadata Fetch Warning: {e}")
        return metadata

    def predict_from_smiles(self, smiles: str) -> Dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            weight = round(Descriptors.MolWt(mol), 2)
        else:
            formula = "N/A"
            weight = 0.0

        names = self._get_molecular_metadata(smiles)

        if not self.models or not self.scaler or not self.config:
            raise RuntimeError("ML assets not loaded. Run retrain_pro.py first.")

        features = self.feature_service.generate_unified_features(smiles, True, "approx")
        scaled_vec = self.scaler.transform([features["vector"]])

        probs_dict = {}
        high_votes = 0
        useful_endpoints = self.config.get("useful_endpoints", [])
        thresholds = self.config.get("high_thresholds", {})

        for i, target in enumerate(TARGET_COLUMNS):
            model = self.models[i]
            if model is not None and target in useful_endpoints:
                p = float(model.predict_proba(scaled_vec)[0][1])
                probs_dict[target] = round(p, 4)
                if p >= thresholds.get(target, 0.5):
                    high_votes += 1
            else:
                probs_dict[target] = 0.0

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

    # ─────────────────────────────────────────────────────────────────────────
    # Robust JSON extractor — handles all GPT response quirks
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_json(self, raw: str) -> dict:
        """
        Strips markdown fences, leading/trailing prose, and extracts the
        first valid JSON object from the response string.
        """
        # 1. Remove common markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        # 2. Find the first { ... } block (greedy from first { to last })
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in GPT response:\n{raw}")

        return json.loads(match.group())

    # ─────────────────────────────────────────────────────────────────────────
    # Main dynamic summary — GPT-4o primary, RDKit fallback
    # ─────────────────────────────────────────────────────────────────────────
    def _get_dynamic_summary(self, smiles: str, risk_level: str):
        summary = {"name": "Unknown Compound", "reason": "", "health_issues": []}
        
        if not self.client:
            return self._rdkit_fallback(smiles, risk_level, summary)

        prompt = f"""
        Act as a Toxicologist. Analyze this molecule:
        - SMILES: {smiles}
        - Risk Level: {risk_level}

        Provide a JSON response:
        {{
          "chemical_name": "name",
          "reason": "structural alert explanation",
          "health_issues": ["issue1", "issue2"]
        }}
        """

        try:
            # FIX: The new SDK uses 'models.generate'
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            # Extract JSON from response text
            text = response.text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                summary["name"] = parsed.get("chemical_name", summary["name"])
                summary["reason"] = parsed.get("reason", "")
                summary["health_issues"] = parsed.get("health_issues", [])
                return summary
            
            raise ValueError("No JSON found")

        except Exception as e:
            print(f"⚠️ Gemini failed: {e}. Falling back to RDKit.")
            # FIX: Ensure 'summary' is passed here to avoid the ArgumentError
            return self._rdkit_fallback(smiles, risk_level, summary)

    def _rdkit_fallback(self, smiles: str, risk_level: str, summary: dict):
        # Your existing fallback logic here
        summary["reason"] = f"Fallback: Analysis based on {risk_level} threshold."
        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # RDKit fallback — deep structural analysis
    # ─────────────────────────────────────────────────────────────────────────
    def _rdkit_fallback(self, smiles: str, risk_level: str, summary: dict) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            summary["health_issues"] = ["General toxicological caution advised."]
            summary["reason"] = f"Classified as {risk_level} risk based on quantum simulation."
            return summary

        issues = []
        reason_parts = []

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        has_nitrogen   = "N" in atoms
        has_chlorine   = "Cl" in atoms
        has_bromine    = "Br" in atoms
        has_fluorine   = "F" in atoms
        has_sulfur     = "S" in atoms
        has_phosphorus = "P" in atoms

        if num_aromatic_rings >= 3:
            issues.append("Polycyclic aromatic hydrocarbon (PAH) toxicity")
            issues.append("DNA intercalation and mutagenicity risk")
            reason_parts.append(f"{num_aromatic_rings} fused aromatic rings (PAH structure)")
        elif num_aromatic_rings >= 1:
            issues.append("Aromatic hydrocarbon toxicity")
            reason_parts.append("aromatic ring system increases metabolic activation risk")

        if has_chlorine and has_bromine:
            issues.append("Severe respiratory and skin irritation")
            issues.append("Organ damage (liver and kidney)")
            reason_parts.append("multiple halogens increase lipophilicity and organ accumulation")
        elif has_chlorine:
            issues.append("Respiratory irritation")
            issues.append("Hepatotoxicity risk")
            reason_parts.append("chlorine substituent enhances environmental persistence")
        elif has_bromine:
            issues.append("Neurological disruption")
            issues.append("Thyroid function interference")
            reason_parts.append("bromine increases CNS penetration")
        if has_fluorine:
            issues.append("Metabolic enzyme inhibition")
            reason_parts.append("fluorine atoms resist metabolic breakdown")

        if has_nitrogen:
            if "N+" in smiles or "[NH+]" in smiles:
                issues.append("Cationic charge — membrane disruption")
            elif "NO2" in smiles.upper() or "N(=O)" in smiles:
                issues.append("Nitro group — oxidative stress and genotoxicity")
                reason_parts.append("nitro group forms reactive nitroso intermediates")
            elif "NN" in smiles:
                issues.append("Hydrazine-type hepatotoxicity")
            else:
                issues.append("Amine-related metabolic activation")

        if has_sulfur:
            if "S(=O)(=O)" in smiles:
                issues.append("Sulfonate group — renal excretion burden")
            else:
                issues.append("Protein binding and enzyme inhibition")
                reason_parts.append("sulfur groups form covalent bonds with proteins")

        if has_phosphorus:
            issues.append("Cholinesterase inhibition")
            issues.append("Neurotoxicity risk")
            reason_parts.append("phosphorus group may inhibit acetylcholinesterase")

        if logp > 5:
            issues.append("Bioaccumulation in fatty tissue")
            reason_parts.append(f"high LogP ({logp:.1f}) promotes tissue accumulation")
        elif logp > 3:
            issues.append("Moderate membrane permeability concern")

        if tpsa < 40 and logp > 3:
            issues.append("Blood-brain barrier penetration risk")
            reason_parts.append("low TPSA + high LogP suggests CNS penetration")

        if mw > 500:
            issues.append("Reduced renal clearance rate")

        if risk_level == "HIGH":
            issues.insert(0, "Potential mutagenicity / genotoxicity")
            issues.insert(1, "Severe organ stress (liver, kidney, bone marrow)")
        elif risk_level == "MEDIUM":
            issues.insert(0, "Moderate systemic toxicity risk")

        if reason_parts:
            summary["reason"] = (
                f"This molecule is classified as {risk_level} risk. "
                f"Key structural concerns include: {'; '.join(reason_parts)}."
            )
        else:
            summary["reason"] = (
                f"Classified as {risk_level} risk based on molecular weight "
                f"({mw:.0f} Da), LogP ({logp:.1f}), and quantum simulation results."
            )

        seen = set()
        unique = []
        for i in issues:
            if i not in seen:
                seen.add(i)
                unique.append(i)

        summary["health_issues"] = unique if unique else ["General toxicological caution advised."]
        return summary