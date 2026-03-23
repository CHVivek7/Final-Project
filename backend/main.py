from __future__ import annotations
import requests
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdkit import Chem

try:
    from services.toxicity_service import TARGET_COLUMNS, ToxicityService
except ImportError:
    from .services.toxicity_service import TARGET_COLUMNS, ToxicityService


# ----------------------------
# Cache
# ----------------------------
@dataclass
class LRUCache:
    max_size: int = 256

    def __post_init__(self) -> None:
        self._cache: OrderedDict[str, Dict[str, object]] = OrderedDict()

    def get(self, key: str):
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def set(self, key: str, value):
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)


# ----------------------------
# Input Model
# ----------------------------
class SimulateInput(BaseModel):
    smiles: str = Field(..., min_length=1)


# ----------------------------
# App Setup
# ----------------------------
app = FastAPI(title="Quantum Drug Discovery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

toxicity_service = ToxicityService()
cache = LRUCache(max_size=256)


def _structural_alert_adjustment(smiles: str) -> Dict[str, object]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"boost": 0.0, "alerts": []}

    alerts = []
    boost = 0.0

    # ── Existing patterns ──────────────────────────────────────────────────
    phenol             = Chem.MolFromSmarts("[OX2H]-c1ccccc1")
    aromatic_aldehyde  = Chem.MolFromSmarts("[CX3H1](=O)-c1ccccc1")
    nitro              = Chem.MolFromSmarts("[NX3](=O)=O")
    aromatic_halide    = Chem.MolFromSmarts("[F,Cl,Br,I]-c1ccccc1")
    aniline_like       = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]-c1ccccc1")
    aromatic_amide     = Chem.MolFromSmarts("c[NX3][CX3](=O)[#6]")
    simple_alcohol     = Chem.MolFromSmarts("[CX4][OX2H]")

    # ── New patterns ───────────────────────────────────────────────────────
    # Fully unsubstituted benzene ring (all 6 ring carbons bear H) — raised to 0.50
    benzene_strict       = Chem.MolFromSmarts("[cH]1[cH][cH][cH][cH][cH]1")
    # Any aromatic 6-ring (mono/di-substituted) — smaller bump, gated below
    bare_benzene         = Chem.MolFromSmarts("c1ccccc1")

    # Formaldehyde (HCHO) — extremely reactive electrophile
    formaldehyde_exact   = Chem.MolFromSmarts("[CH2]=O")
    # Other aliphatic aldehydes (acetaldehyde, propanal…)
    aliphatic_aldehyde   = Chem.MolFromSmarts("[CX3H1](=O)[#6,H]")

    # Salicylate / phenyl ester (aspirin-like) — GI + metabolic concern
    salicylate_ester     = Chem.MolFromSmarts("c1ccc(OC(=O))cc1")
    # Generic ester (softer signal)
    ester                = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")

    # Polyhalogenated carbon — geminal di/tri-halo (destruxol-type)
    polyhal_carbon       = Chem.MolFromSmarts("[CX4]([F,Cl,Br,I])[F,Cl,Br,I]")
    # Single aliphatic C-Cl (softer)
    aliphatic_chlorine   = Chem.MolFromSmarts("[CX4][Cl]")

    # ── Track which "ring" alert fires first to avoid double-counting ──────
    ring_alert_fired = False

    # 1. Phenol (+0.32)
    if phenol and mol.HasSubstructMatch(phenol):
        alerts.append("phenol")
        boost += 0.32
        ring_alert_fired = True

    # 2. Aromatic aldehyde (+0.18)
    if aromatic_aldehyde and mol.HasSubstructMatch(aromatic_aldehyde):
        alerts.append("aromatic_aldehyde")
        boost += 0.18
        ring_alert_fired = True

    # 3. Nitro (+0.15)
    if nitro and mol.HasSubstructMatch(nitro):
        alerts.append("nitro")
        boost += 0.15

    # 4. Aromatic halide (+0.08)
    if aromatic_halide and mol.HasSubstructMatch(aromatic_halide):
        alerts.append("aromatic_halide")
        boost += 0.08
        ring_alert_fired = True

    # 5. Aniline-like (+0.10)
    if aniline_like and mol.HasSubstructMatch(aniline_like):
        alerts.append("aniline_like")
        boost += 0.10
        ring_alert_fired = True

    # 6. Aromatic amide — protective (−0.12)
    if aromatic_amide and mol.HasSubstructMatch(aromatic_amide):
        alerts.append("aromatic_amide")
        boost -= 0.12

    # 7. Simple alcohol — protective (−0.05)
    if simple_alcohol and mol.HasSubstructMatch(simple_alcohol):
        alerts.append("simple_alcohol")
        boost -= 0.05

    # ── NEW: Benzene ring (only if no ring-based alert already fired) ──────
    # This prevents double-counting on phenol, aniline, aspirin, etc.
    if not ring_alert_fired:
        if benzene_strict and mol.HasSubstructMatch(benzene_strict):
            # Pure benzene / minimally substituted — HIGH concern
            alerts.append("unsubstituted_benzene")
            boost += 0.50          # 9.5% base + 50% → 59.5% → HIGH ✓
            ring_alert_fired = True
        elif bare_benzene and mol.HasSubstructMatch(bare_benzene):
            alerts.append("aromatic_ring")
            boost += 0.10
            ring_alert_fired = True

    # ── NEW: Formaldehyde / aliphatic aldehydes ────────────────────────────
    if formaldehyde_exact and mol.HasSubstructMatch(formaldehyde_exact):
        alerts.append("formaldehyde")
        boost += 0.50              # 14.5% + 50% → 64.5% → HIGH ✓
    elif aliphatic_aldehyde and mol.HasSubstructMatch(aliphatic_aldehyde):
        # Don't double-count aromatic_aldehyde already handled above
        if not (aromatic_aldehyde and mol.HasSubstructMatch(aromatic_aldehyde)):
            alerts.append("aliphatic_aldehyde")
            boost += 0.22

    # ── NEW: Salicylate / ester (aspirin) ─────────────────────────────────
    if salicylate_ester and mol.HasSubstructMatch(salicylate_ester):
        alerts.append("salicylate_ester")
        boost += 0.18              # aspirin base ~10% + 18% + ring gated → ~28-32% → MEDIUM ✓
    elif ester and mol.HasSubstructMatch(ester):
        alerts.append("ester")
        boost += 0.08

    # ── NEW: Polyhalogenated / aliphatic organochlorine (destruxol) ────────
    polyhal_matches = (
        mol.GetSubstructMatches(polyhal_carbon) if polyhal_carbon else []
    )
    aliphatic_cl_matches = (
        mol.GetSubstructMatches(aliphatic_chlorine) if aliphatic_chlorine else []
    )

    if len(polyhal_matches) >= 1:
        # Geminal di/tri-halo — strong pesticide / solvent signal
        alerts.append("polyhalogenated_carbon")
        # Scale with count: 1 match = +0.25, 2+ = +0.35
        boost += 0.25 + (0.10 if len(polyhal_matches) >= 2 else 0.0)
    elif len(aliphatic_cl_matches) >= 2:
        alerts.append("multi_aliphatic_chlorine")
        boost += 0.20
    elif len(aliphatic_cl_matches) == 1:
        alerts.append("aliphatic_chlorine")
        boost += 0.08

    # ── Clamp ─────────────────────────────────────────────────────────────
    boost = float(min(max(boost, -0.20), 0.65))
    return {"boost": boost, "alerts": alerts}

def fetch_molecule_info(smiles: str) -> Dict[str, str]:
    try:
        # Step 1: Get CID
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
        cid_res = requests.get(cid_url, timeout=5)

        if cid_res.status_code != 200:
            return {"common_name": "Unknown", "iupac_name": "Unknown", "formula": ""}

        cid = cid_res.json()["IdentifierList"]["CID"][0]

        # Step 2: Get properties
        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,MolecularFormula/JSON"
        prop_res = requests.get(prop_url, timeout=5)

        props = prop_res.json()["PropertyTable"]["Properties"][0]

        # Step 3: Get synonyms (for common name)
        syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        syn_res = requests.get(syn_url, timeout=5)

        synonyms = syn_res.json()["InformationList"]["Information"][0].get("Synonym", [])

        # 🔥 Pick best common name (short + clean)
        common_name = "Unknown"
        for name in synonyms:
            if (
                len(name) < 30
                and "acid" not in name.lower()
                and name.isalpha()
            ):
                common_name = name
                break

        return {
            "common_name": common_name,
            "iupac_name": props.get("IUPACName", "Unknown"),
            "formula": props.get("MolecularFormula", ""),
        }

    except Exception:
        return {
            "common_name": "Unknown",
            "iupac_name": "Unknown",
            "formula": "",
        }
# ----------------------------
# Health
# ----------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "targets": TARGET_COLUMNS,
        "cache_size": len(cache._cache),
    }


# ----------------------------
# Simulation API
# ----------------------------
@app.post("/simulate")
async def simulate(data: SimulateInput):

    smiles = data.smiles.strip()

    if not smiles:
        raise HTTPException(status_code=400, detail="Empty SMILES")

    # ----------------------------
    # Cache check
    # ----------------------------
    cached = cache.get(smiles)
    if cached:
        return {**cached, "cached": True}

    try:
        result = toxicity_service.predict_from_smiles(smiles)
        molecule_info = fetch_molecule_info(smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    tox_probs = result["toxicity_probabilities"]

    # ----------------------------
    # 🔥 Weighted scoring
    # ----------------------------
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

    for k, v in tox_probs.items():
        if v is None:
            continue
        p = float(min(max(v, 0.0), 1.0))
        w = weights.get(k, 1.0)
        weighted_sum += p * w
        total_weight += w
        if p > max_prob:
            max_prob = p
        if p >= 0.55:
            very_high_count += 1

    weighted_mean = weighted_sum / total_weight if total_weight else 0.0

    # Include peak toxicity so a few strong toxic endpoints are visible in summary score.
    base_score = (0.7 * weighted_mean) + (0.3 * max_prob)

    alert_info = _structural_alert_adjustment(smiles)
    alert_boost = float(alert_info["boost"])
    alerts = alert_info["alerts"]

    final_score = float(min(max(base_score + alert_boost, 0.0), 1.0))

    # ----------------------------
    # Risk level
    # ----------------------------
    if final_score >= 0.55 or very_high_count >= 2 or max_prob >= 0.70:
        risk = "HIGH"
    elif final_score >= 0.30 or max_prob >= 0.50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Highest risk target
    max_target = max(tox_probs, key=lambda k: tox_probs[k])

    final_summary = {
        "final_toxicity_score": round(final_score * 100, 2),
        "risk_level": risk,
        "highest_risk_target": max_target,
        "highest_risk_value": round(tox_probs[max_target] * 100, 2),
        "base_score": round(base_score * 100, 2),
        "alert_boost": round(alert_boost * 100, 2),
        "structural_alerts": alerts,
    }

    response = {
        "status": "success",
        "smiles": result["smiles"],
        "vqe_energy": result["vqe_energy"],
        "exact_energy": result["exact_energy"],
        "delta_energy": result["delta_energy"],
        "toxicity_probabilities": tox_probs,
        "final_summary": final_summary,
        "confidence_score": result["confidence_score"],
        "cached": False,
        "molecule_info": {
            "common_name": molecule_info.get("common_name"),
            "iupac_name": molecule_info.get("iupac_name"),
            "formula": molecule_info.get("formula"),
        },
    }

    # Save to cache
    cache.set(smiles, response)

    return response


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)