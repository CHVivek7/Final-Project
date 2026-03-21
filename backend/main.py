from __future__ import annotations

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