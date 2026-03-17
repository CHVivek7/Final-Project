from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.units import DistanceUnit
from qiskit.circuit.library import EfficientSU2
from pymongo import MongoClient
from datetime import datetime
import os
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from joblib import load
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ----------------------------
# MongoDB (optional)
# ----------------------------

MONGO_URI = os.getenv("MONGODB_ATLAS_URI")

client = MongoClient(MONGO_URI) if MONGO_URI is not None else None
db = client["quantum-sim"] if client is not None else None
results_collection = db["results"] if db is not None else None

# ----------------------------
# ML Model (optional)
# ----------------------------

model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "quantum_classical_predictor.joblib")

try:
    model = load(MODEL_PATH)
    logger.info("ML model loaded")
except:
    logger.warning("ML model not found")

# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Data Models
# ----------------------------

class AtomCoord(BaseModel):
    element: str
    x: float
    y: float
    z: float


class MoleculeInput(BaseModel):
    atoms: List[AtomCoord]
    charge: int = 0
    spin: int = 0
    use_quantum_hardware: bool = False
    basis_set: str = "sto3g"


class SimulationSuccess(BaseModel):
    molecule_name: str
    exact_energy: float
    vqe_energy: float
    ansatz_type: str
    status: str = "success"
    molecule_image: Optional[str] = None
    energy_plot: Optional[str] = None
    qubit_count: Optional[int] = None
    elements: Optional[List[str]] = None


class SimulationError(BaseModel):
    status: str = "failed"
    error: str
    suggestion: Optional[str] = None


class PredictionInput(BaseModel):
    num_atoms: int
    num_electrons: int
    num_qubits: int
    basis_set_size: int
    molecular_complexity: float


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    features: Dict[str, float]
    status: str = "success"

# ----------------------------
# Helper Functions
# ----------------------------

def generate_molecule_image(atoms: List[AtomCoord]):

    try:

        mol = Chem.RWMol()

        for atom in atoms:
            mol.AddAtom(Chem.Atom(atom.element))

        img = Draw.MolToImage(mol)

        buffer = BytesIO()
        img.save(buffer, format="PNG")

        return base64.b64encode(buffer.getvalue()).decode()

    except Exception as e:
        logger.warning(e)
        return ""


def create_energy_plot(distances, exact, vqe):

    plt.figure()

    plt.plot(distances, exact, label="Exact")
    plt.plot(distances, vqe, label="VQE")

    plt.legend()

    buffer = BytesIO()

    plt.savefig(buffer, format="png")
    plt.close()

    return base64.b64encode(buffer.getvalue()).decode()

# ----------------------------
# Simulation API
# ----------------------------

@app.post("/simulate/", response_model=Union[SimulationSuccess, SimulationError])
async def simulate_molecule(data: MoleculeInput):

    try:

        if len(data.atoms) < 2:
            return SimulationError(
                error="At least 2 atoms required"
            )

        geometry = "; ".join(
            f"{a.element} {float(a.x):.6f} {float(a.y):.6f} {float(a.z):.6f}"
            for a in data.atoms
        )

        driver = PySCFDriver(
            atom=geometry,
            unit=DistanceUnit.ANGSTROM,
            charge=data.charge,
            spin=data.spin,
            basis=data.basis_set
        )

        problem = driver.run()

        mapper = ParityMapper()

        qubit_op = mapper.map(problem.second_q_ops()[0])

        exact_energy = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
            qubit_op
        ).eigenvalue.real

        ansatz = EfficientSU2(
            qubit_op.num_qubits,
            reps=1,
            entanglement="linear"
        )

        optimizer = COBYLA(maxiter=100)

        estimator = StatevectorEstimator()

        vqe = VQE(estimator, ansatz, optimizer)

        result = vqe.compute_minimum_eigenvalue(qubit_op)

        vqe_energy = float(result.eigenvalue.real)

        molecule_name = "".join(sorted({a.element for a in data.atoms}))

        molecule_image = generate_molecule_image(data.atoms)

        result_data = SimulationSuccess(
            molecule_name=molecule_name,
            exact_energy=float(exact_energy),
            vqe_energy=float(vqe_energy),
            ansatz_type="EfficientSU2",
            qubit_count=qubit_op.num_qubits,
            elements=list({a.element for a in data.atoms}),
            molecule_image=molecule_image
        )

        return result_data

    except Exception as e:

        logger.error(e)

        return SimulationError(
            error=str(e),
            suggestion="Check atom coordinates or basis set"
        )

# ----------------------------
# Prediction API
# ----------------------------

@app.post("/predict/", response_model=Union[PredictionOutput, SimulationError])
async def predict_behavior(data: PredictionInput):

    if model is None:
        return SimulationError(error="ML model not available")

    try:

        features = [
            data.num_atoms,
            data.num_electrons,
            data.num_qubits,
            data.basis_set_size,
            data.molecular_complexity
        ]

        pred = model.predict([features])[0]

        prob = max(model.predict_proba([features])[0])

        return PredictionOutput(
            prediction="Quantum" if pred == 1 else "Classical",
            confidence=float(prob),
            features={
                "num_atoms": data.num_atoms,
                "num_electrons": data.num_electrons,
                "num_qubits": data.num_qubits,
                "basis_set_size": data.basis_set_size,
                "molecular_complexity": data.molecular_complexity
            }
        )

    except Exception as e:

        return SimulationError(error=str(e))

# ----------------------------
# Cache APIs
# ----------------------------

@app.get("/cache/stats")
async def cache_stats():

    if results_collection:
        return {"entries": results_collection.count_documents({})}

    return {"entries": 0}


@app.delete("/cache/clear")
async def clear_cache():

    if results_collection:
        results_collection.delete_many({})

    return {"status": "success"}

# ----------------------------
# Run Server
# ----------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )