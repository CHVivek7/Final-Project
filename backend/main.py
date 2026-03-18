from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from collections import Counter
from qiskit.primitives import StatevectorEstimator
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
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
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdDetermineBonds import DetermineBonds
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
    basis_set: str = "sto-3g"


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
# NEW: Safe geometry validation
# ----------------------------
def is_valid_geometry(atoms):
    import numpy as np

    if len(atoms) < 2:
        return False

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            a1, a2 = atoms[i], atoms[j]
            dist = np.linalg.norm([
                a1.x - a2.x,
                a1.y - a2.y,
                a1.z - a2.z
            ])

            if 0.5 < dist < 3.0:
                return True

    return False


# ----------------------------
# Helper Functions
# ----------------------------

def generate_molecule_image(atoms):
    try:
        xyz = f"{len(atoms)}\n\n"
        for atom in atoms:
            xyz += f"{atom.element} {atom.x} {atom.y} {atom.z}\n"

        mol = Chem.MolFromXYZBlock(xyz)
        if mol is None:
            return ""

        DetermineBonds(mol)
        Chem.SanitizeMol(mol)

        # Better 2D layout
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)

        # ✅ Safe options only
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2
        opts.padding = 0.1

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        img = drawer.GetDrawingText()

        return base64.b64encode(img).decode()

    except Exception as e:
        logger.warning(f"Image generation failed: {e}")
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
# Simulation API (FIXED)
# ----------------------------
@app.post("/simulate/", response_model=Union[SimulationSuccess, SimulationError])
async def simulate_molecule(data: MoleculeInput):

    try:

        # ✅ 1. Basic validation
        if len(data.atoms) < 2:
            return SimulationError(error="At least 2 atoms required")

        if not is_valid_geometry(data.atoms):
            return SimulationError(
                error="Invalid geometry",
                suggestion="Atoms too far or overlapping"
            )

        # ✅ 2. Build geometry string
        geometry = "; ".join(
            f"{a.element} {float(a.x):.6f} {float(a.y):.6f} {float(a.z):.6f}"
            for a in data.atoms
        )

        # ✅ 3. PySCF Driver (stable)
        driver = PySCFDriver(
            atom=geometry,
            unit=DistanceUnit.ANGSTROM,
            charge=data.charge,
            spin=data.spin,
            basis=data.basis_set
        )

        problem = driver.run()

        # ✅ 4. Reduce size for large molecules
        num_particles = problem.num_particles
        num_orbitals = problem.num_spatial_orbitals

        if num_orbitals > 10:
            transformer = ActiveSpaceTransformer(
                num_electrons=min(4, num_particles[0] + num_particles[1]),
                num_spatial_orbitals=4
            )
            problem = transformer.transform(problem)

        # ✅ 5. Mapping
        mapper = ParityMapper()
        qubit_op = mapper.map(problem.second_q_ops()[0])

        num_qubits = qubit_op.num_qubits

        # ✅ 6. Exact solver ONLY for small systems
        if num_qubits <= 12:
            exact_energy = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
                qubit_op
            ).eigenvalue.real
        else:
            exact_energy = None

        # ✅ 7. VQE (safe settings)
        ansatz = EfficientSU2(
            num_qubits,
            reps=1,
            entanglement="linear"
        )

        optimizer = COBYLA(maxiter=50)

        estimator = StatevectorEstimator()

        vqe = VQE(estimator, ansatz, optimizer)

        result = vqe.compute_minimum_eigenvalue(qubit_op)

        vqe_energy = float(result.eigenvalue.real)

        # ✅ 8. Visualization (always works)
        molecule_image = generate_molecule_image(data.atoms)

        counts = Counter([a.element for a in data.atoms])
        molecule_name = "".join(f"{el}{counts[el]}" for el in sorted(counts))

        return SimulationSuccess(
            molecule_name=molecule_name,
            exact_energy=float(exact_energy) if exact_energy is not None else 0.0,
            vqe_energy=vqe_energy,
            ansatz_type="EfficientSU2",
            qubit_count=num_qubits,
            elements=list({a.element for a in data.atoms}),
            molecule_image=molecule_image
        )

    except Exception as e:
        logger.error(f"Simulation failed: {e}")

        return SimulationError(
            error="Simulation failed",
            suggestion="Try smaller molecule or better geometry"
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