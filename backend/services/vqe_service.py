from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit
from rdkit import Chem


@dataclass
class QuantumFeatures:
    vqe_energy: float
    exact_energy: float
    delta_energy: float
    qubit_count: int
    ansatz_type: str


def _mol_to_geometry_string(mol: Chem.Mol) -> str:
    conf = mol.GetConformer()
    atoms = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        atoms.append(
            f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}"
        )

    return "; ".join(atoms)


def compute_quantum_features_from_mol(
    mol: Chem.Mol,
    prediction_mode: bool = True,
    maxiter: int = 10,
) -> Dict[str, float | int | str]:
    """Run a compact VQE workflow and return energies used by the toxicity model."""
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Cannot compute quantum features from an empty molecule")

    geometry = _mol_to_geometry_string(mol)

    driver = PySCFDriver(
        atom=geometry,
        unit=DistanceUnit.ANGSTROM,
        charge=0,
        spin=0,
        basis="sto3g",
    )

    problem = driver.run()

    if prediction_mode and problem.num_spatial_orbitals > 8:
        num_particles = problem.num_particles
        transformer = ActiveSpaceTransformer(
            num_electrons=min(4, num_particles[0] + num_particles[1]),
            num_spatial_orbitals=4,
        )
        problem = transformer.transform(problem)

    mapper = ParityMapper()
    qubit_op = mapper.map(problem.second_q_ops()[0])
    qubit_count = qubit_op.num_qubits

    exact_energy = None
    if qubit_count <= 12:
        exact_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
        exact_energy = float(exact_result.eigenvalue.real)

    reps = 1 if prediction_mode else 2
    safe_maxiter = min(maxiter, 10) if prediction_mode else maxiter

    ansatz = EfficientSU2(qubit_count, reps=reps, entanglement="linear")
    optimizer = COBYLA(maxiter=safe_maxiter)
    estimator = StatevectorEstimator()

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)
    vqe_energy = float(vqe_result.eigenvalue.real)

    if exact_energy is None:
        exact_energy = vqe_energy

    delta_energy = abs(exact_energy - vqe_energy)

    payload = QuantumFeatures(
        vqe_energy=vqe_energy,
        exact_energy=exact_energy,
        delta_energy=delta_energy,
        qubit_count=qubit_count,
        ansatz_type="EfficientSU2",
    )

    return {
        "vqe_energy": payload.vqe_energy,
        "exact_energy": payload.exact_energy,
        "delta_energy": payload.delta_energy,
        "qubit_count": payload.qubit_count,
        "ansatz_type": payload.ansatz_type,
    }
