from __future__ import annotations

from typing import Dict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

try:
    from services.vqe_service import compute_quantum_features_from_mol
except ImportError:
    from .vqe_service import compute_quantum_features_from_mol


class FeatureService:
    FINGERPRINT_BITS = 1024
    _MORGAN_RADIUS = 2
    _AUTO_MAX_HEAVY_ATOMS_FOR_VQE = 8

    @staticmethod
    def _smiles_to_3d_mol(smiles: str) -> Chem.Mol:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as exc:
            raise ValueError(f"SMILES parsing failed: {exc}") from exc

        if mol is None:
            raise ValueError("Invalid SMILES string")

        mol = Chem.AddHs(mol)

        embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if embed_status != 0:
            raise ValueError("Failed to generate a 3D conformer from SMILES")

        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            # If force-field optimization fails, the embedded geometry is still usable.
            pass

        return mol

    @staticmethod
    def _descriptor_vector(mol: Chem.Mol) -> np.ndarray:
        descriptors = np.array(
            [
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                rdMolDescriptors.CalcTPSA(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
            ],
            dtype=np.float32,
        )
        return descriptors

    @staticmethod
    def _approximate_quantum_features(mol: Chem.Mol) -> Dict[str, float | int | str]:
        """Lightweight quantum proxy used when full VQE is too expensive."""
        heavy_atoms = mol.GetNumHeavyAtoms()
        ring_count = rdMolDescriptors.CalcNumRings(mol)
        mw = float(Descriptors.MolWt(mol))

        # Heuristic energies that scale with molecule size/complexity.
        exact_energy = -(0.035 * heavy_atoms) - (0.0008 * mw) - (0.01 * ring_count)
        vqe_energy = exact_energy + (0.002 + 0.0002 * max(heavy_atoms, 1))
        delta_energy = abs(exact_energy - vqe_energy)

        return {
            "vqe_energy": float(vqe_energy),
            "exact_energy": float(exact_energy),
            "delta_energy": float(delta_energy),
            "qubit_count": int(max(2, 2 * heavy_atoms)),
            "ansatz_type": "ApproxQuantumProxy",
        }

    @staticmethod
    def _fingerprint_vector(mol: Chem.Mol, n_bits: int) -> np.ndarray:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FeatureService._MORGAN_RADIUS,
            fpSize=n_bits,
        )
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    @classmethod
    def generate_unified_features(
        cls,
        smiles: str,
        prediction_mode: bool = True,
        quantum_mode: str = "auto",
    ) -> Dict[str, object]:

        mol = cls._smiles_to_3d_mol(smiles)

        heavy_atoms = mol.GetNumHeavyAtoms()

        use_approx = quantum_mode == "approx" or (
            quantum_mode == "auto" and heavy_atoms > cls._AUTO_MAX_HEAVY_ATOMS_FOR_VQE
        )

        if use_approx:
            quantum = cls._approximate_quantum_features(mol)
        else:
            try:
                quantum = compute_quantum_features_from_mol(
                    mol=mol,
                    prediction_mode=prediction_mode,
                    maxiter=10,
                )
            except Exception:
                quantum = cls._approximate_quantum_features(mol)

        # ----------------------------
        # 🔥 NORMALIZE QUANTUM SCALE
        # ----------------------------
        quantum_vec = np.array(
            [
                quantum["vqe_energy"] * 100,   # scale up
                quantum["exact_energy"] * 100,
                quantum["delta_energy"] * 1000,
            ],
            dtype=np.float32,
        )

        descriptor_vec = cls._descriptor_vector(mol)

        # 🔥 normalize descriptors (rough scaling)
        descriptor_vec = descriptor_vec / np.array(
            [500, 10, 200, 10, 10, 20], dtype=np.float32
        )

        fingerprint_vec = cls._fingerprint_vector(mol, cls.FINGERPRINT_BITS)

        feature_vector = np.concatenate(
            [quantum_vec, descriptor_vec, fingerprint_vec], axis=0
        ).astype(np.float32)

        return {
            "smiles": smiles,
            "vector": feature_vector,
            "quantum": quantum,
            "descriptors": descriptor_vec.tolist(),
            "fingerprint_bits": cls.FINGERPRINT_BITS,
        }