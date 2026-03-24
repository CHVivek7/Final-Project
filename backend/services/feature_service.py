# services/feature_service.py
from __future__ import annotations

from typing import Dict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys

try:
    from services.vqe_service import compute_quantum_features_from_mol
except ImportError:
    from .vqe_service import compute_quantum_features_from_mol


class FeatureService:
    MORGAN_BITS  = 2048
    RDKIT_BITS   = 2048
    MORGAN_RADIUS = 2
    _AUTO_MAX_HEAVY_ATOMS_FOR_VQE = 8

    # ── 3D conformer ──────────────────────────────────────────────────────
    @staticmethod
    def _smiles_to_3d_mol(smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status != 0:
            raise ValueError("Failed to generate 3D conformer")
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        return mol

    # ── Fingerprints ──────────────────────────────────────────────────────
    @staticmethod
    def _morgan_fp(mol: Chem.Mol) -> np.ndarray:
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=FeatureService.MORGAN_RADIUS,
            fpSize=FeatureService.MORGAN_BITS,
        )
        arr = np.zeros(FeatureService.MORGAN_BITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        return arr

    @staticmethod
    def _maccs_fp(mol: Chem.Mol) -> np.ndarray:
        # 167-bit MACCS keys — explicit bits for nitro, halogens,
        # aromatic rings, carbonyls, amines — directly maps to Tox21 endpoints
        arr = np.zeros(167, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
        return arr

    @staticmethod
    def _rdkit_fp(mol: Chem.Mol) -> np.ndarray:
        # Topological fingerprint — captures path-based features
        # that Morgan misses for symmetric molecules like benzene
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=FeatureService.RDKIT_BITS,
        )
        arr = np.zeros(FeatureService.RDKIT_BITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        return arr

    # ── Descriptors ───────────────────────────────────────────────────────
    @staticmethod
    def _descriptor_vector(mol: Chem.Mol) -> np.ndarray:
        # Remove Hs for descriptor calculation
        mol_noh = Chem.RemoveHs(mol)

        raw = np.array([
            Descriptors.MolWt(mol_noh),                          # molecular weight
            Crippen.MolLogP(mol_noh),                            # lipophilicity
            rdMolDescriptors.CalcTPSA(mol_noh),                  # polarity
            Lipinski.NumHDonors(mol_noh),                        # H-bond donors
            Lipinski.NumHAcceptors(mol_noh),                     # H-bond acceptors
            Lipinski.NumRotatableBonds(mol_noh),                 # flexibility
            # ── NEW: toxicity-relevant descriptors ──────────────────────
            rdMolDescriptors.CalcNumAromaticRings(mol_noh),      # aromatic rings → NR-AhR
            Descriptors.FractionCSP3(mol_noh),                   # sp3 fraction (low = aromatic)
            rdMolDescriptors.CalcNumHeteroatoms(mol_noh),        # N, O, S, halogens
            rdMolDescriptors.CalcNumHeavyAtoms(mol_noh),         # molecule size
            rdMolDescriptors.CalcNumRings(mol_noh),              # ring count
            # halogen counts — key for organochlorine / organohalide toxicity
            sum(1 for a in mol_noh.GetAtoms() if a.GetAtomicNum() ==  9),  # F
            sum(1 for a in mol_noh.GetAtoms() if a.GetAtomicNum() == 17),  # Cl
            sum(1 for a in mol_noh.GetAtoms() if a.GetAtomicNum() == 35),  # Br
            # formal charge — catches ionic nitro [N+][O-], charged species
            sum(abs(a.GetFormalCharge()) for a in mol_noh.GetAtoms()),
        ], dtype=np.float32)

        # Normalize each descriptor to roughly [0, 1]
        scale = np.array([
            500, 10, 200, 10, 10, 20,   # original 6
            5, 1, 20, 100, 10,          # new ring/heteroatom descriptors
            5, 10, 5, 4,                # halogen counts + charge
        ], dtype=np.float32)

        return np.clip(raw / scale, 0.0, 1.0)

    # ── Quantum (kept for VQE output only, NOT in ML feature vector) ──────
    @staticmethod
    def _approximate_quantum_features(mol: Chem.Mol) -> Dict[str, float | int | str]:
        heavy_atoms = mol.GetNumHeavyAtoms()
        ring_count  = rdMolDescriptors.CalcNumRings(mol)
        mw          = float(Descriptors.MolWt(mol))

        exact_energy = -(0.035 * heavy_atoms) - (0.0008 * mw) - (0.01 * ring_count)
        vqe_energy   = exact_energy + (0.002 + 0.0002 * max(heavy_atoms, 1))
        delta_energy = abs(exact_energy - vqe_energy)

        return {
            "vqe_energy":   float(vqe_energy),
            "exact_energy": float(exact_energy),
            "delta_energy": float(delta_energy),
            "qubit_count":  int(max(2, 2 * heavy_atoms)),
            "ansatz_type":  "ApproxQuantumProxy",
        }

    # ── Main entry point ──────────────────────────────────────────────────
    @classmethod
    def generate_unified_features(
        cls,
        smiles: str,
        prediction_mode: bool = True,
        quantum_mode: str = "auto",
    ) -> Dict[str, object]:

        mol = cls._smiles_to_3d_mol(smiles)
        heavy_atoms = mol.GetNumHeavyAtoms()

        # Quantum — for display only, NOT included in ML feature vector
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

        # ── Feature vector: fingerprints + descriptors only ───────────────
        # Quantum energies removed — they are heuristic proxies that
        # carry no real toxicity signal and dilute the fingerprint features
        morgan_vec     = cls._morgan_fp(mol)       # 2048
        maccs_vec      = cls._maccs_fp(mol)        #  167
        rdkit_vec      = cls._rdkit_fp(mol)        # 2048
        descriptor_vec = cls._descriptor_vector(mol)  #   15

        feature_vector = np.concatenate([
            morgan_vec,
            maccs_vec,
            rdkit_vec,
            descriptor_vec,
        ]).astype(np.float32)
        # Total: 4378 features

        return {
            "smiles":          smiles,
            "vector":          feature_vector,
            "quantum":         quantum,          # still returned for API output
            "descriptors":     descriptor_vec.tolist(),
            "fingerprint_bits": cls.MORGAN_BITS + 167 + cls.RDKIT_BITS,
        }