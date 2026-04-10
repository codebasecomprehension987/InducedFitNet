"""
Atom and residue featurization for InducedFitNet.

Residue features  : one-hot amino-acid type (21 classes) + backbone torsion
                    angles (sin/cos of φ, ψ, ω) = 27-dim vector.
Ligand features   : one-hot element (11 classes) + formal charge + is_aromatic
                    + ring membership + hybridisation (3) = 17-dim vector.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

ELEMENT_ORDER = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]
ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENT_ORDER)}

RESIDUE_FEATURE_DIM = 21 + 6   # 21 one-hot + 6 torsion sin/cos
LIGAND_FEATURE_DIM  = 11 + 6   # 11 one-hot + formal_charge, aromatic, ring, hyb×3


# ---------------------------------------------------------------------------
# Residue featurization
# ---------------------------------------------------------------------------

def residue_features_from_chain(chain) -> Tuple[Tensor, Tensor]:
    """
    Build per-residue feature tensor from a BioPython chain.

    Returns:
        features : (L, RESIDUE_FEATURE_DIM)  float32
        mask     : (L,)                        bool
    """
    from Bio.PDB.Polypeptide import PPBuilder  # type: ignore

    ppb = PPBuilder()
    poly_list = ppb.build_peptides(chain)

    features_list = []
    mask_list = []

    for poly in poly_list:
        phi_psi = poly.get_phi_psi_list()
        seq = poly.get_sequence()

        for i, (aa_char, (phi, psi)) in enumerate(zip(str(seq), phi_psi)):
            res_name = _one_to_three(aa_char)
            aa_onehot = _aa_onehot(res_name)           # (21,)
            torsion   = _torsion_feats(phi, psi, None) # (6,)
            features_list.append(torch.cat([aa_onehot, torsion]))
            mask_list.append(True)

    if not features_list:
        # Fallback: unknown single residue
        features_list = [torch.zeros(RESIDUE_FEATURE_DIM)]
        mask_list = [False]

    features = torch.stack(features_list).float()
    mask     = torch.tensor(mask_list, dtype=torch.bool)
    return features, mask


def _aa_onehot(res_name: str) -> Tensor:
    idx = AA_TO_IDX.get(res_name.upper(), AA_TO_IDX["UNK"])
    v = torch.zeros(21)
    v[idx] = 1.0
    return v


def _torsion_feats(phi, psi, omega) -> Tensor:
    """Convert three backbone torsions to sin/cos features (6-dim)."""
    def sc(angle):
        if angle is None:
            return 0.0, 0.0
        return math.sin(angle), math.cos(angle)

    s_phi, c_phi = sc(phi)
    s_psi, c_psi = sc(psi)
    s_omg, c_omg = sc(omega)
    return torch.tensor([s_phi, c_phi, s_psi, c_psi, s_omg, c_omg], dtype=torch.float32)


_THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}
_ONE_TO_THREE = {v: k for k, v in _THREE_TO_ONE.items()}


def _one_to_three(one: str) -> str:
    return _ONE_TO_THREE.get(one.upper(), "UNK")


# ---------------------------------------------------------------------------
# Ligand featurization
# ---------------------------------------------------------------------------

def ligand_features_from_sdf(sdf_path: str) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Featurize a ligand from an SDF file using RDKit.

    Returns:
        coords   : (N, 3) float32
        features : (N, LIGAND_FEATURE_DIM) float32
        mask     : (N,)  bool
    """
    from rdkit import Chem          # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mol = next(iter(supplier))
    if mol is None:
        raise ValueError(f"Could not parse ligand SDF: {sdf_path}")

    conf = mol.GetConformer()
    positions = conf.GetPositions()   # (N, 3)  numpy

    feats = []
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        idx = ELEM_TO_IDX.get(elem, ELEM_TO_IDX["other"])

        elem_oh = torch.zeros(11)
        elem_oh[idx] = 1.0

        charge     = float(atom.GetFormalCharge())
        aromatic   = float(atom.GetIsAromatic())
        in_ring    = float(atom.IsInRing())
        hyb        = _hybridisation_onehot(atom)

        feat = torch.cat([elem_oh, torch.tensor([charge, aromatic, in_ring]), hyb])
        feats.append(feat)

    features = torch.stack(feats).float()
    coords   = torch.from_numpy(positions).float()
    mask     = torch.ones(coords.shape[0], dtype=torch.bool)

    return coords, features, mask


def _hybridisation_onehot(atom) -> Tensor:
    """Three-hot encoding: SP, SP2, SP3."""
    from rdkit.Chem import rdchem  # type: ignore
    hyb = atom.GetHybridization()
    v = torch.zeros(3)
    if hyb == rdchem.HybridizationType.SP:
        v[0] = 1.0
    elif hyb == rdchem.HybridizationType.SP2:
        v[1] = 1.0
    elif hyb == rdchem.HybridizationType.SP3:
        v[2] = 1.0
    return v
