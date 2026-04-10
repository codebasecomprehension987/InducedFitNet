"""
ProteinLigandComplex — core data container.

Holds receptor backbone coordinates, residue features, ligand atom
coordinates and features, and metadata required for diffusion.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor


@dataclasses.dataclass
class ProteinLigandComplex:
    """
    A receptor + ligand pair ready for diffusion.

    Attributes
    ----------
    backbone_coords : (L, 4, 3)   N, Cα, C, O per residue (float32)
    residue_features: (L, D_res)  per-residue features (amino-acid type, DSSP, …)
    residue_mask    : (L,)        True = valid residue
    ligand_coords   : (N_atm, 3) heavy-atom positions (float32)
    ligand_features : (N_atm, D_atm)
    ligand_mask     : (N_atm,)
    pdb_id          : PDB accession, e.g. "8EA6"
    chain_id        : receptor chain
    similarity      : sequence similarity to nearest training example [0, 100]
    """

    backbone_coords:  Tensor                   # (L, 4, 3)
    residue_features: Tensor                   # (L, D_res)
    residue_mask:     Tensor                   # (L,) bool
    ligand_coords:    Tensor                   # (N_atm, 3)
    ligand_features:  Tensor                   # (N_atm, D_atm)
    ligand_mask:      Tensor                   # (N_atm,) bool
    pdb_id:           str = ""
    chain_id:         str = "A"
    similarity:       float = 100.0            # % identity to training set

    # -----------------------------------------------------------------------
    # Factory helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str | Path,
        ligand_sdf: Optional[str | Path] = None,
        chain_id: str = "A",
        pdb_id: Optional[str] = None,
    ) -> "ProteinLigandComplex":
        """
        Parse a PDB file (+ optional SDF ligand) into a ProteinLigandComplex.

        Residue features are one-hot amino acid type (20 dims) + 1 unknown dim.
        Ligand features are one-hot atomic number (H,C,N,O,F,P,S,Cl,Br,I + other).
        """
        from Bio.PDB import PDBParser  # type: ignore
        from inducedfitnet.data.featurizer import (
            residue_features_from_chain,
            ligand_features_from_sdf,
        )

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("mol", str(pdb_path))
        model = next(iter(structure))

        # Extract backbone atoms for the specified chain
        chain = model[chain_id]
        bb_list: List[np.ndarray] = []
        for residue in chain.get_residues():
            if residue.get_id()[0] != " ":
                continue  # skip HETATM
            try:
                n  = residue["N"].get_vector().get_array()
                ca = residue["CA"].get_vector().get_array()
                c  = residue["C"].get_vector().get_array()
                o  = residue["O"].get_vector().get_array()
                bb_list.append(np.stack([n, ca, c, o]))
            except KeyError:
                pass  # incomplete residue — skip

        backbone_coords = torch.from_numpy(np.stack(bb_list)).float()  # (L, 4, 3)
        L = backbone_coords.shape[0]

        res_feat, res_mask = residue_features_from_chain(chain)

        # Ligand
        if ligand_sdf is not None:
            lig_coords, lig_feat, lig_mask = ligand_features_from_sdf(str(ligand_sdf))
        else:
            # Placeholder: single-atom ligand at origin
            lig_coords = torch.zeros(1, 3)
            lig_feat   = torch.zeros(1, 11)
            lig_mask   = torch.ones(1, dtype=torch.bool)

        return cls(
            backbone_coords  = backbone_coords,
            residue_features = res_feat,
            residue_mask     = res_mask,
            ligand_coords    = lig_coords,
            ligand_features  = lig_feat,
            ligand_mask      = lig_mask,
            pdb_id           = pdb_id or Path(pdb_path).stem,
            chain_id         = chain_id,
        )

    # -----------------------------------------------------------------------
    # Batch helpers
    # -----------------------------------------------------------------------

    def to(self, device: str | torch.device) -> "ProteinLigandComplex":
        """Move all tensors to device."""
        return dataclasses.replace(
            self,
            backbone_coords  = self.backbone_coords.to(device),
            residue_features = self.residue_features.to(device),
            residue_mask     = self.residue_mask.to(device),
            ligand_coords    = self.ligand_coords.to(device),
            ligand_features  = self.ligand_features.to(device),
            ligand_mask      = self.ligand_mask.to(device),
        )

    @staticmethod
    def collate(batch: List["ProteinLigandComplex"]) -> "ProteinLigandComplex":
        """
        Pad and stack a list of complexes into a batched ProteinLigandComplex.

        All tensors gain a leading batch dimension B.
        """
        import torch.nn.functional as F

        L_max   = max(c.backbone_coords.shape[0] for c in batch)
        Na_max  = max(c.ligand_coords.shape[0] for c in batch)

        bb_list, rf_list, rm_list = [], [], []
        lc_list, lf_list, lm_list = [], [], []

        for c in batch:
            L  = c.backbone_coords.shape[0]
            Na = c.ligand_coords.shape[0]

            bb_list.append(F.pad(c.backbone_coords,  (0,0,0,0,0, L_max-L)))
            rf_list.append(F.pad(c.residue_features, (0,0,0, L_max-L)))
            rm_list.append(F.pad(c.residue_mask.float(), (0, L_max-L)).bool())

            lc_list.append(F.pad(c.ligand_coords,   (0,0,0, Na_max-Na)))
            lf_list.append(F.pad(c.ligand_features, (0,0,0, Na_max-Na)))
            lm_list.append(F.pad(c.ligand_mask.float(), (0, Na_max-Na)).bool())

        return ProteinLigandComplex(
            backbone_coords  = torch.stack(bb_list),
            residue_features = torch.stack(rf_list),
            residue_mask     = torch.stack(rm_list),
            ligand_coords    = torch.stack(lc_list),
            ligand_features  = torch.stack(lf_list),
            ligand_mask      = torch.stack(lm_list),
        )

    # -----------------------------------------------------------------------
    # Shape properties
    # -----------------------------------------------------------------------

    @property
    def n_residues(self) -> int:
        return int(self.residue_mask.sum().item())

    @property
    def n_ligand_atoms(self) -> int:
        return int(self.ligand_mask.sum().item())

    def __repr__(self) -> str:
        return (
            f"ProteinLigandComplex(pdb={self.pdb_id!r}, "
            f"L={self.n_residues}, N_lig={self.n_ligand_atoms}, "
            f"sim={self.similarity:.1f}%)"
        )
