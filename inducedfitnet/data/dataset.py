"""
InducedFitNet dataset — wraps PDBBind or any PDB/SDF collection.

Each item is a ProteinLigandComplex, pre-featurized and cached to disk
as a .pt file for fast loading during training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset

from inducedfitnet.data.complex import ProteinLigandComplex

log = logging.getLogger(__name__)


class ProteinLigandDataset(Dataset):
    """
    Dataset of ProteinLigandComplex objects.

    Directory layout expected::

        root/
          <pdb_id>/
            receptor.pdb
            ligand.sdf
            metadata.json   (optional, for similarity annotation)

    Caches featurized complexes to ``root/cache/<pdb_id>.pt``.

    Args:
        root:           Path to dataset root.
        split:          "train", "val", or "test".
        split_file:     Path to a text file with one PDB ID per line.
        chain_id:       Receptor chain to parse (default "A").
        max_residues:   Crop receptor to this length (0 = no crop).
        transform:      Optional callable applied to each complex.
        rebuild_cache:  If True, ignore existing cache files.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        split_file: Optional[str | Path] = None,
        chain_id: str = "A",
        max_residues: int = 512,
        transform: Optional[Callable] = None,
        rebuild_cache: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.chain_id = chain_id
        self.max_residues = max_residues
        self.transform = transform
        self.rebuild_cache = rebuild_cache

        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.pdb_ids = self._load_split(split_file)
        log.info(f"[{split}] {len(self.pdb_ids)} structures loaded from {root}")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, idx: int) -> ProteinLigandComplex:
        pdb_id = self.pdb_ids[idx]
        cache_path = self.cache_dir / f"{pdb_id}.pt"

        if cache_path.exists() and not self.rebuild_cache:
            complex_ = torch.load(cache_path, weights_only=False)
        else:
            complex_ = self._build_complex(pdb_id)
            torch.save(complex_, cache_path)

        # Crop to max_residues
        if self.max_residues > 0:
            complex_ = self._crop(complex_)

        if self.transform is not None:
            complex_ = self.transform(complex_)

        return complex_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_split(self, split_file: Optional[Path]) -> List[str]:
        if split_file is not None and Path(split_file).exists():
            with open(split_file) as f:
                return [line.strip() for line in f if line.strip()]

        # Auto-discover by directory
        candidates = sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and d.name != "cache"
        )

        # Simple 80/10/10 split by sorted order if no file given
        n = len(candidates)
        splits = {
            "train": candidates[:int(0.8 * n)],
            "val":   candidates[int(0.8 * n):int(0.9 * n)],
            "test":  candidates[int(0.9 * n):],
        }
        return splits.get(self.split, candidates)

    def _build_complex(self, pdb_id: str) -> ProteinLigandComplex:
        base = self.root / pdb_id
        pdb_path = base / "receptor.pdb"
        sdf_path = base / "ligand.sdf"

        if not pdb_path.exists():
            raise FileNotFoundError(f"receptor.pdb not found for {pdb_id}")

        sdf_arg = str(sdf_path) if sdf_path.exists() else None
        complex_ = ProteinLigandComplex.from_pdb(
            pdb_path=str(pdb_path),
            ligand_sdf=sdf_arg,
            chain_id=self.chain_id,
            pdb_id=pdb_id,
        )

        # Attach similarity from metadata if available
        meta_path = base / "metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            complex_ = __import__("dataclasses").replace(
                complex_, similarity=float(meta.get("similarity", 100.0))
            )

        return complex_

    def _crop(self, c: ProteinLigandComplex) -> ProteinLigandComplex:
        """Crop receptor to max_residues residues centred on the ligand."""
        import dataclasses

        L = c.backbone_coords.shape[0]
        if L <= self.max_residues:
            return c

        # Centre of mass of ligand
        lig_com = c.ligand_coords.mean(dim=0)   # (3,)
        ca      = c.backbone_coords[:, 1, :]    # (L, 3)  Cα atoms

        dists  = (ca - lig_com.unsqueeze(0)).norm(dim=-1)  # (L,)
        center = int(dists.argmin().item())
        half   = self.max_residues // 2
        lo     = max(0, center - half)
        hi     = min(L, lo + self.max_residues)
        lo     = max(0, hi - self.max_residues)

        return dataclasses.replace(
            c,
            backbone_coords  = c.backbone_coords[lo:hi],
            residue_features = c.residue_features[lo:hi],
            residue_mask     = c.residue_mask[lo:hi],
        )
