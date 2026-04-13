"""
InducedFitNet sampling script.

Generates predicted protein–ligand complex conformations from
a trained checkpoint and an input receptor PDB + ligand SDF.

Usage:
    python scripts/sample.py \
        --checkpoint checkpoints/best.pt \
        --receptor   inputs/8EA6_apo.pdb \
        --ligand     inputs/8EA6_ligand.sdf \
        --n_samples  10 \
        --n_steps    200 \
        --output_dir outputs/8EA6/
"""

import argparse
import logging
from pathlib import Path

import torch

from inducedfitnet import InducedFitNet
from inducedfitnet.data.complex import ProteinLigandComplex

log = logging.getLogger(__name__)


def save_ligand_sdf(coords, template_sdf, out_path):
    """Write sampled ligand coordinates back into an SDF file."""
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        supplier = Chem.SDMolSupplier(template_sdf, removeHs=True)
        mol = next(iter(supplier))
        conf = mol.GetConformer()
        for i, (x, y, z) in enumerate(coords.tolist()):
            conf.SetAtomPosition(i, Point3D(x, y, z))
        writer = Chem.SDWriter(str(out_path))
        writer.write(mol)
        writer.close()
    except Exception as e:
        log.warning(f"Could not write SDF: {e}. Saving as NumPy instead.")
        import numpy as np
        np.save(str(out_path).replace(".sdf", ".npy"), coords.cpu().numpy())


def save_backbone_pdb(bb_coords, template_pdb, out_path):
    """Write sampled backbone Cα coordinates to a minimal PDB."""
    lines = []
    for i, (n, ca, c, o) in enumerate(bb_coords.tolist(), start=1):
        for atom_name, pos in zip(["N", "CA", "C", "O"], [n, ca, c, o]):
            lines.append(
                f"ATOM  {i:5d}  {atom_name:<3s} ALA A{i:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {atom_name[0]}\n"
            )
    with open(out_path, "w") as f:
        f.writelines(lines)
        f.write("END\n")


def main():
    parser = argparse.ArgumentParser(description="InducedFitNet sampling")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--receptor",    required=True, help="Apo receptor PDB")
    parser.add_argument("--ligand",      required=True, help="Ligand SDF (template geometry)")
    parser.add_argument("--chain_id",    default="A")
    parser.add_argument("--n_samples",   type=int, default=10)
    parser.add_argument("--n_steps",     type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir",  default="outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Sampling on {device}")

    # Load model
    model = InducedFitNet.from_pretrained(args.checkpoint, device=device)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # Load input complex
    complex_ = ProteinLigandComplex.from_pdb(
        pdb_path   = args.receptor,
        ligand_sdf = args.ligand,
        chain_id   = args.chain_id,
    )
    log.info(f"Input: {complex_}")

    # Sample
    log.info(f"Sampling {args.n_samples} conformations × {args.n_steps} steps …")
    samples = model.sample(
        complex_    = complex_,
        n_samples   = args.n_samples,
        n_steps     = args.n_steps,
        temperature = args.temperature,
    )

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        lig_path = out_dir / f"sample_{i:03d}_ligand.sdf"
        bb_path  = out_dir / f"sample_{i:03d}_backbone.pdb"
        save_ligand_sdf(sample.ligand_coords, args.ligand, lig_path)
        save_backbone_pdb(sample.backbone_coords, args.receptor, bb_path)
        log.info(f"  Saved sample {i:03d} → {lig_path.name}, {bb_path.name}")

    log.info(f"Done. {len(samples)} samples written to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
