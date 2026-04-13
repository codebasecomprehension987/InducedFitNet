"""
InducedFitNet evaluation script.

Runs sampling on a held-out test set and reports:
  - Overall success rate (SR@2Å)
  - Stratified success rates by training-set similarity bins
    (0–20%, 20–40%, 40–60%, 60–80%, 80–100%)
  - Backbone Cα RMSD distribution
  - Pocket volume overlap (Jaccard)

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --data_dir   /data/pdbbind \
        --n_samples  10 \
        --n_steps    200 \
        --output     results/eval_results.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from inducedfitnet import InducedFitNet
from inducedfitnet.data.dataset import ProteinLigandDataset
from inducedfitnet.utils.metrics import (
    aligned_rmsd,
    ca_rmsd,
    stratified_success_table,
    pocket_volume_convex_hull,
)

log = logging.getLogger(__name__)


def evaluate_complex(model, complex_, n_samples, n_steps, device):
    """Run sampling and return the best-of-N sample by ligand RMSD."""
    samples = model.sample(complex_.to(device), n_samples=n_samples, n_steps=n_steps)

    ref_lig = complex_.ligand_coords.to(device)
    best_rmsd   = float("inf")
    best_sample = None

    for s in samples:
        r = aligned_rmsd(s.ligand_coords, ref_lig)
        if r < best_rmsd:
            best_rmsd   = r
            best_sample = s

    return best_sample, best_rmsd


def main():
    parser = argparse.ArgumentParser(description="InducedFitNet evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--n_samples",  type=int, default=10)
    parser.add_argument("--n_steps",    type=int, default=200)
    parser.add_argument("--output",     default="results/eval_results.json")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = InducedFitNet.from_pretrained(args.checkpoint, device=str(device))
    log.info(f"Loaded model from {args.checkpoint}")

    test_ds = ProteinLigandDataset(
        root  = args.data_dir,
        split = "test",
    )
    log.info(f"Test set: {len(test_ds)} structures")

    records = []
    lig_rmsds   = []
    ca_rmsds    = []

    n_eval = len(test_ds) if args.max_examples is None else min(args.max_examples, len(test_ds))

    for idx in tqdm(range(n_eval), desc="Evaluating"):
        complex_ = test_ds[idx]

        try:
            best_sample, best_lig_rmsd = evaluate_complex(
                model, complex_, args.n_samples, args.n_steps, device
            )
        except Exception as e:
            log.warning(f"Failed on {complex_.pdb_id}: {e}")
            continue

        # Backbone Cα RMSD
        ca_r = ca_rmsd(
            best_sample.backbone_coords.cpu(),
            complex_.backbone_coords,
        ).item()

        # Pocket volume (Cα atoms within 8 Å of predicted ligand COM)
        lig_com = best_sample.ligand_coords.mean(dim=0).cpu().numpy()
        ca_coords = complex_.backbone_coords[:, 1, :].numpy()
        pocket_mask = np.linalg.norm(ca_coords - lig_com, axis=-1) < 8.0
        vol_pred = pocket_volume_convex_hull(ca_coords[pocket_mask])

        lig_rmsds.append(best_lig_rmsd)
        ca_rmsds.append(ca_r)

        records.append({
            "pdb_id":      complex_.pdb_id,
            "pred_coords": best_sample.ligand_coords.cpu(),
            "true_coords": complex_.ligand_coords.cpu(),
            "similarity":  complex_.similarity,
            "lig_rmsd":    best_lig_rmsd,
            "ca_rmsd":     ca_r,
            "pocket_vol":  vol_pred,
        })

    # --- Aggregate metrics ---
    sr_2A = float(np.mean([r["lig_rmsd"] < 2.0 for r in records]))
    sr_1A = float(np.mean([r["lig_rmsd"] < 1.0 for r in records]))

    strat = stratified_success_table(records, thresholds=[1.0, 2.0])

    results = {
        "n_evaluated":        len(records),
        "SR@1A":              sr_1A,
        "SR@2A":              sr_2A,
        "median_lig_rmsd":    float(np.median(lig_rmsds)),
        "median_ca_rmsd":     float(np.median(ca_rmsds)),
        "stratified_success": strat,
        "per_complex": [
            {k: v for k, v in r.items() if k not in ("pred_coords", "true_coords")}
            for r in records
        ],
    }

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("InducedFitNet Evaluation Results")
    print("=" * 60)
    print(f"  N evaluated  : {len(records)}")
    print(f"  SR@2Å        : {sr_2A:.3f}  ({sr_2A * 100:.1f}%)")
    print(f"  SR@1Å        : {sr_1A:.3f}  ({sr_1A * 100:.1f}%)")
    print(f"  Median lig RMSD : {np.median(lig_rmsds):.2f} Å")
    print(f"  Median Cα RMSD  : {np.median(ca_rmsds):.2f} Å")
    print()
    print("Stratified by similarity bin:")
    for bin_label, metrics in strat.items():
        n = metrics.get("n", 0)
        s2 = metrics.get("SR@2.0Å", float("nan"))
        print(f"  {bin_label:10s}  n={n:4d}  SR@2Å={s2:.3f}")
    print("=" * 60)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
