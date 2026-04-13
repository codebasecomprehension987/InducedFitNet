"""
InducedFitNet training script.

Trains the JointScoreNetwork by:
  1. Drawing (x_0, R_0, tr_0) from the dataset.
  2. Sampling a random timestep t and noise level σ(t).
  3. Noising the backbone frames and ligand coordinates.
  4. Running the score network to predict scores.
  5. Computing the weighted denoising score-matching loss.

Loss:
    L = λ_R  · ‖score_R_pred  − score_R_true‖²
      + λ_tr · ‖score_tr_pred − score_tr_true‖²
      + λ_x  · ‖score_x_pred  − score_x_true‖²

Usage:
    python scripts/train.py --config configs/train_default.yaml \
                            --data_dir /data/pdbbind \
                            --output_dir checkpoints/
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from inducedfitnet.data.complex import ProteinLigandComplex
from inducedfitnet.data.dataset import ProteinLigandDataset
from inducedfitnet.diffusion.r3_diffusion import R3Diffusion
from inducedfitnet.diffusion.se3_diffusion import SE3Diffusion
from inducedfitnet.models.score_network import JointScoreNetwork
from inducedfitnet.utils.frames import backbone_frames

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def score_matching_loss(
    score_R_pred:  torch.Tensor,   # (B, L, 3)
    score_R_true:  torch.Tensor,
    score_tr_pred: torch.Tensor,   # (B, L, 3)
    score_tr_true: torch.Tensor,
    score_x_pred:  torch.Tensor,   # (B, N, 3)
    score_x_true:  torch.Tensor,
    res_mask:      torch.Tensor,   # (B, L)
    lig_mask:      torch.Tensor,   # (B, N)
    sigma_r:       torch.Tensor,   # (B,)
    sigma_tr:      torch.Tensor,
    sigma_x:       torch.Tensor,
    lambda_R:  float = 1.0,
    lambda_tr: float = 0.5,
    lambda_x:  float = 1.0,
) -> torch.Tensor:
    """Weighted per-element score-matching MSE loss."""

    def masked_mse(pred, true, mask, sigma):
        # Weight by σ² so that all timesteps contribute equally
        w = sigma.pow(2).view(-1, 1, 1)
        sq = ((pred - true) ** 2).sum(dim=-1)   # (B, *)
        sq = sq * mask.float()
        n  = mask.float().sum(dim=-1).clamp(min=1)  # (B,) or scalar
        return (w.squeeze(-1) * sq / n.unsqueeze(-1)).mean()

    L_R  = masked_mse(score_R_pred,  score_R_true,  res_mask, sigma_r)
    L_tr = masked_mse(score_tr_pred, score_tr_true, res_mask, sigma_tr)
    L_x  = masked_mse(score_x_pred,  score_x_true,  lig_mask, sigma_x)

    return lambda_R * L_R + lambda_tr * L_tr + lambda_x * L_x


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model:       JointScoreNetwork,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    se3_diff:    SE3Diffusion,
    r3_diff:     R3Diffusion,
    device:      torch.device,
    cfg,
    epoch:       int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        batch: ProteinLigandComplex = batch.to(device)

        B = batch.backbone_coords.shape[0]
        L = batch.backbone_coords.shape[1]

        # Build backbone frames from N, Cα, C
        n_c  = batch.backbone_coords[:, :, 0, :]
        ca_c = batch.backbone_coords[:, :, 1, :]
        c_c  = batch.backbone_coords[:, :, 2, :]
        R_0, tr_0 = backbone_frames(n_c, ca_c, c_c)   # (B,L,3,3) / (B,L,3)

        # Sample random timesteps
        t = torch.randint(0, se3_diff.T, (B,), device=device)

        # Forward diffusion
        R_t, tr_t  = se3_diff.q_sample(R_0, tr_0, t)
        x_t, eps_x = r3_diff.q_sample(batch.ligand_coords, t, batch.ligand_mask)

        # Ground-truth scores
        score_R_true  = se3_diff.rotation_score(R_t, R_0, t)
        score_tr_true = se3_diff.translation_score(tr_t, tr_0, t)
        score_x_true  = r3_diff.score(x_t, batch.ligand_coords, t)

        # Predicted scores
        score_R_pred, score_tr_pred, score_x_pred = model(
            R         = R_t,
            tr        = tr_t,
            x_lig     = x_t,
            res_feat  = batch.residue_features,
            lig_feat  = batch.ligand_features,
            timestep  = t,
            res_mask  = batch.residue_mask,
            lig_mask  = batch.ligand_mask,
        )

        sigma_r  = se3_diff.schedule.sigma(t)
        sigma_tr = se3_diff.trans_schedule.sigma(t)
        sigma_x  = r3_diff.schedule.sigma(t)

        loss = score_matching_loss(
            score_R_pred, score_R_true,
            score_tr_pred, score_tr_true,
            score_x_pred, score_x_true,
            batch.residue_mask, batch.ligand_mask,
            sigma_r, sigma_tr, sigma_x,
            lambda_R  = cfg.training.lambda_R,
            lambda_tr = cfg.training.lambda_tr,
            lambda_x  = cfg.training.lambda_x,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train InducedFitNet")
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--resume",     default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on {device}")

    # Data
    train_ds = ProteinLigandDataset(
        root         = args.data_dir,
        split        = "train",
        max_residues = cfg.data.max_residues,
    )
    val_ds = ProteinLigandDataset(
        root         = args.data_dir,
        split        = "val",
        max_residues = cfg.data.max_residues,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.training.batch_size,
        shuffle     = True,
        num_workers = cfg.training.num_workers,
        collate_fn  = ProteinLigandComplex.collate,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.training.batch_size,
        shuffle     = False,
        num_workers = cfg.training.num_workers,
        collate_fn  = ProteinLigandComplex.collate,
    )

    # Model + diffusion
    model    = JointScoreNetwork(cfg.model).to(device)
    se3_diff = SE3Diffusion(cfg.diffusion)
    r3_diff  = R3Diffusion(cfg.diffusion)
    log.info(f"Model: {model}")

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.training.lr,
        weight_decay = cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.n_epochs
    )

    # Resume
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info(f"Resumed from epoch {start_epoch}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional W&B
    try:
        import wandb
        wandb.init(project="inducedfitnet", config=OmegaConf.to_container(cfg))
    except Exception:
        wandb = None

    best_val_loss = float("inf")
    for epoch in range(start_epoch, cfg.training.n_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, se3_diff, r3_diff, device, cfg, epoch
        )
        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                B = batch.backbone_coords.shape[0]
                t = torch.randint(0, se3_diff.T, (B,), device=device)
                n_c  = batch.backbone_coords[:, :, 0, :]
                ca_c = batch.backbone_coords[:, :, 1, :]
                c_c  = batch.backbone_coords[:, :, 2, :]
                R_0, tr_0 = backbone_frames(n_c, ca_c, c_c)
                R_t,  tr_t  = se3_diff.q_sample(R_0, tr_0, t)
                x_t,  _     = r3_diff.q_sample(batch.ligand_coords, t, batch.ligand_mask)
                sR, str_, sx = model(
                    R_t, tr_t, x_t,
                    batch.residue_features, batch.ligand_features,
                    t, batch.residue_mask, batch.ligand_mask,
                )
                vl = score_matching_loss(
                    sR, se3_diff.rotation_score(R_t, R_0, t),
                    str_, se3_diff.translation_score(tr_t, tr_0, t),
                    sx, r3_diff.score(x_t, batch.ligand_coords, t),
                    batch.residue_mask, batch.ligand_mask,
                    se3_diff.schedule.sigma(t),
                    se3_diff.trans_schedule.sigma(t),
                    r3_diff.schedule.sigma(t),
                )
                val_losses.append(vl.item())
        val_loss = float(np.mean(val_losses))

        log.info(f"Epoch {epoch:04d}  train={train_loss:.4f}  val={val_loss:.4f}")
        if wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        # Checkpoint
        ckpt_data = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config":               OmegaConf.to_container(cfg),
            "val_loss":             val_loss,
        }
        torch.save(ckpt_data, output_dir / "latest.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_data, output_dir / "best.pt")
            log.info(f"  ✓ New best checkpoint (val={val_loss:.4f})")

    log.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
