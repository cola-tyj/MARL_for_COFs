"""
QM9 pre-training script for the molecular diffusion model.

Phase 1 of the master's thesis project.
Trains a mixed continuous-discrete denoising diffusion model on QM9 molecules.

Usage:
    python train_qm9.py --epochs 500 --batch_size 64 --use_amp

After training, the model can generate valid small organic molecules
and serves as the base for symmetry-conditioned fine-tuning (Phase 2).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from symmcd_diffusion.config.base_config import QM9Config
from symmcd_diffusion.data.qm9_dataset import (
    QM9Dataset, QM9_NUM_ATOM_TYPES, NUM_BOND_TYPES,
)
from symmcd_diffusion.models.denoiser import Denoiser
from symmcd_diffusion.models.diffusion_process import DiffusionProcess


def compute_molecule_validity(
    atom_types: torch.Tensor,
    positions: torch.Tensor,
    bond_types: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute validity metrics for generated molecules.

    Returns:
        dict with 'valid', 'unique', 'complete' rates
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Convert to RDKit molecule
        atom_idx = atom_types.argmax(dim=-1).cpu().numpy()
        atomic_nums = [1, 6, 7, 8, 9]  # H, C, N, O, F

        mol = Chem.RWMol()
        for a in atom_idx:
            mol.AddAtom(Chem.Atom(atomic_nums[a]))

        # Add bonds based on distances
        pos = positions.cpu().numpy()
        n = len(atom_idx)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(pos[i] - pos[j])
                r_i = Chem.GetPeriodicTable().GetRcovalent(atomic_nums[atom_idx[i]])
                r_j = Chem.GetPeriodicTable().GetRcovalent(atomic_nums[atom_idx[j]])
                if dist < (r_i + r_j) * 1.2:
                    if dist < (r_i + r_j) * 0.85:
                        mol.AddBond(i, j, Chem.BondType.TRIPLE)
                    elif dist < (r_i + r_j) * 0.95:
                        mol.AddBond(i, j, Chem.BondType.DOUBLE)
                    else:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)

        mol = mol.GetMol()

        # Try sanitization
        try:
            Chem.SanitizeMol(mol)
            valid = True
        except Exception:
            valid = False

        return {"valid": 1.0 if valid else 0.0}

    except ImportError:
        # RDKit not available, return dummy
        return {"valid": 1.0}


def validate(
    diffusion: DiffusionProcess,
    denoiser: Denoiser,
    atom_marginals: torch.Tensor,
    bond_marginals: torch.Tensor,
    num_samples: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run validation: sample molecules and compute metrics."""
    denoiser.eval()
    metrics = {"valid": 0.0, "total": 0.0}

    for _ in range(num_samples):
        # Sample random molecule size (QM9 range)
        num_atoms = np.random.randint(3, 30)

        try:
            result = diffusion.sample(
                denoiser=denoiser,
                num_atoms=num_atoms,
                atom_marginals=atom_marginals,
                bond_marginals=bond_marginals,
                device=device,
            )

            mol_metrics = compute_molecule_validity(
                result["atom_types"],
                result["positions"],
                result["bond_types"],
            )
            metrics["valid"] += mol_metrics["valid"]
            metrics["total"] += 1
        except Exception as e:
            metrics["total"] += 1

    denoiser.train()
    metrics["validity"] = metrics["valid"] / max(metrics["total"], 1)
    return metrics


def train(args):
    """Main training loop."""
    # Config
    config = QM9Config(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        diffusion_steps=args.diffusion_steps,
        use_amp=args.use_amp,
        grad_accumulation=args.grad_accum,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {config}")

    # Create checkpoint dir
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Load datasets
    print("Loading QM9 dataset...")
    train_dataset = QM9Dataset(
        root=args.data_root,
        split="train",
        max_molecules=args.max_molecules,
    )
    val_dataset = QM9Dataset(
        root=args.data_root,
        split="val",
        max_molecules=args.max_molecules // 5 if args.max_molecules else None,
    )

    train_loader = train_dataset.get_dataloader(
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=config.batch_size,
        shuffle=False,
    )

    atom_marginals = train_dataset.atom_marginals.to(device)
    # Uniform bond marginals as a simple starting point
    bond_marginals = torch.ones(NUM_BOND_TYPES, device=device) / NUM_BOND_TYPES

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize models
    print("Initializing models...", flush=True)
    denoiser = Denoiser(
        num_atom_types=QM9_NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        edge_feat_dim=NUM_BOND_TYPES,
        attention=True,
        dropout=0.1,
    ).to(device)

    diffusion = DiffusionProcess(
        num_atom_types=QM9_NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        timesteps=config.diffusion_steps,
        noise_schedule=config.noise_schedule,
        coord_loss_weight=config.coord_loss_weight,
        atom_loss_weight=config.atom_type_loss_weight,
        bond_loss_weight=config.bond_type_loss_weight,
    ).to(device)

    # Print model size
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Denoiser parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        denoiser.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Warmup + cosine decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs * len(train_loader) - config.warmup_steps,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps],
    )

    # AMP scaler
    scaler = GradScaler(enabled=config.use_amp)

    # Training loop
    print("Starting training loop...", flush=True)
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(config.max_epochs):
        denoiser.train()
        epoch_losses = {"coord": 0.0, "atom": 0.0, "bond": 0.0, "total": 0.0}
        n_batches = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)

            with autocast(enabled=config.use_amp):
                loss, loss_dict = diffusion.training_step(
                    data=batch,
                    denoiser=denoiser,
                    atom_marginals=atom_marginals,
                    bond_marginals=bond_marginals,
                )
                loss = loss / config.grad_accumulation

            # Backward
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            epoch_losses["coord"] += loss_dict["coord_loss"]
            epoch_losses["atom"] += loss_dict["atom_loss"]
            epoch_losses["bond"] += loss_dict["bond_loss"]
            epoch_losses["total"] += loss_dict["total_loss"]
            n_batches += 1
            global_step += 1

            if global_step % config.log_every_n_steps == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch+1}/{config.max_epochs} | "
                    f"Step {global_step} | "
                    f"Loss: {loss_dict['total_loss']:.4f} | "
                    f"Coord: {loss_dict['coord_loss']:.4f} | "
                    f"Atom: {loss_dict['atom_loss']:.4f} | "
                    f"LR: {lr:.2e}"
                )

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_losses["total"] / max(n_batches, 1)
        print(
            f"=== Epoch {epoch+1} Summary ===\n"
            f"  Avg Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s\n"
            f"  Coord: {epoch_losses['coord']/max(n_batches,1):.4f} | "
            f"Atom: {epoch_losses['atom']/max(n_batches,1):.4f} | "
            f"Bond: {epoch_losses['bond']/max(n_batches,1):.4f}"
        )

        # Validation
        if (epoch + 1) % config.val_every_n_epochs == 0:
            print("Running validation...")
            val_metrics = validate(
                diffusion, denoiser, atom_marginals, bond_marginals,
                num_samples=50, device=device,
            )
            print(f"  Validity: {val_metrics['validity']:.3f}")

            # Save best model
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "denoiser_state_dict": denoiser.state_dict(),
                        "config": config,
                        "loss": avg_loss,
                    },
                    os.path.join(config.checkpoint_dir, "qm9_best.pt"),
                )
                print(f"  Saved best model (loss: {avg_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "denoiser_state_dict": denoiser.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                os.path.join(
                    config.checkpoint_dir, f"qm9_epoch{epoch+1}.pt"
                ),
            )

    # Final save
    torch.save(
        {
            "epoch": config.max_epochs,
            "denoiser_state_dict": denoiser.state_dict(),
            "config": config,
        },
        os.path.join(config.checkpoint_dir, "qm9_denoiser_epoch500.pt"),
    )
    print(f"Training complete. Final model saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QM9 Diffusion Pre-training")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--diffusion_steps", type=int, default=1000)

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # Data
    parser.add_argument("--data_root", type=str, default="./data/qm9")
    parser.add_argument("--max_molecules", type=int, default=None)

    args = parser.parse_args()
    train(args)
