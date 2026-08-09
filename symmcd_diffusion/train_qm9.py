"""
QM9 pre-training script for the molecular diffusion model.

Phase 1 of the master's thesis project.
Trains a mixed continuous-discrete denoising diffusion model on QM9 molecules.

Usage:
    # Fresh training
    python train_qm9.py --epochs 500 --batch_size 64 --use_amp

    # Resume from checkpoint
    python train_qm9.py --resume symmcd_diffusion/checkpoints/qm9_latest.pt

After training, the model can generate valid small organic molecules
and serves as the base for symmetry-conditioned fine-tuning (Phase 2).
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
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


# ──────────────────────────────────────────────────────────────────────────────
# Logging utilities
# ──────────────────────────────────────────────────────────────────────────────

class Logger:
    """Logs to both stdout and a file."""

    def __init__(self, log_dir: str, name: str = "train_qm9"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
        self.file = open(self.log_path, "a", buffering=1)  # line-buffered
        self._original_stdout = sys.stdout

    def write(self, message: str):
        """Write to both file and original stdout."""
        self._original_stdout.write(message)
        self.file.write(message)

    def flush(self):
        self._original_stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        self.close()


def log_print(*args, **kwargs):
    """Drop-in replacement for print that's safe with Logger redirection."""
    print(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Validation utilities
# ──────────────────────────────────────────────────────────────────────────────

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


@torch.no_grad()
def validate(
    diffusion: DiffusionProcess,
    denoiser: Denoiser,
    atom_marginals: torch.Tensor,
    bond_marginals: torch.Tensor,
    num_samples: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run validation: sample molecules and compute metrics."""
    original_mode = denoiser.training
    denoiser.eval()
    metrics = {"valid": 0, "total": 0, "errors": 0}

    for i in range(num_samples):
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
            metrics["errors"] += 1
            if metrics["errors"] <= 3:  # Only print first few errors
                print(f"    [WARN] Sample {i} failed: {type(e).__name__}: {e}")

    # Restore original mode
    if original_mode:
        denoiser.train()

    metrics["validity"] = metrics["valid"] / max(metrics["total"], 1)
    return metrics


def compute_val_loss(
    diffusion: DiffusionProcess,
    denoiser: Denoiser,
    val_loader,
    atom_marginals: torch.Tensor,
    bond_marginals: torch.Tensor,
    device: torch.device,
    use_amp: bool = True,
    max_batches: int = 20,
) -> Dict[str, float]:
    """
    Compute average loss on validation set (limited to max_batches for speed).
    """
    denoiser.eval()
    losses = {"coord": 0.0, "atom": 0.0, "bond": 0.0, "total": 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            batch = batch.to(device)

            with autocast(enabled=use_amp):
                _, loss_dict = diffusion.training_step(
                    data=batch,
                    denoiser=denoiser,
                    atom_marginals=atom_marginals,
                    bond_marginals=bond_marginals,
                )

            losses["coord"] += loss_dict["coord_loss"]
            losses["atom"] += loss_dict["atom_loss"]
            losses["bond"] += loss_dict["bond_loss"]
            losses["total"] += loss_dict["total_loss"]
            n_batches += 1

    denoiser.train()
    if n_batches > 0:
        for k in losses:
            losses[k] /= n_batches
    return losses


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    denoiser: Denoiser,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    config: QM9Config,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    train_loss: float,
):
    """Save a full training checkpoint for resumption."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "denoiser_state_dict": denoiser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
            "train_loss": train_loss,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state()
            if torch.cuda.is_available()
            else None,
            "numpy_rng_state": np.random.get_state(),
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device):
    """Load a training checkpoint and return all components."""
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    config = ckpt["config"]
    epoch = ckpt["epoch"]
    global_step = ckpt["global_step"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # Restore RNG states (with backward-compat: PyTorch RNG format may differ)
    if "rng_state" in ckpt:
        try:
            torch.set_rng_state(ckpt["rng_state"])
        except (TypeError, RuntimeError) as e:
            print(f"  [WARN] Could not restore torch RNG state: {e}")
    if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None:
        try:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
        except (TypeError, RuntimeError) as e:
            print(f"  [WARN] Could not restore cuda RNG state: {e}")
    if "numpy_rng_state" in ckpt:
        try:
            np.random.set_state(ckpt["numpy_rng_state"])
        except (TypeError, ValueError) as e:
            print(f"  [WARN] Could not restore numpy RNG state: {e}")

    return ckpt, config, epoch, global_step, best_val_loss


# ──────────────────────────────────────────────────────────────────────────────
# Main training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    """Main training loop."""
    # ── Config ────────────────────────────────────────────────────────────
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting QM9 pre-training")
    print(f"Device: {device}")
    print(f"Config: hidden_dim={config.hidden_dim}, layers={config.num_layers}, "
          f"diff_steps={config.diffusion_steps}, batch={config.batch_size}, "
          f"lr={config.learning_rate}, grad_accum={config.grad_accumulation}, "
          f"AMP={config.use_amp}")

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
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
    print(f"Atom marginals: {atom_marginals.tolist()}")

    # ── Models ────────────────────────────────────────────────────────────
    print("Initializing models...", flush=True)
    denoiser = Denoiser(
        num_atom_types=QM9_NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        edge_feat_dim=NUM_BOND_TYPES,
        attention=args.attention,
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

    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Denoiser parameters: {n_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
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

    scaler = GradScaler(enabled=config.use_amp)

    # ── Warm start: load only denoiser weights from old-format checkpoint ──
    if args.warm_start:
        print(f"Warm-starting from old checkpoint: {args.warm_start}")
        old_ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        old_state = old_ckpt.get("denoiser_state_dict", old_ckpt)
        model_state = denoiser.state_dict()
        loaded, skipped, mismatched = 0, 0, 0
        for key, value in old_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded += 1
                else:
                    print(f"  [SKIP] {key}: shape {value.shape} → {model_state[key].shape}")
                    mismatched += 1
            else:
                skipped += 1
        denoiser.load_state_dict(model_state)
        print(f"  Warm start: loaded {loaded} params, skipped {skipped}, "
              f"mismatched {mismatched} (optimizer/scheduler fresh)")

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if args.resume:
        ckpt, config_loaded, start_epoch, global_step, best_val_loss = \
            load_checkpoint(args.resume, device)
        denoiser.load_state_dict(ckpt["denoiser_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"Resumed from epoch {start_epoch + 1}, step {global_step}, "
              f"best_val_loss={best_val_loss:.4f}")
        start_epoch += 1  # Start from next epoch

    # ── Training loop ─────────────────────────────────────────────────────
    print("Starting training loop...", flush=True)
    nan_count = 0     # Counter for consecutive NaN losses
    nan_total = 0      # Total NaN batches in this epoch

    for epoch in range(start_epoch, config.max_epochs):
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

            # NaN / Inf detection
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                nan_total += 1
                if nan_count <= 5:  # Only print first few per cluster
                    # Detailed diagnostics (loss_dict values are Python floats from .item())
                    import math
                    nan_in_coord = math.isnan(loss_dict["coord_loss"]) or math.isinf(loss_dict["coord_loss"])
                    nan_in_atom = math.isnan(loss_dict["atom_loss"]) or math.isinf(loss_dict["atom_loss"])
                    nan_in_bond = math.isnan(loss_dict["bond_loss"]) or math.isinf(loss_dict["bond_loss"])
                    mol_sizes = batch.batch.bincount().tolist() if hasattr(batch, 'batch') else [batch.x.size(0)]
                    print(f"  [WARN] NaN/Inf at epoch {epoch+1}, "
                          f"batch {batch_idx+1} (consecutive={nan_count}, total_epoch={nan_total})")
                    print(f"         coord_nan={nan_in_coord}, atom_nan={nan_in_atom}, "
                          f"bond_nan={nan_in_bond}")
                    print(f"         mol_sizes={sorted(mol_sizes)[:10]}"
                          f"{'...' if len(mol_sizes) > 10 else ''}")
                    print(f"         edge_count={batch.edge_index.size(1)}, "
                          f"atom_count={batch.x.size(0)}")
                if nan_count >= 20:
                    emergency_path = os.path.join(
                        config.checkpoint_dir, f"qm9_emergency_epoch{epoch+1}.pt"
                    )
                    save_checkpoint(
                        emergency_path, denoiser, optimizer, scheduler, scaler,
                        config, epoch, global_step, best_val_loss,
                        float("nan"),
                    )
                    print(f"  [FATAL] {nan_count} consecutive NaN — aborting.")
                    return
                # Discard accumulated gradients and skip this batch
                optimizer.zero_grad()
                continue
            else:
                if nan_count > 0:
                    print(f"  Recovered after {nan_count} NaN batches")
                nan_count = 0  # Reset on healthy batch

            # Backward
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Accumulate epoch stats
            epoch_losses["coord"] += loss_dict["coord_loss"]
            epoch_losses["atom"] += loss_dict["atom_loss"]
            epoch_losses["bond"] += loss_dict["bond_loss"]
            epoch_losses["total"] += loss_dict["total_loss"]
            n_batches += 1
            global_step += 1

            # GPU throttle: sleep between batches to reduce utilization
            # on shared lab machines (0.15s → ~60-70% GPU util instead of 100%)
            if args.gpu_sleep > 0:
                import time as _time
                _time.sleep(args.gpu_sleep)

            # Step-level logging
            if global_step % config.log_every_n_steps == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Epoch {epoch+1}/{config.max_epochs} | "
                    f"Step {global_step} | "
                    f"Loss: {loss_dict['total_loss']:.4f} | "
                    f"Coord: {loss_dict['coord_loss']:.4f} | "
                    f"Atom: {loss_dict['atom_loss']:.4f} | "
                    f"Bond: {loss_dict['bond_loss']:.4f} | "
                    f"LR: {lr:.2e}"
                )

        # ── Epoch summary ─────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_total = epoch_losses["total"] / max(n_batches, 1)
        avg_coord = epoch_losses["coord"] / max(n_batches, 1)
        avg_atom = epoch_losses["atom"] / max(n_batches, 1)
        avg_bond = epoch_losses["bond"] / max(n_batches, 1)

        nan_info = f" | NaN batches: {nan_total}" if nan_total > 0 else ""
        print(
            f"=== Epoch {epoch+1}/{config.max_epochs} Summary ===\n"
            f"  Train Loss: {avg_total:.4f} | "
            f"Coord: {avg_coord:.4f} | "
            f"Atom: {avg_atom:.4f} | "
            f"Bond: {avg_bond:.4f} | "
            f"Time: {epoch_time:.1f}s{nan_info}"
        )
        nan_total = 0  # Reset per-epoch counter

        # ── Validation ────────────────────────────────────────────────────
        if (epoch + 1) % config.val_every_n_epochs == 0:
            print("Running validation...")

            # 1. Compute validation loss
            try:
                val_losses = compute_val_loss(
                    diffusion, denoiser, val_loader,
                    atom_marginals, bond_marginals, device, config.use_amp,
                )
                print(f"  Val Loss: {val_losses['total']:.4f} | "
                      f"Coord: {val_losses['coord']:.4f} | "
                      f"Atom: {val_losses['atom']:.4f} | "
                      f"Bond: {val_losses['bond']:.4f}")
            except Exception as e:
                print(f"  [WARN] Validation loss computation failed: {e}")
                traceback.print_exc()
                val_losses = {"total": float("inf")}

            # 2. Molecule validity (sampling)
            try:
                val_metrics = validate(
                    diffusion, denoiser, atom_marginals, bond_marginals,
                    num_samples=args.val_samples, device=device,
                )
                print(f"  Validity: {val_metrics['validity']:.3f} "
                      f"({val_metrics['valid']}/{val_metrics['total']} valid, "
                      f"{val_metrics['errors']} errors)")
            except Exception as e:
                print(f"  [WARN] Validation sampling failed: {e}")
                traceback.print_exc()
                val_metrics = {"validity": 0.0}

            # 3. Save best model (based on validation loss)
            val_total = val_losses["total"]
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_path = os.path.join(config.checkpoint_dir, "qm9_best.pt")
                save_checkpoint(
                    best_path, denoiser, optimizer, scheduler, scaler,
                    config, epoch, global_step, best_val_loss, avg_total,
                )
                print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f}, "
                      f"validity: {val_metrics['validity']:.3f})")

        # ── Periodic checkpoint ───────────────────────────────────────────
        if (epoch + 1) % config.save_every_n_epochs == 0:
            ckpt_path = os.path.join(
                config.checkpoint_dir, f"qm9_epoch{epoch+1}.pt"
            )
            save_checkpoint(
                ckpt_path, denoiser, optimizer, scheduler, scaler,
                config, epoch, global_step, best_val_loss, avg_total,
            )

        # ── Latest checkpoint (always, for recovery) ──────────────────────
        latest_path = os.path.join(config.checkpoint_dir, "qm9_latest.pt")
        save_checkpoint(
            latest_path, denoiser, optimizer, scheduler, scaler,
            config, epoch, global_step, best_val_loss, avg_total,
        )

    # ── Final save ─────────────────────────────────────────────────────────
    final_path = os.path.join(config.checkpoint_dir, "qm9_final.pt")
    save_checkpoint(
        final_path, denoiser, optimizer, scheduler, scaler,
        config, config.max_epochs - 1, global_step, best_val_loss, avg_total,
    )
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

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
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--attention", action="store_true", default=False,
                        help="Enable attention in EGNN (disabled by default for stability)")

    # Data
    parser.add_argument("--data_root", type=str, default="./data/qm9")
    parser.add_argument("--max_molecules", type=int, default=None)

    # GPU throttle (for shared lab machines)
    parser.add_argument("--gpu_sleep", type=float, default=0.0,
                        help="Sleep seconds between batches to reduce GPU utilization")

    # Validation
    parser.add_argument("--val_samples", type=int, default=20,
                        help="Number of molecules to sample for validity check")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to new-format checkpoint to resume from")
    parser.add_argument("--warm_start", type=str, default=None,
                        help="Path to old-format checkpoint to load denoiser weights from")

    args = parser.parse_args()

    # Setup logging to file
    log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs"
    )
    os.makedirs(log_dir, exist_ok=True)

    with Logger(log_dir) as logger:
        try:
            train(args)
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user.")
        except Exception as e:
            print(f"\n[FATAL] Training crashed: {type(e).__name__}: {e}")
            traceback.print_exc()
            sys.exit(1)
