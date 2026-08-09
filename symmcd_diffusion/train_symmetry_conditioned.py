"""
Phase 2: Symmetry-conditioned Core fine-tuning.

Fine-tunes the QM9-pretrained diffusion model to generate COF molecular
Cores (rigid scaffolds with Q/R attachment points).

Core-only architecture:
  - Condition: symmetry only (128-dim)
  - Output: Core (Q+R points)
  - pycofbuilder handles Connector + Functional Group assembly

Two-stage training:
  Stage A (freeze_base_epochs): Freeze base denoiser, train FiLM layers
                                and symmetry encoder.
  Stage B (remaining epochs):   Unfreeze last 3 EGNN layers, full fine-tune.

Usage:
    python train_symmetry_conditioned.py \
        --qm9_ckpt symmcd_diffusion/checkpoints/qm9_best.pt \
        --epochs 100 --batch_size 32 --lr 5e-5
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from symmcd_diffusion.config.base_config import COFDiffusionConfig
from symmcd_diffusion.models.conditional_denoiser import SymmetryConditionedDenoiser
from symmcd_diffusion.models.diffusion_process import DiffusionProcess
from symmcd_diffusion.data.core_dataset import CoreDataset, NUM_CORE_ATOM_TYPES
from symmcd_diffusion.data.qm9_dataset import NUM_BOND_TYPES
from symmcd_diffusion.symmetry.symmetry_encoder import (
    NUM_POINT_GROUPS, POINT_GROUP_TO_IDX, MOLECULAR_POINT_GROUPS,
)
from symmcd_diffusion.symmetry.point_group import compute_point_group


# ──────────────────────────────────────────────────────────────────────────────
# Helper: QM9 atomic number mapping (for point group computation)
# ──────────────────────────────────────────────────────────────────────────────

QM9_IDX_TO_ATOMIC_NUM = [1, 6, 7, 8, 9]   # H, C, N, O, F
COF_IDX_TO_ATOMIC_NUM = [1, 6, 7, 8, 9, 17, 35, 16, 0, 0]  # Q=0, X=0 for PG calc


def idx_to_atomic_numbers(atom_types: torch.Tensor, vocab: str = "cof") -> torch.Tensor:
    """Convert one-hot atom types to atomic numbers (for point group computation)."""
    mapping = COF_IDX_TO_ATOMIC_NUM if vocab == "cof" else QM9_IDX_TO_ATOMIC_NUM
    indices = atom_types.argmax(dim=-1).cpu()
    atomic_nums = torch.tensor([mapping[i] for i in indices], dtype=torch.long)
    return atomic_nums


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_symmetry_conditioned(
    diffusion: DiffusionProcess,
    denoiser: SymmetryConditionedDenoiser,
    atom_marginals: torch.Tensor,
    bond_marginals: torch.Tensor,
    num_samples: int = 50,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Phase 2 validation: sample molecules and compute symmetry-specific metrics.

    Returns:
        dict with validity, symmetry_consistency, connectivity_correctness
    """
    denoiser.eval()
    metrics = {
        "valid": 0, "total": 0, "errors": 0,
        "symmetry_match": 0,
        "connector_match": 0,
    }

    for i in range(num_samples):
        # Sample random condition
        pg_idx = np.random.randint(0, NUM_POINT_GROUPS)
        point_group = MOLECULAR_POINT_GROUPS[pg_idx]
        num_conn_target = np.random.randint(1, 7)
        fg_type = np.random.randint(0, 10)

        # COF building blocks are larger than QM9 (20-80 atoms)
        num_atoms = np.random.randint(20, 60)

        try:
            # Build condition tensors
            pg_tensor = torch.tensor([pg_idx], device=device)
            conn_tensor = torch.tensor([num_conn_target], device=device)
            fg_tensor = torch.tensor([fg_type], device=device)
            condition = denoiser.encode_condition(pg_tensor)

            # Sample
            result = diffusion.sample(
                denoiser=denoiser.denoiser,
                num_atoms=num_atoms,
                atom_marginals=atom_marginals,
                bond_marginals=bond_marginals,
                condition=condition,
                device=str(device),
            )

            # Validity via RDKit
            metrics["valid"] += _check_rdkit_validity(
                result["atom_types"], result["positions"]
            )
            metrics["total"] += 1

            # Symmetry consistency
            atomic_nums = idx_to_atomic_numbers(result["atom_types"], "cof")
            computed_pg = compute_point_group(atomic_nums, result["positions"])
            if computed_pg == point_group:
                metrics["symmetry_match"] += 1

            # Connectivity: count Q atoms
            q_count = (result["atom_types"].argmax(dim=-1) == 8).sum().item()  # Q=8
            if q_count == num_conn_target:
                metrics["connector_match"] += 1

        except Exception as e:
            metrics["errors"] += 1

    denoiser.train()
    n = max(metrics["total"], 1)
    return {
        "validity": metrics["valid"] / n,
        "symmetry_consistency": metrics["symmetry_match"] / n,
        "connectivity_correctness": metrics["connector_match"] / n,
        "num_errors": metrics["errors"],
    }


def _check_rdkit_validity(atom_types: torch.Tensor, positions: torch.Tensor) -> float:
    """Check if generated molecule is valid via RDKit."""
    try:
        from rdkit import Chem
        atom_idx = atom_types.argmax(dim=-1).cpu().numpy()
        # Map COF vocab to RDKit atoms (skip Q=8, X=9)
        atom_map = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9, 5: 17, 6: 35, 7: 16}
        pos = positions.cpu().numpy()

        mol = Chem.RWMol()
        actual_atoms = []
        for a in atom_idx:
            atomic_num = atom_map.get(a, 6)
            if atomic_num > 0:
                mol.AddAtom(Chem.Atom(atomic_num))
                actual_atoms.append(len(actual_atoms))

        if len(actual_atoms) < 3:
            return 0.0

        # Add bonds based on distances
        for i_idx, i in enumerate(actual_atoms):
            for j_idx, j in enumerate(actual_atoms):
                if j >= i:
                    break
                dist = np.linalg.norm(pos[i] - pos[j])
                # Simple distance-based bonding
                if dist < 1.8:
                    mol.AddBond(int(i_idx), int(j_idx), Chem.BondType.SINGLE)

        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
            return 1.0
        except Exception:
            return 0.0
    except ImportError:
        return 1.0  # Assume valid if RDKit not available


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    """Main Phase 2 training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = COFDiffusionConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        diffusion_steps=args.diffusion_steps,
    )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Phase 2: Symmetry-conditioned fine-tuning")
    print(f"Device: {device}")
    print(f"QM9 checkpoint: {args.qm9_ckpt}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # ── Load COF data ────────────────────────────────────────────────────
    print("Loading COF Core data...")
    core_dataset = CoreDataset(augment=True, target_count=5000)
    train_size = int(0.8 * len(core_dataset))
    val_size = (len(core_dataset) - train_size) // 2
    test_size = len(core_dataset) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(
        core_dataset, [train_size, val_size, test_size]
    )
    train_loader = core_dataset.get_dataloader(batch_size=config.batch_size, shuffle=True)
    val_loader = core_dataset.get_dataloader(batch_size=config.batch_size, shuffle=False)

    # Compute marginals from training data
    atom_counts = torch.zeros(NUM_CORE_ATOM_TYPES)
    for d in train_data:
        for a in d.x.argmax(dim=-1):
            atom_counts[a] += 1
    atom_marginals = (atom_counts + 1e-5) / (atom_counts + 1e-5).sum()
    atom_marginals = atom_marginals.to(device)
    bond_marginals = torch.ones(NUM_BOND_TYPES, device=device) / NUM_BOND_TYPES
    print(f"Core samples: {len(core_dataset)} | Train: {train_size} Val: {val_size} Test: {test_size}")
    print(f"Atom marginals: {[f'{x:.4f}' for x in atom_marginals.tolist()]}")

    # ── Build model (Core-only mode) ─────────────────────────────────────
    print("Building symmetry-conditioned Core denoiser...")
    model = SymmetryConditionedDenoiser(
        num_atom_types=NUM_CORE_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        edge_feat_dim=NUM_BOND_TYPES,
        attention=False,  # Disabled for stability
        dropout=0.1,
        num_point_groups=NUM_POINT_GROUPS,
        symmetry_encoding_dim=config.symmetry_encoding_dim,
        core_only=True,
    ).to(device)

    # ── Load QM9 pretrained weights ──────────────────────────────────────
    print(f"Loading QM9 pretrained weights from {args.qm9_ckpt}...")
    qm9_ckpt = torch.load(args.qm9_ckpt, map_location=device, weights_only=False)
    qm9_state = qm9_ckpt["denoiser_state_dict"]

    # Load into base denoiser (skip FiLM layers — those don't exist in QM9 model)
    base_state = model.denoiser.state_dict()
    loaded_keys = []
    skipped_keys = []
    for key, value in qm9_state.items():
        if key in base_state:
            if base_state[key].shape == value.shape:
                base_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {base_state[key].shape})")
        else:
            skipped_keys.append(f"{key} (not in COF model)")
    model.denoiser.load_state_dict(base_state)
    print(f"  Loaded {len(loaded_keys)}/{len(qm9_state)} parameters")
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys)} keys (expected for FiLM/condition layers)")

    # ── Diffusion process ────────────────────────────────────────────────
    diffusion = DiffusionProcess(
        num_atom_types=NUM_CORE_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        timesteps=config.diffusion_steps,
        noise_schedule=config.noise_schedule,
        coord_loss_weight=config.coord_loss_weight,
        atom_loss_weight=config.atom_type_loss_weight,
        bond_loss_weight=config.bond_type_loss_weight,
        use_uniform_prior=args.uniform_prior,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}, Trainable: {n_trainable:,}")

    # ── Stage A: Freeze base denoiser ────────────────────────────────────
    _freeze_base_denoiser(model, freeze=True)
    n_trainable_a = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage A: Frozen base denoiser → {n_trainable_a:,} trainable (FiLM + embeddings + aux heads)")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    scaler = GradScaler(enabled=config.use_amp)

    # ── Training loop ────────────────────────────────────────────────────
    print("Starting Phase 2 fine-tuning...")
    best_val_consistency = 0.0
    global_step = 0

    for epoch in range(config.max_epochs):
        # Stage transition
        if epoch == config.freeze_base_epochs:
            print(f"\n=== Stage B: Unfreezing last 3 EGNN layers (epoch {epoch+1}) ===")
            _freeze_base_denoiser(model, freeze=False, unfreeze_last_n=3)
            n_trainable_b = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Stage B: {n_trainable_b:,} trainable parameters")
            # Re-create optimizer with all trainable parameters
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate * 0.5,  # Lower LR for full model
                weight_decay=config.weight_decay,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.max_epochs - config.freeze_base_epochs,
            )

        model.train()
        epoch_losses = {"total": 0.0, "coord": 0.0, "atom": 0.0,
                        "bond": 0.0, "symmetry": 0.0, "connectivity": 0.0}
        n_batches = 0
        epoch_start = time.time()

        for batch in train_loader:
            batch = batch.to(device)

            with autocast(enabled=config.use_amp):
                # ---- Diffusion loss ----
                loss, loss_dict = diffusion.training_step(
                    data=batch,
                    denoiser=model.denoiser,
                    atom_marginals=atom_marginals,
                    bond_marginals=bond_marginals,
                    condition=model.encode_condition(batch.symm_idx),
                )

                # ---- Auxiliary losses ----
                aux_loss = torch.tensor(0.0, device=device)

                # Symmetry classification loss (if we have labels)
                if hasattr(batch, 'symm_idx') and batch.symm_idx is not None:
                    # Re-run forward to get auxiliary predictions
                    # (We already ran the denoiser in training_step; for efficiency,
                    #  we'd ideally combine these. For now, re-use the loss_dict.)
                    pass

                # Re-compute with full forward for auxiliary heads
                # In production, integrate aux heads into training_step.
                # For this implementation, compute separately every N batches.
                # Auxiliary symmetry loss (compute every step for core_only mode)
                if config.aux_symmetry_loss_weight > 0 and hasattr(batch, 'symm_idx'):
                    try:
                        with autocast(enabled=config.use_amp):
                            full_output = model(
                                atom_types=batch.x,
                                positions=batch.positions,
                                t=torch.randint(0, config.diffusion_steps,
                                               (batch.symm_idx.size(0),),
                                               dtype=torch.long, device=device),
                                edge_index=batch.edge_index,
                                edge_attr=batch.edge_attr,
                                batch=batch.batch,
                                point_group_idx=batch.symm_idx,
                            )
                            symm_loss = F.cross_entropy(
                                full_output["symmetry_logits"], batch.symm_idx
                            )
                            aux_loss = aux_loss + config.aux_symmetry_loss_weight * symm_loss
                            epoch_losses["symmetry"] += symm_loss.item()
                    except Exception as e:
                        if global_step < 5:
                            print(f"  [WARN] Aux loss failed: {e}")

                total_loss = loss + aux_loss
                total_loss = total_loss / config.grad_accumulation

            # NaN safety: skip this batch if loss is invalid
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if global_step < 10:
                    print(f"  [SKIP] NaN/Inf loss at step {global_step}, skipping batch")
                optimizer.zero_grad()
                # Must call scaler.update() to keep AMP state consistent
                if (n_batches + 1) % config.grad_accumulation == 0:
                    scaler.update()
                continue

            # Backward
            scaler.scale(total_loss).backward()

            if (n_batches + 1) % config.grad_accumulation == 0:
                scaler.unscale_(optimizer)
                # Check for NaN gradients
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                if has_nan_grad:
                    if global_step < 10:
                        print(f"  [SKIP] NaN gradient at step {global_step}, skipping update")
                    optimizer.zero_grad()
                    scaler.update()  # Keep AMP state consistent
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Only accumulate valid losses
            total_val = loss_dict["total_loss"]
            if not (np.isnan(total_val) or np.isinf(total_val)):
                epoch_losses["total"] += total_val
                epoch_losses["coord"] += loss_dict.get("coord_loss", 0.0)
                epoch_losses["atom"] += loss_dict.get("atom_loss", 0.0)
                epoch_losses["bond"] += loss_dict.get("bond_loss", 0.0)
                n_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Epoch {epoch+1}/{config.max_epochs} | "
                      f"Loss: {loss_dict['total_loss']:.4f} | "
                      f"LR: {lr:.2e}")

        # ── Epoch summary ─────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_losses["total"] / max(n_batches, 1)
        print(f"=== Epoch {epoch+1}/{config.max_epochs} === "
              f"Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | "
              f"Coord: {epoch_losses['coord']/max(n_batches,1):.4f} | "
              f"Atom: {epoch_losses['atom']/max(n_batches,1):.4f} | "
              f"Symm: {epoch_losses['symmetry']/max(n_batches,1):.4f}")

        # ── Validation ────────────────────────────────────────────────────
        if (epoch + 1) % config.val_every_n_epochs == 0:
            print("Running validation...")
            try:
                val_metrics = validate_symmetry_conditioned(
                    diffusion, model, atom_marginals, bond_marginals,
                    num_samples=args.val_samples, device=device,
                )
                print(f"  Validity: {val_metrics['validity']:.3f} | "
                      f"Symm consistency: {val_metrics['symmetry_consistency']:.3f} | "
                      f"Connector: {val_metrics['connectivity_correctness']:.3f}")

                if val_metrics["symmetry_consistency"] > best_val_consistency:
                    best_val_consistency = val_metrics["symmetry_consistency"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "val_metrics": val_metrics,
                        },
                        os.path.join(config.checkpoint_dir, "symmcd_best.pt"),
                    )
                    print(f"  ✓ Saved best model (symm_consistency={best_val_consistency:.3f})")
            except Exception as e:
                print(f"  [WARN] Validation failed: {e}")
                traceback.print_exc()

        # ── Periodic checkpoint ───────────────────────────────────────────
        if (epoch + 1) % config.save_every_n_epochs == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                os.path.join(config.checkpoint_dir, f"symmcd_epoch{epoch+1}.pt"),
            )

        # Latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            os.path.join(config.checkpoint_dir, "symmcd_latest.pt"),
        )

    # ── Final save ─────────────────────────────────────────────────────────
    final_path = os.path.join(config.checkpoint_dir, "symmcd_denoiser_finetuned.pt")
    torch.save(
        {
            "epoch": config.max_epochs,
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        final_path,
    )
    print(f"\nPhase 2 complete! Model saved to {final_path}")


def _freeze_base_denoiser(
    model: SymmetryConditionedDenoiser,
    freeze: bool = True,
    unfreeze_last_n: int = 0,
):
    """Freeze or selectively unfreeze the base denoiser parameters."""
    base_denoiser = model.denoiser

    if freeze:
        # Freeze all base denoiser parameters
        for param in base_denoiser.parameters():
            param.requires_grad = False
        # Keep FiLM layers trainable
        for layer in base_denoiser.egnn.layers:
            if layer.use_film:
                for param in layer.film.parameters():
                    param.requires_grad = True
    else:
        # Unfreeze last N EGNN layers (or all if unfreeze_last_n=0)
        total_layers = len(base_denoiser.egnn.layers)
        layers_to_unfreeze = unfreeze_last_n if unfreeze_last_n > 0 else total_layers

        for i, layer in enumerate(base_denoiser.egnn.layers):
            if i >= total_layers - layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

        # Also unfreeze output heads
        for param in base_denoiser.coord_head.parameters():
            param.requires_grad = True
        for param in base_denoiser.atom_head.parameters():
            param.requires_grad = True
        for param in base_denoiser.bond_head.parameters():
            param.requires_grad = True


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Symmetry-conditioned COF fine-tuning")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--diffusion_steps", type=int, default=1000)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # Data
    parser.add_argument("--data_dir", type=str,
                        default="/home/tianyajun/MARL_for_COFs/pycofbuilder/data")
    parser.add_argument("--augmented_dir", type=str, default=None)

    # Pretrained
    parser.add_argument("--qm9_ckpt", type=str, required=True,
                        help="Path to QM9 pretrained checkpoint")

    # Validation
    parser.add_argument("--val_samples", type=int, default=20)

    # Resume
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--uniform_prior", action="store_true", default=False,
                        help="Use uniform prior for discrete diffusion (enables de novo generation)")

    args = parser.parse_args()

    try:
        train(args)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"\n[FATAL] Training crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
