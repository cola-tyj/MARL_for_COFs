"""
Joint training: QM9 + rule-generated symmetric molecules.

Two-stage training:
  Stage A (warmup): Freeze EGNN backbone, train only FiLM + output heads.
  Stage B (fine-tune): Unfreeze last 3 EGNN layers, lower LR.

The model learns to denoise molecules conditioned on a binary sym_label:
  - label=0: arbitrary molecules (QM9 distribution)
  - label=1: symmetric molecules (rule-generated distribution)

After training, setting label=1 conditions the model to generate
molecules with the learned symmetric characteristics.

Usage:
    # Stage A only (quick test)
    python train_joint.py --stage A --epochs 50 --batch_size 64

    # Full two-stage training
    python train_joint.py --stage all --epochs_a 50 --epochs_b 100
"""
import argparse, os, sys, time, json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from symmcd_diffusion.data.joint_dataset import JointDataset, SYM_LABEL_DIM
from symmcd_diffusion.data.qm9_dataset import QM9_NUM_ATOM_TYPES, NUM_BOND_TYPES
from symmcd_diffusion.models.denoiser import Denoiser
from symmcd_diffusion.models.diffusion_process import DiffusionProcess


# ── Logging ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_dir, name="train_joint"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{name}_{ts}.log")
        self.file = open(self.log_path, "a", buffering=1)
        self._orig = sys.stdout

    def write(self, msg):
        self._orig.write(msg)
        self.file.write(msg)

    def flush(self):
        self._orig.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *a):
        sys.stdout = self._orig
        self.close()


# ── Model ──────────────────────────────────────────────────────────────────────

class JointDenoiser(nn.Module):
    """
    Denoiser with binary sym_label FiLM conditioning.

    Wraps the base Denoiser with a learnable sym_label embedding.
    QM9 pre-trained weights are loaded for the EGNN backbone;
    FiLM layers and sym_embed are initialized from scratch.

    Compatible with DiffusionProcess.training_step() — callers should
    pre-compute condition = model.encode_condition(sym_label) and pass
    it as the `condition` kwarg.
    """

    def __init__(self, hidden_dim=256, num_layers=9, attention=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = SYM_LABEL_DIM

        # Binary sym_label → FiLM condition vector
        self.sym_embed = nn.Embedding(2, SYM_LABEL_DIM)

        # Denoiser with FiLM conditioning
        self.denoiser = Denoiser(
            num_atom_types=QM9_NUM_ATOM_TYPES,
            num_bond_types=NUM_BOND_TYPES,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_feat_dim=NUM_BOND_TYPES,
            attention=attention,
            dropout=0.1,
            condition_dim=SYM_LABEL_DIM,
            use_film=True,
        )

    def encode_condition(self, sym_label):
        """sym_label: (batch,) int tensor → (batch, condition_dim)"""
        return self.sym_embed(sym_label)

    def forward(self, atom_types, positions, t, edge_index,
                edge_attr=None, condition=None, node_mask=None, batch=None):
        """Forward pass. `condition` is pre-computed via encode_condition()."""
        return self.denoiser(
            atom_types=atom_types, positions=positions, t=t,
            edge_index=edge_index, edge_attr=edge_attr,
            condition=condition, node_mask=node_mask, batch=batch,
        )

    def load_qm9_backbone(self, ckpt_path, device):
        """Load QM9 pre-trained weights, skipping FiLM layers."""
        print(f"Loading QM9 backbone from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        old_state = ckpt.get("denoiser_state_dict", ckpt)

        model_state = self.denoiser.state_dict()
        loaded, skipped, mismatched = 0, 0, 0

        for key, value in old_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded += 1
                else:
                    print(f"  [SKIP] {key}: {list(value.shape)} → {list(model_state[key].shape)}")
                    mismatched += 1
            else:
                skipped += 1

        self.denoiser.load_state_dict(model_state)
        print(f"  Loaded {loaded}, skipped {skipped} (FiLM etc), mismatched {mismatched}")

    def freeze_backbone(self):
        """Freeze EGNN backbone, keep FiLM + heads + embedding trainable."""
        for p in self.denoiser.parameters():
            p.requires_grad = False
        # Unfreeze: output heads, embedding, FiLM layers (inside EGNN)
        for name, p in self.denoiser.named_parameters():
            if any(x in name for x in ['coord_head', 'atom_head', 'bond_head',
                                        'embedding', 'film']):
                p.requires_grad = True
        # sym_embed always trainable
        for p in self.sym_embed.parameters():
            p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"  Frozen backbone: {n_trainable:,}/{n_total:,} params trainable "
              f"({100*n_trainable/n_total:.1f}%)")

    def unfreeze_last_n_layers(self, n=3):
        """Unfreeze the last n EGNN layers for fine-tuning."""
        layers = self.denoiser.egnn.layers
        for i in range(max(0, len(layers) - n), len(layers)):
            for p in layers[i].parameters():
                p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"  Unfroze last {n} EGNN layers: {n_trainable:,}/{n_total:,} "
              f"params trainable ({100*n_trainable/n_total:.1f}%)")


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, diffusion, loader, optimizer, scaler,
                atom_marginals, bond_marginals, device, use_amp,
                epoch, total_epochs, grad_accum=2):
    """Single training epoch."""
    model.train()
    losses = {"coord": 0, "atom": 0, "bond": 0, "total": 0}
    n_batches = 0
    nan_count = 0

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        # Encode sym_label → FiLM condition
        if hasattr(batch, 'sym_label') and batch.sym_label is not None:
            condition = model.encode_condition(batch.sym_label)
        else:
            condition = None

        with autocast(enabled=use_amp):
            loss, loss_dict = diffusion.training_step(
                data=batch, denoiser=model,
                atom_marginals=atom_marginals,
                bond_marginals=bond_marginals,
                condition=condition,
            )
            loss = loss / grad_accum

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"  [WARN] NaN at batch {batch_idx+1} (total={nan_count})")
            if nan_count >= 20:
                print(f"  [FATAL] {nan_count} consecutive NaN, aborting epoch")
                return {k: float('nan') for k in losses}
            optimizer.zero_grad()
            continue
        nan_count = 0

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses["coord"] += loss_dict["coord_loss"]
        losses["atom"] += loss_dict["atom_loss"]
        losses["bond"] += loss_dict["bond_loss"]
        losses["total"] += loss_dict["total_loss"]
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Epoch {epoch}/{total_epochs} batch {batch_idx+1}: "
                  f"loss={loss_dict['total_loss']:.3f} "
                  f"coord={loss_dict['coord_loss']:.3f} "
                  f"atom={loss_dict['atom_loss']:.3f} "
                  f"bond={loss_dict['bond_loss']:.3f}")

    for k in losses:
        losses[k] /= max(n_batches, 1)
    return losses


def main():
    parser = argparse.ArgumentParser()
    # Stage control
    parser.add_argument("--stage", type=str, default="A",
                        choices=["A", "B", "all"],
                        help="Training stage: A (warmup), B (fine-tune), all")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--attention", action="store_true", default=False)

    # Training
    parser.add_argument("--epochs_a", type=int, default=50,
                        help="Stage A epochs")
    parser.add_argument("--epochs_b", type=int, default=100,
                        help="Stage B epochs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_a", type=float, default=1e-3)
    parser.add_argument("--lr_b", type=float, default=5e-5)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true", default=False)

    # Data
    parser.add_argument("--qm9_root", type=str, default="./data/qm9")
    parser.add_argument("--sym_dir", type=str,
                        default="/home/tianyajun/MARL_for_COFS/data/symmetric_molecules")
    parser.add_argument("--qm9_fraction", type=float, default=0.7)

    # QM9 pre-trained
    parser.add_argument("--qm9_ckpt", type=str,
                        default="symmcd_diffusion/checkpoints/qm9_best.pt")

    # Resume
    parser.add_argument("--resume", type=str, default=None)

    # Output
    parser.add_argument("--ckpt_dir", type=str,
                        default="symmcd_diffusion/checkpoints/joint")
    parser.add_argument("--log_dir", type=str,
                        default="symmcd_diffusion/logs")
    parser.add_argument("--device", type=str, default="cuda:2")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Joint Training")
    print(f"Device: {device}")
    print(f"Stage: {args.stage}")
    print(f"QM9 fraction: {args.qm9_fraction}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = JointDataset(
        qm9_root=args.qm9_root, sym_data_dir=args.sym_dir,
        split="train", qm9_fraction=args.qm9_fraction,
    )
    val_dataset = JointDataset(
        qm9_root=args.qm9_root, sym_data_dir=args.sym_dir,
        split="val", qm9_fraction=args.qm9_fraction,
    )

    train_loader = train_dataset.get_dataloader(batch_size=args.batch_size)
    val_loader = val_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    atom_marginals = train_dataset.qm9_dataset.atom_marginals.to(device)
    bond_marginals = torch.ones(NUM_BOND_TYPES, device=device) / NUM_BOND_TYPES

    print(f"Train: {len(train_dataset)} (QM9: {train_dataset._qm9_len}, "
          f"Sym: {train_dataset._sym_len})")
    print(f"Val: {len(val_dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = JointDenoiser(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        attention=args.attention,
    ).to(device)

    # Load QM9 pre-trained backbone
    if os.path.exists(args.qm9_ckpt):
        model.load_qm9_backbone(args.qm9_ckpt, device)
    else:
        print(f"  [WARN] QM9 checkpoint not found: {args.qm9_ckpt}")
        print(f"  Training from scratch!")

    diffusion = DiffusionProcess(
        num_atom_types=QM9_NUM_ATOM_TYPES,
        num_bond_types=NUM_BOND_TYPES,
        timesteps=1000,
        noise_schedule="cosine",
        coord_loss_weight=1.0,
        atom_loss_weight=1.0,
        bond_loss_weight=0.5,
    ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_total:,}")

    # ── Stage A: Warmup ──────────────────────────────────────────────────────
    if args.stage in ("A", "all"):
        print(f"\n{'='*60}")
        print("Stage A: Freeze backbone, train FiLM + heads")
        print(f"{'='*60}")

        model.freeze_backbone()

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr_a, weight_decay=1e-5,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_a)
        scaler = GradScaler(enabled=args.use_amp)

        best_loss = float('inf')
        start_epoch = 0

        for epoch in range(start_epoch, args.epochs_a):
            t0 = time.time()
            losses = train_epoch(
                model, diffusion, train_loader, optimizer, scaler,
                atom_marginals, bond_marginals, device, args.use_amp,
                epoch + 1, args.epochs_a, args.grad_accum,
            )
            scheduler.step()
            elapsed = time.time() - t0

            print(f"  Stage A Epoch {epoch+1}/{args.epochs_a}: "
                  f"loss={losses['total']:.4f} coord={losses['coord']:.4f} "
                  f"atom={losses['atom']:.4f} bond={losses['bond']:.4f} "
                  f"({elapsed:.1f}s)")

            if losses['total'] < best_loss:
                best_loss = losses['total']
                torch.save({
                    'epoch': epoch, 'stage': 'A',
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, os.path.join(args.ckpt_dir, 'stageA_best.pt'))

        # Save final Stage A
        torch.save({
            'epoch': args.epochs_a, 'stage': 'A',
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, os.path.join(args.ckpt_dir, 'stageA_final.pt'))
        print(f"Stage A complete! Best loss: {best_loss:.4f}")

        # Prepare for Stage B
        args.resume = os.path.join(args.ckpt_dir, 'stageA_final.pt')

    # ── Stage B: Fine-tune ───────────────────────────────────────────────────
    if args.stage in ("B", "all"):
        print(f"\n{'='*60}")
        print("Stage B: Unfreeze last layers, fine-tune")
        print(f"{'='*60}")

        # Load Stage A checkpoint if running Stage B standalone
        if args.stage == "B" and args.resume:
            print(f"Loading Stage A checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
                  f"best_loss={ckpt.get('best_loss', '?'):.4f}")

        model.unfreeze_last_n_layers(n=3)

        # Different LR for backbone vs new components
        backbone_params = []
        new_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'egnn.layers' in name:
                backbone_params.append(p)
            else:
                new_params.append(p)

        optimizer = AdamW([
            {'params': new_params, 'lr': args.lr_a * 0.5},
            {'params': backbone_params, 'lr': args.lr_b},
        ], weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_b)
        scaler = GradScaler(enabled=args.use_amp)

        best_loss = float('inf')

        for epoch in range(args.epochs_b):
            t0 = time.time()
            losses = train_epoch(
                model, diffusion, train_loader, optimizer, scaler,
                atom_marginals, bond_marginals, device, args.use_amp,
                epoch + 1, args.epochs_b, args.grad_accum,
            )
            scheduler.step()
            elapsed = time.time() - t0

            print(f"  Stage B Epoch {epoch+1}/{args.epochs_b}: "
                  f"loss={losses['total']:.4f} coord={losses['coord']:.4f} "
                  f"atom={losses['atom']:.4f} bond={losses['bond']:.4f} "
                  f"({elapsed:.1f}s)")

            if losses['total'] < best_loss:
                best_loss = losses['total']
                torch.save({
                    'epoch': epoch, 'stage': 'B',
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, os.path.join(args.ckpt_dir, 'stageB_best.pt'))

            # Save periodic
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch, 'stage': 'B',
                    'model_state_dict': model.state_dict(),
                }, os.path.join(args.ckpt_dir, f'stageB_epoch{epoch+1}.pt'))

        torch.save({
            'epoch': args.epochs_b, 'stage': 'B',
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, os.path.join(args.ckpt_dir, 'stageB_final.pt'))
        print(f"Stage B complete! Best loss: {best_loss:.4f}")

    print(f"\nTraining done! Checkpoints in {args.ckpt_dir}")


if __name__ == "__main__":
    main()
