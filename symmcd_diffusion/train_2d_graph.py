"""
2D molecular graph diffusion training (no coordinates).

Key changes from Phase 1/2:
  1. No coordinate diffusion — only discrete atom + bond diffusion
  2. Correct empirical bond marginals (87% none, not uniform 20%)
  3. Coordinates fixed to zero — EGNN still runs but distances are uniform
  4. FiLM conditioning on symmetry label (binary: QM9=0, symmetric=1)

This is much simpler than full 3D diffusion and avoids the coord_noise
prediction problem we discovered with the EGNN backbone.

Usage:
    python symmcd_diffusion/train_2d_graph.py --epochs 100 --batch_size 64
"""
import argparse, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from symmcd_diffusion.data.joint_dataset import JointDataset, SYM_LABEL_DIM
from symmcd_diffusion.data.qm9_dataset import QM9_NUM_ATOM_TYPES, NUM_BOND_TYPES
from symmcd_diffusion.models.denoiser import Denoiser
from symmcd_diffusion.models.noise_schedule import MixedNoiseScheduler


# ── Empirical marginals from QM9 ──────────────────────────────────────────────
# Atom: H=51.6%, C=35.7%, N=4.9%, O=7.8%, F=0.06%
QM9_ATOM_MARGINALS = torch.tensor([0.516, 0.357, 0.049, 0.078, 0.001])
# Bond: none=87.3%, single=11.1%, double=0.74%, triple=0.0%, aromatic=0.89%
QM9_BOND_MARGINALS = torch.tensor([0.873, 0.111, 0.0074, 0.0001, 0.0089])
QM9_BOND_MARGINALS = QM9_BOND_MARGINALS / QM9_BOND_MARGINALS.sum()


# ── 2D Denoiser ───────────────────────────────────────────────────────────────

class Graph2DDenoiser(nn.Module):
    """
    Denoiser for 2D molecular graphs with FiLM symmetry conditioning.
    Wraps the base Denoiser — coord_head is ignored, only atom/bond used.
    """

    def __init__(self, hidden_dim=256, num_layers=9):
        super().__init__()
        self.sym_embed = nn.Embedding(2, SYM_LABEL_DIM)

        self.denoiser = Denoiser(
            num_atom_types=QM9_NUM_ATOM_TYPES,
            num_bond_types=NUM_BOND_TYPES,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_feat_dim=NUM_BOND_TYPES,
            attention=False,
            dropout=0.1,
            condition_dim=SYM_LABEL_DIM,
            use_film=True,
        )

    def encode_condition(self, sym_label):
        return self.sym_embed(sym_label)

    def forward(self, atom_types, positions, t, edge_index,
                edge_attr=None, condition=None, node_mask=None, batch=None):
        return self.denoiser(
            atom_types=atom_types, positions=positions, t=t,
            edge_index=edge_index, edge_attr=edge_attr,
            condition=condition, node_mask=node_mask, batch=batch,
        )

    def load_qm9_backbone(self, ckpt_path, device):
        """Load QM9 weights, skipping FiLM layers (same as JointDenoiser)."""
        print(f"Loading QM9 backbone from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        old_state = ckpt.get("denoiser_state_dict", ckpt)
        model_state = self.denoiser.state_dict()
        loaded, skipped = 0, 0
        for key, value in old_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded += 1
            else:
                skipped += 1
        self.denoiser.load_state_dict(model_state)
        print(f"  Loaded {loaded}, skipped {skipped}")

    def freeze_backbone(self):
        """Freeze EGNN, keep FiLM + atom/bond heads + sym_embed trainable."""
        for p in self.denoiser.parameters():
            p.requires_grad = False
        for name, p in self.denoiser.named_parameters():
            if any(x in name for x in ['atom_head', 'bond_head', 'embedding', 'film']):
                p.requires_grad = True
        for p in self.sym_embed.parameters():
            p.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"  Frozen backbone: {n_trainable:,}/{n_total:,} trainable "
              f"({100*n_trainable/n_total:.1f}%)")


# ── 2D Discrete Diffusion (no coordinates) ────────────────────────────────────

class DiscreteGraphDiffusion(nn.Module):
    """
    Discrete diffusion for 2D molecular graphs.

    Forward: sample t, add noise to atoms and bonds via marginals.
    Reverse: denoiser predicts clean logits → compute CE loss.
    """

    def __init__(self, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.scheduler = MixedNoiseScheduler(
            timesteps=timesteps,
            num_atom_types=QM9_NUM_ATOM_TYPES,
            num_bond_types=NUM_BOND_TYPES,
            schedule="cosine",
        )

    def training_step(self, data, denoiser, atom_marginals, bond_marginals,
                      condition=None):
        """Single training step: noise → predict → CE loss."""
        device = data.x.device

        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = int(data.batch.max().item()) + 1
            batch = data.batch
        else:
            batch_size = 1
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # Sample timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)

        # Forward diffusion (discrete only — no coordinates)
        at, a0 = self.scheduler.forward_discrete_batched(
            data.x, t, batch, atom_marginals.to(device), "atom")

        et, e0 = None, None
        if data.edge_attr is not None:
            et, e0 = self.scheduler.forward_discrete_batched(
                data.edge_attr, t, batch, bond_marginals.to(device), "bond",
                edge_index=data.edge_index)

        # Dummy coordinates (zeros — distances uniform, EGNN ignores them)
        dummy_pos = torch.zeros(data.x.size(0), 3, device=device)

        # Denoiser forward
        pred = denoiser(
            atom_types=at,
            positions=dummy_pos,
            t=t[batch],
            edge_index=data.edge_index,
            edge_attr=et,
            condition=condition,
        )

        # Cross-entropy loss
        atom_loss = F.cross_entropy(
            pred["atom_logits"], a0.argmax(dim=-1),
            reduction='mean')
        bond_loss = F.cross_entropy(
            pred["bond_logits"], e0.argmax(dim=-1),
            reduction='mean') if et is not None else torch.tensor(0.0, device=device)

        total_loss = atom_loss + bond_loss

        return total_loss, {
            "atom_loss": atom_loss.item(),
            "bond_loss": bond_loss.item(),
            "total_loss": total_loss.item(),
        }


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, diffusion, loader, optimizer, scaler,
                atom_marg, bond_marg, device, use_amp,
                epoch, total_epochs, grad_accum=2):
    model.train()
    losses = {"atom": 0, "bond": 0, "total": 0}
    n_batches = 0
    nan_count = 0

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        if hasattr(batch, 'sym_label') and batch.sym_label is not None:
            condition = model.encode_condition(batch.sym_label)
        else:
            condition = None

        with autocast(enabled=use_amp):
            loss, loss_dict = diffusion.training_step(
                data=batch, denoiser=model,
                atom_marginals=atom_marg,
                bond_marginals=bond_marg,
                condition=condition,
            )
            loss = loss / grad_accum

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"  [WARN] NaN at batch {batch_idx+1}")
            if nan_count >= 20:
                print(f"  [FATAL] Too many NaN, aborting")
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

        losses["atom"] += loss_dict["atom_loss"]
        losses["bond"] += loss_dict["bond_loss"]
        losses["total"] += loss_dict["total_loss"]
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Epoch {epoch}/{total_epochs} batch {batch_idx+1}: "
                  f"loss={loss_dict['total_loss']:.3f} "
                  f"atom={loss_dict['atom_loss']:.3f} "
                  f"bond={loss_dict['bond_loss']:.3f}")

    for k in losses:
        losses[k] /= max(n_batches, 1)
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=9)

    parser.add_argument("--qm9_root", type=str, default="./data/qm9")
    parser.add_argument("--sym_dir", type=str,
                        default="/home/tianyajun/MARL_for_COFS/data/symmetric_molecules")
    parser.add_argument("--qm9_ckpt", type=str,
                        default="symmcd_diffusion/checkpoints/qm9_best.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--ckpt_dir", type=str,
                        default="symmcd_diffusion/checkpoints/2d_graph")

    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 2D Graph Diffusion Training")
    print(f"Device: {device}")
    print(f"Atom marginals: {QM9_ATOM_MARGINALS.tolist()}")
    print(f"Bond marginals: {QM9_BOND_MARGINALS.tolist()}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = JointDataset(
        qm9_root=args.qm9_root, sym_data_dir=args.sym_dir, split="train")
    train_loader = train_dataset.get_dataloader(batch_size=args.batch_size)

    atom_marginals = QM9_ATOM_MARGINALS.to(device)
    bond_marginals = QM9_BOND_MARGINALS.to(device)

    print(f"Train: {len(train_dataset)} (QM9: {train_dataset._qm9_len}, "
          f"Sym: {train_dataset._sym_len})")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = Graph2DDenoiser(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)

    if os.path.exists(args.qm9_ckpt):
        model.load_qm9_backbone(args.qm9_ckpt, device)
    else:
        print(f"  [WARN] QM9 checkpoint not found, training from scratch")

    diffusion = DiscreteGraphDiffusion(timesteps=1000).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_total:,}")

    # ── Stage A: Freeze backbone, train FiLM + heads ─────────────────────────
    print(f"\n{'='*60}")
    print("Stage A: Freeze EGNN backbone, train FiLM + output heads")
    print(f"{'='*60}")

    model.freeze_backbone()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.use_amp)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()
        losses = train_epoch(
            model, diffusion, train_loader, optimizer, scaler,
            atom_marginals, bond_marginals, device, args.use_amp,
            epoch + 1, args.epochs, args.grad_accum)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"loss={losses['total']:.4f} atom={losses['atom']:.4f} "
              f"bond={losses['bond']:.4f} ({elapsed:.1f}s)")

        if losses['total'] < best_loss:
            best_loss = losses['total']
            torch.save({
                'epoch': epoch, 'loss': best_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.ckpt_dir, 'best.pt'))

        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch, 'loss': losses['total'],
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.ckpt_dir, f'epoch{epoch+1}.pt'))

    # Final save
    torch.save({
        'epoch': args.epochs, 'loss': losses['total'],
        'model_state_dict': model.state_dict(),
    }, os.path.join(args.ckpt_dir, 'final.pt'))

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {args.ckpt_dir}")


if __name__ == "__main__":
    main()
