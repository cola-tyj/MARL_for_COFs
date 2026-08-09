"""
Atom Type Predictor (Stage B of de novo generation).

Given 3D coordinates, predicts atom types for each atom.
Trained on Core data as supervised classification.

Input:  positions (N, 3) + edge_index
Output: atom_logits (N, 12)
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW

from .egnn import EGNN


class AtomTypePredictor(nn.Module):
    """Predict atom types from 3D coordinates only (no condition needed)."""

    def __init__(self, num_atom_types=12, hidden_dim=128, num_layers=4):
        super().__init__()
        self.dummy_embed = nn.Parameter(torch.zeros(1, hidden_dim))

        self.egnn = EGNN(hidden_dim=hidden_dim, num_layers=num_layers,
                         edge_feat_dim=0, attention=False,
                         use_film=False, dropout=0.1)

        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_atom_types))

    def forward(self, positions, edge_index):
        """positions: (N, 3). Returns (N, num_types) logits."""
        n = positions.size(0)
        h = self.dummy_embed.expand(n, -1)
        h, _ = self.egnn(h, positions, edge_index, None)
        return self.type_head(h)


def train_predictor(model, loader, device, epochs=50, lr=1e-3):
    """Train on Core data."""
    opt = AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total_loss, total_acc, n = 0, 0, 0
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.positions, batch.edge_index)
            true_atoms = batch.x.argmax(-1)
            loss = F.cross_entropy(logits, true_atoms)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * batch.positions.size(0)
            total_acc += (logits.argmax(-1) == true_atoms).float().sum().item()
            n += batch.positions.size(0)
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={total_loss/n:.4f} acc={total_acc/n:.0%}")
    return model
