"""
Symmetry encoder for molecular point groups.

Adapted from SymmCD (Levy et al., 2024):
In SymmCD, space groups are encoded as binary matrices:
  15 axes x 26 symmetry operations = 390 bits
  (15 possible crystallographic axes, 26 screw/glide/rotation operations)

For molecular point groups, we simplify:
  3 principal axes x 13 symmetry operations = 39 bits
  (molecules lack translational symmetry, only rotational + reflection)

The 13 symmetry operations per axis:
  Identity (1), Inversion (-1),
  Rotations: C2, C3, C4, C6,
  Rotoreflections: S2(=i), S3, S4, S6,
  Mirror: sigma(=S1),
  Combined: 2/m(=C2+sigma_h), 4/m, 6/m

The 3 principal axes are:
  Axis 0: principal axis (highest-order rotation)
  Axis 1: secondary axis (perpendicular to principal)
  Axis 2: tertiary axis (orthogonal to both)
"""

import torch
import torch.nn as nn
from torch import Tensor


# Molecular point groups relevant to COF building blocks
# Grouped by symmetry order
MOLECULAR_POINT_GROUPS = [
    "C1",    # no symmetry (general position)
    "C2",    # 2-fold rotation
    "C2v",   # 2-fold + mirror (L2 linear connector)
    "C2h",   # 2-fold + horizontal mirror
    "C3",    # 3-fold rotation
    "C3v",   # 3-fold + vertical mirrors
    "D3h",   # 3-fold + horizontal mirror + C2' (T3 trigonal core)
    "C4",    # 4-fold rotation
    "C4v",   # 4-fold + vertical mirrors
    "D4h",   # 4-fold + horizontal mirror + C2' (S4 square core)
    "C6",    # 6-fold rotation
    "C6v",   # 6-fold + vertical mirrors
    "D6h",   # 6-fold + horizontal mirror + C2' (H6 hexagonal core)
    "D2h",   # 3 orthogonal C2 + mirrors (extended planar)
]

POINT_GROUP_TO_IDX = {pg: i for i, pg in enumerate(MOLECULAR_POINT_GROUPS)}
IDX_TO_POINT_GROUP = {i: pg for pg, i in POINT_GROUP_TO_IDX.items()}
NUM_POINT_GROUPS = len(MOLECULAR_POINT_GROUPS)

# Symmetry operations per axis (13 total)
SYMMETRY_OPS = [
    "1",    # Identity
    "-1",   # Inversion
    "2",    # C2 rotation
    "3",    # C3 rotation
    "4",    # C4 rotation
    "6",    # C6 rotation
    "m",    # Mirror (sigma = S1)
    "-2",   # S2 = inversion (same as -1 for point groups, kept for compatibility)
    "-3",   # S3 rotoinversion
    "-4",   # S4 rotoinversion
    "-6",   # S6 rotoinversion
    "2/m",  # C2 + perpendicular mirror
    "4/m",  # C4 + perpendicular mirror
]

NUM_AXES = 3
NUM_OPS = len(SYMMETRY_OPS)  # 13
BINARY_DIM = NUM_AXES * NUM_OPS  # 39


def build_point_group_binary_matrix() -> torch.Tensor:
    """
    Build the binary encoding matrix for all molecular point groups.

    Each point group is encoded as a (3, 13) binary matrix:
      - Row i: symmetry operations present along axis i
      - Column j: presence of operation j

    Returns:
        binary_matrix: (NUM_POINT_GROUPS, 39) binary tensor
    """
    binary_matrix = torch.zeros(NUM_POINT_GROUPS, BINARY_DIM)

    for pg_name, pg_idx in POINT_GROUP_TO_IDX.items():
        pg_vec = torch.zeros(NUM_AXES, NUM_OPS)

        if pg_name == "C1":
            # Only identity on all axes
            pg_vec[:, 0] = 1.0

        elif pg_name == "C2":
            pg_vec[0, 0] = 1.0  # Identity on principal
            pg_vec[0, 2] = 1.0  # C2 on principal
            pg_vec[1, 0] = 1.0  # Identity on secondary
            pg_vec[2, 0] = 1.0  # Identity on tertiary

        elif pg_name == "C2v":
            pg_vec[0, 0] = 1.0  # Identity
            pg_vec[0, 2] = 1.0  # C2 on principal
            pg_vec[1, 0] = 1.0  # Identity on secondary
            pg_vec[1, 6] = 1.0  # sigma_v on secondary
            pg_vec[2, 0] = 1.0  # Identity on tertiary
            pg_vec[2, 6] = 1.0  # sigma_v' on tertiary

        elif pg_name == "C2h":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 2] = 1.0  # C2
            pg_vec[0, 1] = 1.0  # inversion
            pg_vec[1, 0] = 1.0
            pg_vec[2, 0] = 1.0
            pg_vec[0, 6] = 1.0  # sigma_h (perpendicular to C2)

        elif pg_name == "C3":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 3] = 1.0  # C3
            pg_vec[1, 0] = 1.0
            pg_vec[2, 0] = 1.0

        elif pg_name == "C3v":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 3] = 1.0  # C3
            pg_vec[1, 0] = 1.0
            pg_vec[1, 6] = 1.0  # sigma_v
            pg_vec[2, 0] = 1.0
            pg_vec[2, 6] = 1.0  # sigma_v'

        elif pg_name == "D3h":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 3] = 1.0  # C3
            pg_vec[0, 2] = 1.0  # C2' (perpendicular to C3)
            pg_vec[1, 0] = 1.0
            pg_vec[1, 6] = 1.0  # sigma_v
            pg_vec[2, 0] = 1.0
            pg_vec[2, 6] = 1.0  # sigma_h (in plane)

        elif pg_name == "C4":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 4] = 1.0  # C4
            pg_vec[0, 2] = 1.0  # C2 (= C4^2)
            pg_vec[1, 0] = 1.0
            pg_vec[2, 0] = 1.0

        elif pg_name == "C4v":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 4] = 1.0  # C4
            pg_vec[0, 2] = 1.0  # C2
            pg_vec[1, 0] = 1.0
            pg_vec[1, 6] = 1.0  # sigma_v
            pg_vec[2, 0] = 1.0
            pg_vec[2, 6] = 1.0  # sigma_d

        elif pg_name == "D4h":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 4] = 1.0  # C4
            pg_vec[0, 2] = 1.0  # C2'
            pg_vec[0, 1] = 1.0  # inversion
            pg_vec[1, 0] = 1.0
            pg_vec[1, 6] = 1.0  # sigma_v
            pg_vec[2, 0] = 1.0
            pg_vec[2, 6] = 1.0  # sigma_h

        elif pg_name == "C6":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 5] = 1.0  # C6
            pg_vec[0, 3] = 1.0  # C3 (= C6^2)
            pg_vec[0, 2] = 1.0  # C2 (= C6^3)
            pg_vec[1, 0] = 1.0
            pg_vec[2, 0] = 1.0

        elif pg_name == "C6v":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 5] = 1.0
            pg_vec[0, 3] = 1.0
            pg_vec[0, 2] = 1.0
            pg_vec[1, 0] = 1.0
            pg_vec[1, 6] = 1.0
            pg_vec[2, 0] = 1.0
            pg_vec[2, 6] = 1.0

        elif pg_name == "D6h":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 5] = 1.0  # C6
            pg_vec[0, 3] = 1.0  # C3
            pg_vec[0, 2] = 1.0  # C2'
            pg_vec[0, 1] = 1.0  # inversion
            pg_vec[1, 0] = 1.0
            pg_vec[1, 6] = 1.0
            pg_vec[2, 0] = 1.0
            pg_vec[2, 6] = 1.0

        elif pg_name == "D2h":
            pg_vec[0, 0] = 1.0
            pg_vec[0, 2] = 1.0  # C2(z)
            pg_vec[0, 1] = 1.0  # inversion
            pg_vec[1, 0] = 1.0
            pg_vec[1, 2] = 1.0  # C2(y)
            pg_vec[2, 0] = 1.0
            pg_vec[2, 2] = 1.0  # C2(x)
            pg_vec[0, 6] = 1.0  # sigma(xy)
            pg_vec[1, 6] = 1.0  # sigma(xz)
            pg_vec[2, 6] = 1.0  # sigma(yz)

        # Flatten to 39-dim vector
        binary_matrix[pg_idx] = pg_vec.reshape(-1)

    return binary_matrix


class SymmetryEncoder(nn.Module):
    """
    Encodes molecular point groups using SymmCD's binary encoding scheme.

    The binary matrix captures structural similarities between point groups:
    e.g., D3h and D4h share similar axis structures but differ in rotation order,
    allowing the model to generalize across symmetry types.
    """

    def __init__(
        self,
        num_groups: int = NUM_POINT_GROUPS,
        encoding_dim: int = 128,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.encoding_dim = encoding_dim
        self.binary_dim = BINARY_DIM

        # Build and register binary codes
        binary_matrix = build_point_group_binary_matrix()
        self.register_buffer("group_binary_code", binary_matrix)

        # MLP: binary code → dense embedding
        self.encoder = nn.Sequential(
            nn.Linear(self.binary_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, encoding_dim),
        )

        # Also provide a learned embedding as fallback/enhancement
        self.learned_embed = nn.Embedding(num_groups, encoding_dim)

        # Fusion weight (learnable)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, point_group_idx: Tensor) -> Tensor:
        """
        Args:
            point_group_idx: (batch,) integer indices into POINT_GROUP_TO_IDX

        Returns:
            symmetry_embedding: (batch, encoding_dim) encoded symmetry
        """
        # Binary encoding
        binary = self.group_binary_code[point_group_idx]  # (batch, 39)
        binary_embed = self.encoder(binary)  # (batch, encoding_dim)

        # Learned embedding
        learned_embed = self.learned_embed(point_group_idx)  # (batch, encoding_dim)

        # Fused embedding
        alpha = torch.sigmoid(self.fusion_alpha)
        fused = alpha * binary_embed + (1 - alpha) * learned_embed

        return fused

    def get_binary_code(self, point_group: str) -> Tensor:
        """Get the raw binary code for a point group (for analysis)."""
        pg_idx = POINT_GROUP_TO_IDX[point_group]
        return self.group_binary_code[pg_idx]
