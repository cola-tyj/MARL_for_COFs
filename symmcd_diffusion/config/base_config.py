"""Base configuration for the diffusion model."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseDiffusionConfig:
    """Base configuration shared across all diffusion model variants."""

    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 9
    attention: bool = True
    use_cross_attention: bool = False

    # Diffusion process
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"  # "cosine" or "linear"
    diffusion_type: str = "mixed"   # "mixed" (continuous + discrete)

    # Atom type vocabulary
    # QM9: H(0), C(1), N(2), O(3), F(4)
    # COF extended: +Cl, Br, S, Si, Q(connector), X(placeholder)
    num_atom_types: int = 5
    num_bond_types: int = 5  # none, single, double, triple, aromatic

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-12
    warmup_steps: int = 1000
    max_epochs: int = 500
    grad_accumulation: int = 2
    use_amp: bool = True

    # Loss weights
    coord_loss_weight: float = 1.0
    atom_type_loss_weight: float = 1.0
    bond_type_loss_weight: float = 1.0

    # A6000 memory optimization
    gradient_checkpointing: bool = True
    sample_batch_size: int = 8  # smaller for inference

    # Logging
    log_every_n_steps: int = 100
    val_every_n_epochs: int = 5
    save_every_n_epochs: int = 10
    use_wandb: bool = False

    # Paths
    data_dir: str = "/home/tianyajun/MARL_for_COFs/data"
    checkpoint_dir: str = "/home/tianyajun/MARL_for_COFs/symmcd_diffusion/checkpoints"
    log_dir: str = "/home/tianyajun/MARL_for_COFs/symmcd_diffusion/logs"


@dataclass
class QM9Config(BaseDiffusionConfig):
    """Configuration for QM9 pre-training."""
    num_atom_types: int = 5  # H, C, N, O, F
    num_bond_types: int = 5
    max_atoms: int = 29  # QM9 max atoms per molecule


@dataclass
class COFDiffusionConfig(BaseDiffusionConfig):
    """Configuration for COF-specific diffusion training."""
    # Extended atom vocabulary for COF building blocks
    num_atom_types: int = 10  # H, C, N, O, F, Cl, Br, S, Q, X
    num_bond_types: int = 5

    # Symmetry conditioning
    num_point_groups: int = 14
    symmetry_encoding_dim: int = 128
    num_symmetry_axes: int = 3    # molecular (3 axes) vs crystal (15 axes)
    num_symmetry_ops: int = 13    # identity, inversion, C2-6, sigma, S2-6, etc.
    symmetry_binary_dim: int = 39  # 3 * 13

    # Connectivity conditioning
    num_connector_types: int = 7   # 1-6 connectors + none
    num_func_group_types: int = 10
    connector_embedding_dim: int = 128
    func_group_embedding_dim: int = 128

    # Combined condition dimension
    condition_dim: int = 384  # 128 + 128 + 128

    # Fine-tuning
    freeze_base_epochs: int = 50
    unfreeze_epochs: int = 50
    aux_symmetry_loss_weight: float = 0.1
    aux_connectivity_loss_weight: float = 0.1

    # Data augmentation
    num_augmented_samples: int = 2000
    max_atoms: int = 80  # COF BBs can be larger

    # Self-play
    self_play_cycles: int = 5
    self_play_episodes_per_cycle: int = 200
    self_play_top_k: int = 50
    self_play_ft_epochs: int = 10
    self_play_max_data_ratio: float = 0.2  # cap at 20% of training set
