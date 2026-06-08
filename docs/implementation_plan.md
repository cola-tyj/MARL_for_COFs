# Implementation Plan: Symmetry-Conditioned Diffusion for MARL-Guided COF Design

## Context

This plan implements a master's thesis project that replaces the current fixed-vocabulary MAPPO approach (53 discrete building block actions) with a **symmetry-conditioned diffusion model** that generates novel COF molecular building blocks. The project adapts SymmCD's symmetry encoding and asymmetric unit generation paradigm to the molecular domain, integrated into a redesigned 3-agent MARL controller.

**Current state:** The codebase at `D:\Demo\MARL_for_COFs` uses MAPPO with 6 agents selecting from 53 fixed building blocks, assembled via pycofbuilder into 5 topology types (HCB_A, SQL, SQL_A, KGD, HXL_A), all using AA stacking. The system is a closed-loop combinatorial optimizer — it can only recombine known fragments, not create new ones.

**Target state:** A 3-agent MARL system where agents select high-level design parameters (topology, stacking, symmetry type, functional group), a SymmCD-inspired diffusion model generates the actual building blocks conditioned on these parameters, and a self-play loop continuously improves the diffusion model with successful designs.

## Recommended Base Framework: MiDi

Build on **MiDi** (Vignac et al., 2023, "MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation") because:
1. Native mixed continuous (3D coords) + discrete (atom types, bond types) diffusion — matches COF building block representation exactly
2. EGNN-based denoising backbone can be directly augmented with symmetry conditioning
3. Published QM9 training recipes, well-documented PyTorch Geometric codebase
4. Compact enough for a single-developer project

---

## Phase 1: Foundation — Molecular Diffusion Pre-training (Months 1-3)

### Goal
Set up MiDi, pre-train on QM9 (130K molecules), establish baseline molecular generation before adding symmetry conditioning.

### New Directory Structure
```
MARL_for_COFs/
  symmcd_diffusion/                   # New top-level module
    __init__.py
    config/
      base_config.py                  # Base configuration dataclass
      qm9_config.py                   # QM9 pre-training config
      cof_config.py                   # COF-specific config
    data/
      __init__.py
      qm9_dataset.py                  # QM9 loader (PyG)
      cof_dataset.py                  # COF building block dataset
      augmentation.py                 # Data augmentation (194 → 2000)
      cjson_io.py                     # cjson ↔ PyG Data conversion
    models/
      __init__.py
      egnn.py                         # EGNN backbone (from MiDi)
      denoiser.py                     # Mixed diffusion denoiser
      noise_schedule.py               # Noise schedules
      diffusion_process.py            # Forward/reverse diffusion
    symmetry/
      __init__.py
      point_group.py                  # Point group computation
      symmetry_encoder.py             # Binary symmetry encoding (SymmCD-adapted)
    filters/
      __init__.py
      legality_filter.py              # 5-layer legality filter
      connectivity.py                 # Connection point verification
    generation/
      __init__.py
      generator.py                    # End-to-end BB generator
      sampler.py                      # Sampling utilities
    marl_interface/
      __init__.py
      diffusion_env.py                # Diffusion wrapper for MARL
      reward_bridge.py                # Reward computation bridge
    train_qm9.py                      # QM9 pre-training script
    train_symmetry_conditioned.py     # Symmetry-conditioned fine-tuning
    train_self_play.py                # Self-play enhancement loop
```

### Key Files to Create

**`symmcd_diffusion/models/egnn.py`** — Port from MiDi:
- `EGNNLayer`: message function + coordinate update + node update
- `EGNN`: stacked layers with optional attention

**`symmcd_diffusion/models/denoiser.py`** — Core denoising network:
- `Denoiser(nn.Module)`: inputs noisy X/A/E + timestep t → predicts clean X/A/E
- X head: MLP from EGNN node features
- A head: linear from node features
- E head: MLP from edge features

**`symmcd_diffusion/models/diffusion_process.py`**:
- `DiffusionProcess`: `forward()` adds noise, `reverse()` ancestral sampling
- Continuous diffusion (cosine schedule) for coordinates
- Discrete diffusion (uniform transition matrix) for atom types + bond types

**`symmcd_diffusion/data/qm9_dataset.py`**:
- `QM9Dataset(Dataset)`: loads QM9 → PyG Data with `x`, `positions`, `edge_attr`

### QM9 Training Config
```python
hidden_dim: 256
num_layers: 9
diffusion_steps: 1000
batch_size: 64
learning_rate: 1e-4
grad_accumulation: 2  # A6000 24GB constraint
use_amp: True          # mixed precision
```

### Validation Criteria
- RDKit validity > 90%
- Uniqueness > 95%
- Atom type distribution within 5% KL divergence of QM9 training set

### Deliverable
`qm9_denoiser_epoch500.pt`

---

## Phase 2: Symmetry Conditioning (Months 3-5)

### Goal
Add symmetry conditioning to the diffusion model, compute point groups for QM9 molecules, train conditional denoiser, and fine-tune on augmented COF data.

### Key Files to Create

**`symmcd_diffusion/symmetry/point_group.py`**:
- `compute_point_group(atomic_numbers, positions) -> str`
- Algorithm: compute inertia tensor → diagonalize → get principal axes → test candidate point groups via symmetry operations → return best match (0.3Å RMSD tolerance)
- Target groups: C1, C2, C2v, C2h, C3, C3v, D3h, C4, D4h, C6, D6h, D2h, D2d, Td

**`symmcd_diffusion/symmetry/symmetry_encoder.py`**:
- `SymmetryEncoder(nn.Module)`: SymmCD-adapted binary encoding
- 3 axes × 13 operations = 39-bit binary matrix per point group
- MLP: 39 → 64 → 128 (encoding_dim)
- Unlike SymmCD's 15 axes for space groups, we use 3 principal axes (reduced from 15 since molecules lack translational symmetry)

**`symmcd_diffusion/models/conditional_denoiser.py`**:
- `SymmetryConditionedDenoiser`: wraps base denoiser + symmetry encoder + FiLM conditioning
- Condition sources: point_group (128-dim) + num_connectors (128-dim embedding) + func_group (128-dim embedding) → concatenated 384-dim
- FiLM layers in each EGNN layer: h' = γ(condition) * h + β(condition)

**`symmcd_diffusion/data/augmentation.py`**:
- `COFAugmenter`: 194 original → ~2000 augmented samples
- Strategies: random SO(3) rotation ×5, functional group perturbation ×2, scaffold extension ×1.5, position noise ×2

**`symmcd_diffusion/data/cof_dataset.py`**:
- `COFBBDataModule`: loads cjson files → PyG Data with `x`, `positions`, `edge_attr`, `symm_idx`, `num_connectors`, `func_group_type`
- Atom vocabulary: C, N, O, H, F, Cl, Br, S, Q (connector), X (placeholder)

### Fine-tuning Strategy
- Stage A (epochs 1-50): freeze base denoiser, train only FiLM layers + condition embeddings
- Stage B (epochs 50-100): unfreeze last 3 EGNN layers, full fine-tuning
- Auxiliary loss: symmetry classifier (predict point_group from node features), weight 0.1

### Validation Criteria
- Symmetry consistency > 80% (generated molecule's computed point group matches condition)
- Connectivity correctness > 85% (correct number of Q attachment points)
- RDKit validity > 75%

### Deliverable
`symmcd_denoiser_finetuned.pt`

---

## Phase 3: COF Building Block Generator (Months 5-7)

### Goal
Build the end-to-end generator with legality filtering and cjson export compatible with pycofbuilder.

### Key Files to Create

**`symmcd_diffusion/filters/legality_filter.py`** — 5 cascaded layers:
```
Layer 1: Atom-level checks (valence, charge neutrality) — pure geometry
Layer 2: RDKit parsing + SanitizeMol — chemical validity
Layer 3: Symmetry verification — computed point group matches target
Layer 4: Connectivity check — correct number/geometry of Q attachment points
Layer 5: COF assembly test — attempt pycofbuilder assembly with partner block
```
Expected cumulative pass rates: 70% → 60% → 70% → 80% → 70% ≈ 16% overall

**`symmcd_diffusion/filters/connectivity.py`**:
- `ConnectionPointVerifier`: checks Q/X positions match symmetry-expected geometry
  - L2: angle ~180°, T3: angles ~120°, S4: angles ~90°, H6: angles ~60°

**`symmcd_diffusion/data/cjson_io.py`**:
- `CJSONExporter`: diffusion output (atom types/positions/bonds) → ChemJSON → .cjson file
- Reuses `pycofbuilder.tools.smiles_to_xsmiles()` for xsmiles generation

**`symmcd_diffusion/generation/generator.py`**:
- `COFBBGenerator.generate(point_group, num_connectors, func_group, num_samples)`:
  1. Sample from diffusion model with condition
  2. Run 5-layer legality filter
  3. Export valid results to cjson
  4. Rejection sampling until num_samples valid or max_retries

### Validation Criteria
- > 50% of generated samples pass filters (per symmetry type)
- Generated BBs can be assembled into ≥ 3 different COF topologies

### Deliverable
Functioning `COFBBGenerator` integrated with pycofbuilder

---

## Phase 4: MARL Integration (Months 7-10)

### Goal
Connect diffusion generator to MARL, redesign from 6-agent token-selection to 3-agent design-specification, implement self-play improvement loop.

### Key Files to Create/Modify

**`mappo/env_v2.py`** (new, alongside existing `env.py`):
- `COFDesignEnv(gym.Env)` with 3 agents:
  - Agent 0: selects topology (14 types) + stacking (8 modes)
  - Agent 1: selects symmetry (5) + connector count (6) + functional group (10) for BB-A
  - Agent 2: same for BB-B
- `step()`: agents select → diffusion generates BBs → cjson export → COF assembly → predictor → rewards
- Reuses existing `TransformerEncoder` for 128-dim observations

**`mappo/mappo_mpe_v2.py`** (new):
- `Actor_MultiDiscrete`: multi-headed actor for heterogeneous action spaces
- `Critic_MLP`: same as existing, takes global state
- Per-agent action heads: Agent 0 (14+8 dims), Agents 1/2 (5+6+10 dims each)

**`mappo/reward_v2.py`** (new):
- `diffusion_cof_reward()`: bridges diffusion output to existing reward infrastructure
- Reward = N2_adsorption × 0.1 + symmetry_compatibility_bonus + validity_bonus + RND exploration
- Reuses existing `RND` class from `reward.py`

**`symmcd_diffusion/marl_interface/diffusion_env.py`**:
- `DiffusionGeneratorWrapper`: caching layer to avoid re-generating same (symm, conn, fg) combinations
- Pre-generation of candidate pools for common specifications

**`symmcd_diffusion/train_self_play.py`**:
- `SelfPlayLoop`: 5 cycles of (200 MARL episodes → select top-50 BBs by reward → add to diffusion training set → fine-tune 10 epochs)
- Self-play data capped at 20% of total training set to prevent diversity collapse

### Validation Criteria
- MARL + diffusion produces COFs with higher mean N2 adsorption than fixed-vocabulary baseline
- Self-play shows improving trend over cycles (cycle 5 > cycle 3 > cycle 1)
- Topology diversity: at least 8 of 14 topologies represented in generated designs

### Deliverable
End-to-end running system: `python run_v2.py --use-diffusion`

---

## Phase 5: Experiments (Months 10-13)

### Goal
Comprehensive evaluation against baselines, ablation studies, statistical analysis.

### New Directory
```
MARL_for_COFs/
  experiments/
    run_baselines.py      # All baseline implementations
    metrics.py            # Unified metric computation
    ablation.py           # Ablation study runner
    full_comparison.py    # Head-to-head comparison
    plot_results.py       # Publication-quality figures
```

### Baselines
1. **Fixed-vocabulary MAPPO** (existing) — primary comparison
2. **Random baseline** — random topology + stacking + BB selection
3. **Diffusion-only** — random condition sampling, no MARL optimization
4. **Genetic algorithm** — GA on diffusion condition space
5. **Bayesian optimization** — GP over condition space

### Metrics
- Primary: N2/O2 adsorption (mean, max, top-10)
- Quality: validity rate, diversity (Tanimoto), novelty (% not in training), synthesizability
- Efficiency: samples-to-best, time-per-valid-COF
- Symmetry: symmetry consistency, topology coverage

### Ablation Studies (8 dimensions, 5 seeds each)
1. Symmetry conditioning on/off
2. Diffusion steps: 100/250/500/1000
3. FiLM vs concatenation conditioning
4. Legality filter layers: 1-5
5. Self-play cycles: 0/1/3/5
6. Agent count: 3 vs 6
7. RND exploration on/off
8. QM9 pre-training on/off

### Deliverable
All experimental data + publication-quality figures

---

## Phase 6: Thesis Writing (Months 13-15)

- Clean codebase: docstrings, type hints, README
- Reproducibility: Dockerfile, exact dependency versions, pretrained weights
- Supplementary: scripts to reproduce all figures

---

## Gantt Summary
```
Months:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
Phase 1: [███ Foundation ███]
Phase 2:          [███ Symm Cond ███]
Phase 3:                   [███ BB Gen ███]
Phase 4:                           [████ MARL Int ████]
Phase 5:                                        [████ Exp ████]
Phase 6:                                                    [██ Thesis ██]
```

## Critical Existing Files (Reuse)
- `mappo/env.py` → base for `env_v2.py` (Embedding_Layer, TransformerEncoder, vocab/mask logic)
- `mappo/mappo_mpe.py` → base for `mappo_mpe_v2.py` (GAE, PPO clip, value norm)
- `mappo/reward.py` → `RND` class, `predictor()` call pattern for `reward_v2.py`
- `pycofbuilder/framework.py` → `Framework.from_building_blocks()`, 14-topology `TOPOLOGY_DICT`
- `pycofbuilder/cjson.py` → `ChemJSON` class for cjson export
- `pycofbuilder/tools.py` → `smiles_to_xsmiles()` for SMILES↔cjson conversion
- `cof_predictor/main.py` → `doPredict()` for property evaluation

## Key Technical Decisions
1. **MiDi over EDM**: native mixed continuous+discrete diffusion avoids custom discrete diffusion module (~2 months saved)
2. **3 agents vs 6**: diffusion generates complete BBs in one pass; only high-level specs need RL
3. **FiLM conditioning**: per-layer adaptive modulation proven effective in SymmCD
4. **Binary symmetry encoding**: inherited from SymmCD, proven generalizability, reduced from 15→3 axes for molecules
5. **Rejection sampling over constrained sampling**: simpler, debugable, fast early filter layers reject most invalids cheaply
6. **A6000 strategy**: AMP + gradient accumulation + batch_size=8 for diffusion sampling

## Verification Plan
- **Phase 1**: `python symmcd_diffusion/train_qm9.py` → validity > 90%
- **Phase 2**: `python symmcd_diffusion/train_symmetry_conditioned.py` → symmetry consistency > 80%
- **Phase 3**: `python -c "from symmcd_diffusion.generation import COFBBGenerator; ..."` → > 50% pass rate
- **Phase 4**: `python mappo/run_v2.py --use-diffusion --episodes 100` → COFs generated with valid rewards
- **Phase 5**: `python experiments/full_comparison.py` → diffusion+MARL outperforms fixed-vocabulary baseline
