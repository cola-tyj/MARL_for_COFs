"""
marl_interface/ — DEPRECATED (historical MARL experiments E001-E008).

This directory contains iterative MARL environment and training script versions
developed during June 2026. All have been superseded by the mappo/ implementation.

⚠️  Current MARL code is in: /home/tianyajun/MARL_for_COFs/mappo/

Historical versions (kept for thesis documentation):
    env_v2.py              — 2-agent sym+template selection
    env_v3.py              — 3-agent with topology→symmetry mapping
    env_v4.py              — 3-agent sym→topo with diversity bonus
    env_v5.py              — Full action space (connector + functional group)
    env_v6.py              — Rich state (42-dim) + sequential coordination
    train_mappo_2agent.py  — MAPPO for 2-agent
    train_mappo_3agent.py  — MAPPO for 3-agent with RND
    train_mappo_cof.py     — Early COF training attempt
    train_mappo_v5.py      — Multi-head actor for v5
    train_mappo_v6.py      — Rich state training (replaced by mappo/run_symtopo.py)
    diffusion_env.py        — Diffusion model integration (unfinished)
    reward_bridge.py        — Reward bridge for pycofbuilder assembly
    rnd_module.py           — Standalone RND implementation

Experiment results:
    E001: Single-agent pre-validated → 72% success, N2=22.2
    E004: 3-Agent no-RND → 28%
    E005: 3-Agent +RND → 31%
    E006: 3-Agent v4 sym→topo → 10%
    E007: 3-Agent v4b diversity → 8%
    E008: 3-Agent v5 conn+FG → 13%
"""
