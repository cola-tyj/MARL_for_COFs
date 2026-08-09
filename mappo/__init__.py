"""
mappo/ — Multi-Agent PPO for COF material design.

Directory structure:
    Core algorithm:
        mappo_mpe.py       — MAPPO implementation (actors, critic, PPO training)
        mappo_mpe_v2.py    — MAPPO v2 with MultiDiscrete action spaces
        replay_buffer.py   — Experience replay buffer for PPO
        normalization.py   — Reward normalization utilities

    Environments:
        env.py              — Original 6-agent vocabulary-based COF construction
        env_v2.py           — 3-agent diffusion-integrated COF design
        env_symtopo.py      — 3-agent symmetry+topology selection (CURRENT)

    Runners:
        run.py              — Training runner for 6-agent COF construction
        run_symtopo.py      — Training runner for symmetry+topology MAPPO (CURRENT)

    Utilities:
        transformer.py      — Transformer-based molecular state encoder
        reward.py           — Reward functions + RND exploration
        ccjson.py           — ChemJSON file reading utilities
        cof_fromname.py     — COF assembly from agent-generated names
        combine_substructures.py — RDKit molecular substructure assembly
        xyz.py              — 2D coordinate generation for molecules

    Pre-trained:
        model/              — Pre-trained embedding and transformer weights

    Legacy (moved to legacy/):
        test.py, test2.py, train.py — Older test/training scripts
        draw.py, y_tsne_plot.py     — Visualization scripts
        rand_gen.py, smiles2cjson.py— Random generation utilities
"""

import sys
import os
PATH = "/home/tianyajun/MARL_for_COFs/pycofbuilder"
sys.path.append(PATH)
