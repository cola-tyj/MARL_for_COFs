"""
主入口 — 批量生成分子图数据集。

用法：
    # 默认生成 C3 苯环 + 三种取代基，5000 图
    python generate.py

    # 自定义参数
    python generate.py --ring-size 6 --sym 3 --attach-pos 0,2,4 \\
                       --subst methyl,ethyl,isopropyl --num 5000 --seed 42

    # 只生成一种取代基
    python generate.py --subst isopropyl --num 1000
"""

import os
import sys
import argparse
import json
import pickle
import time
import numpy as np

# 支持从任意位置调用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ring_builder import build_ring
from substituents import list_substituents, get_recipe
from assembler import assemble
from utils import shuffle_graph, graph_to_dict, graph_to_smiles, visualize_grid


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate molecular graph dataset with ring + substituents"
    )
    p.add_argument('--ring-size', type=int, default=6,
                   help='Ring size (default: 6)')
    p.add_argument('--sym', type=int, default=3,
                   help='C_n symmetry order (default: 3)')
    p.add_argument('--attach-pos', type=str, default='0,2,4',
                   help='Comma-separated attachment positions on ring (default: 0,2,4)')
    p.add_argument('--subst', type=str, default='methyl,ethyl,isopropyl',
                   help='Comma-separated substituent types (default: methyl,ethyl,isopropyl)')
    p.add_argument('--num', type=int, default=5000,
                   help='Number of graphs to generate (default: 5000)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (default: 42)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: ./output)')
    p.add_argument('--no-shuffle', action='store_true',
                   help='Do not shuffle vertex order')
    p.add_argument('--vis-samples', type=int, default=20,
                   help='Number of samples to visualize (default: 20, 0 to skip)')
    p.add_argument('--name', type=str, default=None,
                   help='Custom output file name prefix')
    return p.parse_args()


def main():
    args = parse_args()

    # ── 解析参数 ──
    attach_positions = [int(x.strip()) for x in args.attach_pos.split(',')]
    subst_types = [x.strip() for x in args.subst.split(',')]
    n_sym = args.sym

    for st in subst_types:
        get_recipe(st)  # 校验取代基是否存在

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output'
    )
    os.makedirs(output_dir, exist_ok=True)

    # ── 文件名 ──
    if args.name:
        base_name = args.name
    else:
        subst_short = '_'.join(st[:3] for st in subst_types)
        base_name = f"ring{args.ring_size}_C{n_sym}_{subst_short}_{args.num}"

    pkl_path = os.path.join(output_dir, f"{base_name}.pkl")
    txt_path = os.path.join(output_dir, f"{base_name}.txt")

    # ── 生成 ──
    print("=" * 60)
    print("Molecular Graph Dataset Generator")
    print("=" * 60)
    print(f"  Ring size:     {args.ring_size}")
    print(f"  Symmetry:      C_{n_sym}")
    print(f"  Attach at:     {attach_positions}")
    print(f"  Substituents:  {subst_types}")
    print(f"  Total graphs:  {args.num}")
    print(f"  Seed:          {args.seed}")
    print(f"  Shuffle:       {not args.no_shuffle}")
    print()

    # 构建环（复用同一对象）
    ring = build_ring(args.ring_size)

    rng = np.random.RandomState(args.seed)
    dataset = []
    subst_counts = {st: 0 for st in subst_types}

    t0 = time.time()
    for i in range(args.num):
        stype = str(rng.choice(subst_types))

        # 拼接
        G = assemble(ring, stype, attach_positions)

        # 打乱
        if not args.no_shuffle:
            G = shuffle_graph(G, seed=args.seed + i)

        # 转换格式
        data = graph_to_dict(G, n_sym=n_sym)
        dataset.append(data)
        subst_counts[stype] += 1

        if (i + 1) % max(1, args.num // 10) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i + 1}/{args.num}] {rate:.0f} graphs/s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({args.num / elapsed:.1f} graphs/s)")

    # ── 保存 .pkl ──
    with open(pkl_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset → {pkl_path}")
    print(f"  Size: {os.path.getsize(pkl_path) / 1024 / 1024:.1f} MB")

    # ── 统计 ──
    nodes = [d['adj'].shape[0] for d in dataset]
    edges = [int(d['adj'].sum() // 2) for d in dataset]

    n_dist = {}
    for n_val in nodes:
        key = str(n_val)
        n_dist[key] = n_dist.get(key, 0) + 1

    meta = {
        'n_graphs': args.num,
        'max_nodes': max(nodes),
        'min_nodes': min(nodes),
        'avg_nodes': sum(nodes) / args.num,
        'avg_edges': sum(edges) / args.num,
        'stats': {
            'subst_type_dist': subst_counts,
            'N_dist': n_dist,
            'total_attempts': args.num,
        },
        'param_space': {
            'ring_size': args.ring_size,
            'symmetry': f'C_{n_sym}',
            'attach_positions': attach_positions,
            'subst_type': subst_types,
        },
        'seed': args.seed,
        'mode': 'ring_subst_topo',
    }

    with open(txt_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {txt_path}")

    # ── 统计输出 ──
    print(f"\nStatistics:")
    print(f"  N range:   {min(nodes)} – {max(nodes)}")
    print(f"  N avg:     {meta['avg_nodes']:.1f}")
    print(f"  Edge avg:  {meta['avg_edges']:.1f}")
    for st in subst_types:
        print(f"  {st:>12}: {subst_counts[st]}")

    # ── 可视化 ──
    if args.vis_samples > 0:
        print(f"\n--- Visualization ({min(args.vis_samples, args.num)} samples) ---")
        # 取每种取代基的若干样本
        vis_graphs = []
        from ring_builder import build_ring as br
        from assembler import assemble as asm
        ring_vis = br(args.ring_size)
        for st in subst_types:
            for si in range(min(args.vis_samples // len(subst_types), 5)):
                G = asm(ring_vis, st, attach_positions)
                if not args.no_shuffle:
                    G = shuffle_graph(G, seed=si * 100)
                vis_graphs.append(G)

        vis_path = os.path.join(output_dir, f"{base_name}_samples.png")
        visualize_grid(vis_graphs, vis_path, n_cols=min(len(vis_graphs), 5))

    print("\nDone!")


if __name__ == '__main__':
    main()
