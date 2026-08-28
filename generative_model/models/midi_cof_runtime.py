"""MiDi 官方模型运行时所需的 COF dataset statistics/info。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _semantic_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_midi_statistics(path: str | Path, dataset: Iterable) -> dict:
    """严格读取并校验外部 MiDi 统计；不接受 vocabulary fallback。"""

    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    semantic_sha = record.pop("semantic_sha256", None)
    if semantic_sha != _semantic_sha256(record):
        raise ValueError("MiDi statistics semantic SHA-256 不一致")
    record["semantic_sha256"] = semantic_sha
    if record.get("status") != "PASS":
        raise ValueError("MiDi statistics 未标记为 PASS")
    if tuple(record.get("atom_vocabulary", ())) != tuple(
        dataset.manifest["atom_vocabulary"]
    ):
        raise ValueError("MiDi statistics atom vocabulary 与数据包不一致")
    if tuple(map(int, record.get("formal_charge_vocabulary", ()))) != tuple(
        map(int, dataset.manifest["formal_charge_vocabulary"])
    ):
        raise ValueError("MiDi statistics charge vocabulary 与数据包不一致")
    if record.get("bond_vocabulary") != dataset.manifest["bond_vocabulary"]:
        raise ValueError("MiDi statistics bond vocabulary 与数据包不一致")
    return record


def build_midi_dataset_info(dataset: Iterable, statistics: dict | None = None):
    """从严格 PyG 数据集构造官方 MiDi 所需的边际分布。

    统计只用于噪声先验和模型接口；canonical ground truth 仍来自 v2/导出 NPZ。
    """

    import torch
    import torch.nn.functional as F
    from midi.datasets.dataset_utils import Statistics
    from midi.diffusion.distributions import DistributionNodes
    from midi.utils import PlaceHolder

    atom_vocab = tuple(dataset.manifest["atom_vocabulary"])
    charge_vocab = tuple(map(int, dataset.manifest["formal_charge_vocabulary"]))
    if statistics is None:
        samples = [dataset[index] for index in range(len(dataset))]
        if not samples:
            raise ValueError("MiDi dataset 不能为空")
        atom_counts = torch.zeros(len(atom_vocab), dtype=torch.float64)
        edge_counts = torch.zeros(5, dtype=torch.float64)
        charge_counts = torch.zeros((len(atom_vocab), len(charge_vocab)), dtype=torch.float64)
        node_counts: Counter[int] = Counter()
        charge_to_index = {charge: index for index, charge in enumerate(charge_vocab)}
        for data in samples:
            node_count = int(data.num_nodes)
            node_counts[node_count] += 1
            atom_counts += F.one_hot(data.x, num_classes=len(atom_vocab)).sum(dim=0)
            directed_pair_count = node_count * (node_count - 1)
            if data.edge_index.shape[1] > directed_pair_count:
                raise ValueError("MiDi directed edge 数超过完整无 self-loop pair 数")
            edge_counts[0] += directed_pair_count - data.edge_index.shape[1]
            edge_counts[1:] += F.one_hot(data.edge_attr - 1, num_classes=4).sum(dim=0)
            for atom, charge in zip(data.x.tolist(), data.charges.tolist()):
                if charge not in charge_to_index:
                    raise ValueError(f"MiDi charge vocabulary 缺少 {charge}")
                charge_counts[atom, charge_to_index[charge]] += 1
        overall_charge = charge_counts.sum(dim=0)
    else:
        counts = statistics.get("counts", {})
        node_counts = Counter({int(k): int(v) for k, v in counts["node_counts"].items()})
        atom_counts = torch.as_tensor(counts["atom_counts"], dtype=torch.float64)
        edge_counts = torch.as_tensor(counts["directed_edge_counts"], dtype=torch.float64)
        charge_counts = torch.as_tensor(
            counts["conditional_charge_counts"], dtype=torch.float64
        )
        overall_charge = torch.as_tensor(
            counts["overall_charge_counts"], dtype=torch.float64
        )
        if atom_counts.shape != (len(atom_vocab),):
            raise ValueError("MiDi external atom counts shape 错误")
        if edge_counts.shape != (5,):
            raise ValueError("MiDi external edge counts shape 错误")
        if charge_counts.shape != (len(atom_vocab), len(charge_vocab)):
            raise ValueError("MiDi external conditional charge counts shape 错误")
        if overall_charge.shape != (len(charge_vocab),):
            raise ValueError("MiDi external overall charge counts shape 错误")
        if min(atom_counts.min(), edge_counts.min(), charge_counts.min(), overall_charge.min()) < 0:
            raise ValueError("MiDi external statistics 含负计数")

    atom_marginals = (atom_counts / atom_counts.sum()).float()
    edge_marginals = (edge_counts / edge_counts.sum()).float()
    row_sums = charge_counts.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1
    conditional_charge = (charge_counts / row_sums).float()
    overall_charge = (overall_charge / overall_charge.sum()).float()

    empty_valencies = {symbol: Counter() for symbol in atom_vocab}
    empty_bond_lengths = {kind: Counter() for kind in range(1, 5)}
    empty_angles = np.zeros((len(atom_vocab), 1801), dtype=np.float64)
    statistics = Statistics(
        num_nodes=node_counts,
        atom_types=atom_marginals,
        bond_types=edge_marginals,
        charge_types=conditional_charge,
        valencies=empty_valencies,
        bond_lengths=empty_bond_lengths,
        bond_angles=empty_angles,
    )

    class COFMiDiDatasetInfo:
        name = "cof_midi_overfit_v1"
        remove_h = False
        need_to_strip = False
        atom_decoder = list(atom_vocab)
        atom_encoder = {symbol: index for index, symbol in enumerate(atom_vocab)}
        num_atom_types = len(atom_vocab)
        collapse_charges = torch.tensor(charge_vocab, dtype=torch.int32)
        input_dims = PlaceHolder(pos=3, X=len(atom_vocab), charges=len(charge_vocab), E=5, y=1)
        output_dims = PlaceHolder(pos=3, X=len(atom_vocab), charges=len(charge_vocab), E=5, y=0)
        n_nodes = None
        nodes_dist = DistributionNodes(dict(node_counts))
        max_n_nodes = max(node_counts)
        atom_types = atom_marginals
        edge_types = edge_marginals
        charges_types = conditional_charge
        charges_marginals = overall_charge
        statistics = {name: statistics for name in ("train", "val", "test")}

        @staticmethod
        def to_one_hot(X, charges, E, node_mask):
            X = F.one_hot(X, num_classes=len(atom_vocab)).float()
            charges = F.one_hot(charges + 2, num_classes=len(charge_vocab)).float()
            E = F.one_hot(E, num_classes=5).float()
            placeholder = PlaceHolder(
                pos=None, X=X, charges=charges, E=E, y=None
            ).mask(node_mask)
            return placeholder.X, placeholder.charges, placeholder.E

        @staticmethod
        def one_hot_charges(charges):
            return F.one_hot(charges + 2, num_classes=len(charge_vocab)).float()

    return COFMiDiDatasetInfo()
