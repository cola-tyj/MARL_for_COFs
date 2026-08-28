"""在冻结 1/4/32 COF tier 上训练并严格审计官方 UAE-3D autoencoder。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np

from generative_model.data import COFSymmetryDataset
from generative_model.data.build_cof_package import save_deterministic_npz
from generative_model.models.uae3d_bridge import (
    DEFAULT_SOURCE,
    OFFICIAL_GEOM_POSITION_STD,
    assert_uae3d_official_source,
    canonical_to_uae3d_data,
    uae3d_model_args,
)
from generative_model.models.uae3d_reconstruction import (
    RECONSTRUCTION_METRICS_VERSION,
    evaluate_mean_reconstruction,
    reconstruction_gate,
)
from generative_model.models.uae3d_hierarchical_bonds import (
    constrained_hierarchical_predictions,
)
from generative_model.models.uae3d_objectives import (
    EXACT_RECONSTRUCTION_OBJECTIVE_VERSION,
    HARD_PAIR_OBJECTIVE_VERSION,
    HIERARCHICAL_BOND_OBJECTIVE_VERSION,
    deterministic_mean_auxiliaries,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "generative_model/data/processed/v2"
DEFAULT_SPLIT = ROOT / "generative_model/smoke/splits/uae3d_v1.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_fingerprint(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _configure_trainable_scope(model, scope: str) -> list[str]:
    """显式冻结参数，并返回进入 optimizer 的稳定名称顺序。"""

    if scope not in {"all", "bond-head"}:
        raise ValueError(f"未知 trainable scope: {scope}")
    trainable_names = []
    for name, parameter in model.named_parameters():
        trainable = scope == "all" or name.startswith("decoder.bond_head.")
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError(f"trainable scope {scope} 没有参数")
    if scope == "bond-head" and not all(
        name.startswith("decoder.bond_head.") for name in trainable_names
    ):
        raise RuntimeError("bond-head scope 泄漏到其他参数")
    return trainable_names


def _parameter_fingerprint(model, names: list[str]) -> str:
    parameters = dict(model.named_parameters())
    missing = [name for name in names if name not in parameters]
    if missing:
        raise RuntimeError(f"parameter fingerprint 缺少参数: {missing}")
    return _state_fingerprint({name: parameters[name] for name in names})


def _project_conflicting_gradients(vectors):
    """按固定 strata 顺序执行确定性 PCGrad 投影。"""

    import torch

    if len(vectors) < 2 or any(vector.ndim != 1 for vector in vectors):
        raise ValueError("PCGrad 至少需要两个一维 gradient vectors")
    if len({int(vector.numel()) for vector in vectors}) != 1:
        raise ValueError("PCGrad gradient vectors 长度不一致")
    if not all(bool(torch.isfinite(vector).all()) for vector in vectors):
        raise ValueError("PCGrad gradient vector 含 NaN/Inf")
    conflict_pairs = sum(
        float(torch.dot(vectors[left], vectors[right])) < 0
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    )
    projected = [vector.clone() for vector in vectors]
    for left in range(len(projected)):
        for right in range(len(vectors)):
            if left == right:
                continue
            denominator = torch.dot(vectors[right], vectors[right])
            if float(denominator) == 0.0:
                continue
            dot = torch.dot(projected[left], vectors[right])
            if float(dot) < 0.0:
                projected[left] = projected[left] - dot / denominator * vectors[right]
    return projected, conflict_pairs


def _pcgrad_backward(task_losses, parameters) -> tuple[float, int]:
    """计算 task gradients、执行固定顺序 PCGrad，并写入 ``parameter.grad``。"""

    import torch

    raw_vectors = []
    for index, loss in enumerate(task_losses):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index < len(task_losses) - 1,
            allow_unused=True,
        )
        raw_vectors.append(torch.cat([
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]))
    projected, conflict_pairs = _project_conflicting_gradients(raw_vectors)
    aggregate = torch.stack(projected).mean(dim=0)
    if not bool(torch.isfinite(aggregate).all()):
        raise RuntimeError("PCGrad aggregate gradient 含 NaN/Inf")
    begin = 0
    for parameter in parameters:
        end = begin + parameter.numel()
        parameter.grad = aggregate[begin:end].reshape_as(parameter).clone()
        begin = end
    if begin != aggregate.numel():
        raise RuntimeError("PCGrad gradient offsets 不一致")
    return float(torch.linalg.vector_norm(aggregate).detach().cpu()), conflict_pairs


def _seed_everything(seed: int, deterministic: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.set_float32_matmul_precision("highest" if deterministic else "high")
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _load_frozen_data(args: argparse.Namespace) -> tuple[dict[str, Any], list[Any]]:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if split.get("schema_version") != "1.0" or split.get("model_family") != "UAE-3D":
        raise ValueError("UAE-3D frozen split schema/model_family 错误")
    indices = list(map(int, split["tier_indices"][str(args.tier)]))
    if len(indices) != args.tier or len(indices) != len(set(indices)):
        raise ValueError("冻结 tier 的数量或唯一性错误")
    dataset = COFSymmetryDataset(args.package_dir)
    if dataset.manifest["dataset_fingerprint"] != split["dataset_fingerprint"]:
        raise RuntimeError("canonical dataset fingerprint 与 UAE split 不一致")
    data_list = [
        canonical_to_uae3d_data(
            dataset[index],
            source_root=args.source,
            position_std=args.position_std,
            verify_source=False,
        )
        for index in indices
    ]
    if [int(data.idx) for data in data_list] != indices:
        raise RuntimeError("UAE tier package index 顺序发生变化")
    return split, data_list


def _step_indices(size: int, batch_size: int, step: int, seed: int) -> list[int]:
    """按 epoch 独立重建 permutation，使断点恢复不依赖 sampler 隐状态。"""

    import torch

    batches_per_epoch = (size + batch_size - 1) // batch_size
    epoch = (step - 1) // batches_per_epoch
    batch_index = (step - 1) % batches_per_epoch
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    order = torch.randperm(size, generator=generator).tolist()
    begin = batch_index * batch_size
    return order[begin : begin + batch_size]


def _augment_batch(batch, *, step: int, seed: int, rotation: bool,
                   translation: bool, translation_scale: float):
    """复现官方“归一化后旋转/平移”的语义，随机量由 step 唯一确定。"""

    if not rotation and not translation:
        return batch
    import torch
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed + 10_000_019 * step)
    graph_count = int(batch.ptr.numel() - 1)
    if rotation:
        matrices = Rotation.random(graph_count, random_state=rng).as_matrix().astype(np.float32)
        matrices = torch.as_tensor(matrices, device=batch.pos.device)
        batch.pos = torch.bmm(
            batch.pos.unsqueeze(1), matrices[batch.batch].transpose(1, 2)
        ).squeeze(1)
    if translation:
        shifts = rng.normal(0.0, translation_scale, size=(graph_count, 3)).astype(np.float32)
        shifts = torch.as_tensor(shifts, device=batch.pos.device)
        batch.pos = batch.pos + shifts[batch.batch]
    return batch


def _checkpoint_payload(model, optimizer, step: int, identity: dict[str, Any],
                        losses: list[dict[str, float]], evaluations: list[dict[str, Any]],
                        initial_evaluation: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        "schema_version": "uae3d-reconstruction-checkpoint-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "identity": identity,
        "losses": losses,
        "evaluations": evaluations,
        "initial_evaluation": initial_evaluation,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_rng(checkpoint: dict[str, Any], device) -> None:
    import torch

    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])


def _default_output(args: argparse.Namespace) -> Path:
    augmentation = "aug" if args.aug_rotation or args.aug_translation else "noaug"
    learning_rate = f"{args.learning_rate:.0e}".replace("-", "m")
    objective = (
        "exactaux"
        if (
            args.balanced_bond_aux_weight > 0
            or args.mean_coordinate_aux_weight > 0
            or args.hierarchical_bond_weight > 0
        )
        else "official"
    )
    return (
        ROOT / f"generative_model/runs/uae3d_overfit{args.tier}"
        / f"train_{args.max_steps:04d}_lr{learning_rate}_{augmentation}_{objective}"
    )


def _checkpoint_base_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """移除分支来源；其余字段用于 resume 的严格一致性检查。"""

    return {key: value for key, value in identity.items() if key != "initialization"}


def _validate_initialization_checkpoint(
    checkpoint: dict[str, Any], expected_identity: dict[str, Any],
    *, allow_tier_expansion: bool = False,
) -> None:
    """只允许同任务分叉，或显式允许冻结嵌套 tier 的 1→4→32 扩展。"""

    if checkpoint.get("schema_version") != "uae3d-reconstruction-checkpoint-v1":
        raise RuntimeError("init checkpoint schema 错误")
    source_identity = checkpoint.get("identity", {})
    immutable_keys = (
        "profile",
        "position_std",
        "official_source_commit",
        "package_manifest_sha256",
        "dataset_fingerprint",
        "split_sha256",
        "split_fingerprint",
        "model_args",
    )
    mismatches = [
        key for key in immutable_keys
        if source_identity.get(key) != expected_identity.get(key)
    ]
    if mismatches:
        raise RuntimeError(f"init checkpoint 与当前 frozen task 不一致: {mismatches}")
    source_tier = source_identity.get("tier")
    target_tier = expected_identity.get("tier")
    source_indices = source_identity.get("package_indices")
    target_indices = expected_identity.get("package_indices")
    if allow_tier_expansion:
        allowed_transitions = {(1, 4), (1, 32), (4, 32)}
        if (source_tier, target_tier) not in allowed_transitions:
            raise RuntimeError(
                f"tier expansion 只允许 1→4、1→32 或 4→32，实际为 "
                f"{source_tier}→{target_tier}"
            )
        if not isinstance(source_indices, list) or not isinstance(target_indices, list):
            raise RuntimeError("tier expansion package_indices schema 错误")
        if not set(source_indices).issubset(target_indices):
            raise RuntimeError("tier expansion 来源索引不是目标冻结 tier 的子集")
        if len(source_indices) != source_tier or len(target_indices) != target_tier:
            raise RuntimeError("tier expansion 索引数量与 tier 不一致")
    elif source_tier != target_tier or source_indices != target_indices:
        raise RuntimeError("init checkpoint tier/package_indices 与当前 frozen task 不一致")
    if int(checkpoint.get("step", -1)) <= 0:
        raise RuntimeError("init checkpoint step 非法")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_commit = assert_uae3d_official_source(args.source)
    if args.tier not in {1, 4, 32}:
        raise ValueError("tier 只允许 1/4/32")
    if args.max_steps <= 0 or args.checkpoint_every <= 0:
        raise ValueError("max_steps 和 checkpoint_every 必须为正数")
    if args.batch_size <= 0 or args.batch_size > args.tier:
        raise ValueError("batch_size 必须为 1..tier")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip <= 0:
        raise ValueError("optimizer 参数非法")
    if args.official_loss_weight < 0:
        raise ValueError("official loss 权重不得为负")
    if args.aug_translation_scale < 0 or args.coordinate_rmsd_limit <= 0:
        raise ValueError("augmentation/gate 参数非法")
    if (
        args.balanced_bond_aux_weight < 0
        or args.mean_coordinate_aux_weight < 0
        or args.hard_pair_ce_weight < 0
        or args.hard_pair_margin_weight < 0
        or args.hierarchical_bond_weight < 0
    ):
        raise ValueError("auxiliary loss 权重不得为负")
    hard_pair_enabled = args.hard_pair_ce_weight > 0 or args.hard_pair_margin_weight > 0
    hierarchical_enabled = args.hierarchical_bond_weight > 0
    if hard_pair_enabled and hierarchical_enabled:
        raise ValueError("hard-pair 与 hierarchical objective 不允许同时启用")
    if hard_pair_enabled and args.hard_pair_margin <= 0:
        raise ValueError("hard-pair margin 必须为正")
    if args.gradient_surgery == "pcgrad-strata":
        if not hard_pair_enabled or args.trainable_scope != "bond-head":
            raise ValueError("pcgrad-strata 只允许 hard-pair bond-head 训练")
        if (
            args.official_loss_weight != 0
            or args.balanced_bond_aux_weight != 0
            or args.mean_coordinate_aux_weight != 0
        ):
            raise ValueError("pcgrad-strata 要求 official/balanced/coordinate 权重为 0")
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume 与 --init-checkpoint 互斥")
    output_dir = args.output_dir or _default_output(args)
    existing = [] if not output_dir.exists() else [
        path for path in output_dir.iterdir() if path.name != "console.log"
    ]
    if existing and args.resume is None:
        raise FileExistsError(f"输出目录已有训练产物；请换目录或显式 --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    source_text = str(args.source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import torch
    import torch_geometric
    from rdkit import rdBase
    from torch_geometric.data import Batch
    from model.autoencoder.unified_autoencoder import UnifiedAutoEncoder

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    _seed_everything(args.seed, args.deterministic)
    split, data_list = _load_frozen_data(args)
    model_args = uae3d_model_args(args.profile)
    model = UnifiedAutoEncoder(model_args).to(device)
    trainable_names = _configure_trainable_scope(model, args.trainable_scope)
    trainable_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name in set(trainable_names)
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def evaluate_current_model():
        if not hierarchical_enabled:
            return evaluate_mean_reconstruction(
                model, data_list, device, position_std=args.position_std
            )
        return evaluate_mean_reconstruction(
            model,
            data_list,
            device,
            position_std=args.position_std,
            bond_prediction_transform=lambda logits, counts: (
                constrained_hierarchical_predictions(
                    logits,
                    counts,
                    present_score_reduction=args.hierarchical_present_score,
                )
            ),
            bond_decode_mode=(
                "hierarchical-structural-self-"
                f"{args.hierarchical_present_score}-present"
            ),
        )
    package_manifest_sha = _sha256_file(args.package_dir / "manifest.json")
    split_sha = _sha256_file(args.split)
    run_identity = {
        "tier": args.tier,
        "package_indices": [int(data.idx) for data in data_list],
        "profile": args.profile,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "position_std": args.position_std,
        "aug_rotation": args.aug_rotation,
        "aug_translation": args.aug_translation,
        "aug_translation_scale": args.aug_translation_scale,
        "deterministic": args.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "official_source_commit": source_commit,
        "package_manifest_sha256": package_manifest_sha,
        "dataset_fingerprint": split["dataset_fingerprint"],
        "split_sha256": split_sha,
        "split_fingerprint": split["split_fingerprint"],
        "model_args": vars(model_args),
    }
    auxiliary_enabled = (
        args.balanced_bond_aux_weight > 0
        or args.mean_coordinate_aux_weight > 0
        or hard_pair_enabled
        or hierarchical_enabled
    )
    if auxiliary_enabled:
        run_identity.update({
            "objective_version": (
                HARD_PAIR_OBJECTIVE_VERSION
                if hard_pair_enabled
                else (
                    HIERARCHICAL_BOND_OBJECTIVE_VERSION
                    if hierarchical_enabled
                    else EXACT_RECONSTRUCTION_OBJECTIVE_VERSION
                )
            ),
            "balanced_bond_aux_weight": args.balanced_bond_aux_weight,
            "mean_coordinate_aux_weight": args.mean_coordinate_aux_weight,
        })
    if args.trainable_scope != "all":
        run_identity["trainable_scope"] = args.trainable_scope
        run_identity["trainable_parameter_names"] = trainable_names
    if hard_pair_enabled:
        run_identity.update({
            "official_loss_weight": args.official_loss_weight,
            "hard_pair_ce_weight": args.hard_pair_ce_weight,
            "hard_pair_margin_weight": args.hard_pair_margin_weight,
            "hard_pair_margin": args.hard_pair_margin,
        })
    if hierarchical_enabled:
        run_identity.update({
            "official_loss_weight": args.official_loss_weight,
            "hierarchical_bond_weight": args.hierarchical_bond_weight,
            "hierarchical_present_score": args.hierarchical_present_score,
            "hierarchical_structural_self": True,
        })
    if args.gradient_surgery != "none":
        run_identity["gradient_surgery"] = args.gradient_surgery
    identity = run_identity
    initialization: dict[str, Any] = {"mode": "random_initialization"}
    initial_step = 0
    losses: list[dict[str, float]] = []
    evaluations: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != "uae3d-reconstruction-checkpoint-v1":
            raise RuntimeError("resume checkpoint schema 错误")
        if _checkpoint_base_identity(checkpoint["identity"]) != run_identity:
            raise RuntimeError("resume checkpoint identity 与当前任务不一致")
        identity = checkpoint["identity"]
        initialization = identity.get("initialization", initialization)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_step = int(checkpoint["step"])
        if initial_step >= args.max_steps:
            raise ValueError("resume step 必须小于 max_steps")
        losses = list(checkpoint["losses"])
        evaluations = list(checkpoint["evaluations"])
        initial_evaluation = checkpoint["initial_evaluation"]
        legacy_version = "uae3d-mean-reconstruction-v1.0-diagonal-precheck"
        initial_evaluation.setdefault("schema_version", legacy_version)
        for evaluation in evaluations:
            evaluation["metrics"].setdefault("schema_version", legacy_version)
        _restore_rng(checkpoint, device)
    else:
        if args.init_checkpoint is not None:
            checkpoint = torch.load(
                args.init_checkpoint, map_location="cpu", weights_only=False
            )
            _validate_initialization_checkpoint(
                checkpoint,
                run_identity,
                allow_tier_expansion=args.allow_init_tier_expansion,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            source_tier = int(checkpoint["identity"]["tier"])
            initialization = {
                "mode": (
                    "model_only_tier_expansion"
                    if args.allow_init_tier_expansion
                    else "model_only_checkpoint"
                ),
                "checkpoint_path": str(args.init_checkpoint.resolve()),
                "checkpoint_sha256": _sha256_file(args.init_checkpoint),
                "source_step": int(checkpoint["step"]),
                "source_tier": source_tier,
                "target_tier": args.tier,
                "source_package_indices": checkpoint["identity"]["package_indices"],
                "target_package_indices": run_identity["package_indices"],
                "source_model_state_fingerprint": _state_fingerprint(
                    checkpoint["model_state_dict"]
                ),
                "optimizer_reinitialized": True,
                "rng_reinitialized_from_branch_seed": True,
            }
            identity = {**run_identity, "initialization": initialization}
        initial_evaluation = evaluate_current_model()
        evaluations.append({"step": 0, "metrics": initial_evaluation})

    trainable_name_set = set(trainable_names)
    frozen_names = [
        name for name, _ in model.named_parameters() if name not in trainable_name_set
    ]
    frozen_fingerprint_before = _parameter_fingerprint(model, frozen_names)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for step in range(initial_step + 1, args.max_steps + 1):
        local_indices = _step_indices(len(data_list), args.batch_size, step, args.seed)
        batch = Batch.from_data_list([data_list[index] for index in local_indices]).to(device)
        batch = _augment_batch(
            batch,
            step=step,
            seed=args.seed,
            rotation=args.aug_rotation,
            translation=args.aug_translation,
            translation_scale=args.aug_translation_scale,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(batch)
        official_loss = loss_dict["loss"]
        if auxiliary_enabled:
            auxiliary = deterministic_mean_auxiliaries(
                model,
                batch,
                hard_pair_margin=(args.hard_pair_margin if hard_pair_enabled else None),
                hierarchical_present_score=(
                    args.hierarchical_present_score if hierarchical_enabled else None
                ),
            )
            exact_aux_loss = (
                args.balanced_bond_aux_weight * auxiliary["balanced_bond_loss"]
                + args.mean_coordinate_aux_weight * auxiliary["mean_coordinate_loss"]
                + args.hard_pair_ce_weight * auxiliary["hard_pair_ce_loss"]
                + args.hard_pair_margin_weight * auxiliary["hard_pair_margin_loss"]
                + args.hierarchical_bond_weight
                * auxiliary["hierarchical_bond_loss"]
            )
        else:
            zero = official_loss.new_zeros(())
            auxiliary = {
                "balanced_bond_loss": zero,
                "bond_none_stratum_loss": zero,
                "bond_present_stratum_loss": zero,
                "bond_self_stratum_loss": zero,
                "bond_none_pairs": 0,
                "bond_present_pairs": 0,
                "bond_self_pairs": 0,
                "mean_coordinate_loss": zero,
                "hard_pair_ce_loss": zero,
                "hard_pair_margin_loss": zero,
                "hard_pair_none_ce_loss": zero,
                "hard_pair_present_ce_loss": zero,
                "hard_pair_self_ce_loss": zero,
                "hard_pair_none_margin_loss": zero,
                "hard_pair_present_margin_loss": zero,
                "hard_pair_self_margin_loss": zero,
                "hard_pair_count": 0,
                "hard_pair_none_count": 0,
                "hard_pair_present_count": 0,
                "hard_pair_self_count": 0,
                "hierarchical_bond_loss": zero,
                "bond_existence_loss": zero,
                "bond_existence_none_loss": zero,
                "bond_existence_present_loss": zero,
                "conditional_bond_type_loss": zero,
                "bond_existence_correct": 0,
                "bond_existence_total": 0,
                "conditional_bond_type_correct": 0,
                "conditional_bond_type_total": 0,
            }
            exact_aux_loss = zero
        total_loss = args.official_loss_weight * official_loss + exact_aux_loss
        if not bool(torch.isfinite(total_loss)):
            raise RuntimeError(f"step {step} loss 含 NaN/Inf")
        pcgrad_conflict_pairs = 0
        pcgrad_gradient_norm = None
        if args.gradient_surgery == "pcgrad-strata":
            task_losses = [
                args.hard_pair_ce_weight * auxiliary[f"hard_pair_{name}_ce_loss"]
                + args.hard_pair_margin_weight
                * auxiliary[f"hard_pair_{name}_margin_loss"]
                for name in ("none", "present", "self")
            ]
            pcgrad_gradient_norm, pcgrad_conflict_pairs = _pcgrad_backward(
                task_losses, trainable_parameters
            )
        else:
            total_loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError(f"step {step} gradient 缺失或含 NaN/Inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, args.gradient_clip
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"step {step} gradient norm 含 NaN/Inf")
        optimizer.step()
        record = {
            "step": step,
            "gradient_norm_before_clip": float(grad_norm.cpu()),
            "pcgrad_conflict_pairs": int(pcgrad_conflict_pairs),
            "pcgrad_gradient_norm_before_clip": (
                float(pcgrad_gradient_norm)
                if pcgrad_gradient_norm is not None else np.nan
            ),
            "official_loss": float(official_loss.detach().cpu()),
            "exact_aux_loss": float(exact_aux_loss.detach().cpu()),
            "balanced_bond_loss": float(auxiliary["balanced_bond_loss"].detach().cpu()),
            "bond_none_stratum_loss": float(
                auxiliary["bond_none_stratum_loss"].detach().cpu()
            ),
            "bond_present_stratum_loss": float(
                auxiliary["bond_present_stratum_loss"].detach().cpu()
            ),
            "bond_self_stratum_loss": float(
                auxiliary["bond_self_stratum_loss"].detach().cpu()
            ),
            "mean_coordinate_loss": float(
                auxiliary["mean_coordinate_loss"].detach().cpu()
            ),
            "hard_pair_ce_loss": float(
                auxiliary["hard_pair_ce_loss"].detach().cpu()
            ),
            "hard_pair_margin_loss": float(
                auxiliary["hard_pair_margin_loss"].detach().cpu()
            ),
            "hard_pair_none_ce_loss": float(
                auxiliary["hard_pair_none_ce_loss"].detach().cpu()
            ),
            "hard_pair_present_ce_loss": float(
                auxiliary["hard_pair_present_ce_loss"].detach().cpu()
            ),
            "hard_pair_self_ce_loss": float(
                auxiliary["hard_pair_self_ce_loss"].detach().cpu()
            ),
            "hard_pair_none_margin_loss": float(
                auxiliary["hard_pair_none_margin_loss"].detach().cpu()
            ),
            "hard_pair_present_margin_loss": float(
                auxiliary["hard_pair_present_margin_loss"].detach().cpu()
            ),
            "hard_pair_self_margin_loss": float(
                auxiliary["hard_pair_self_margin_loss"].detach().cpu()
            ),
            "hard_pair_count": int(auxiliary["hard_pair_count"]),
            "hard_pair_none_count": int(auxiliary["hard_pair_none_count"]),
            "hard_pair_present_count": int(auxiliary["hard_pair_present_count"]),
            "hard_pair_self_count": int(auxiliary["hard_pair_self_count"]),
            "hierarchical_bond_loss": float(
                auxiliary["hierarchical_bond_loss"].detach().cpu()
            ),
            "bond_existence_loss": float(
                auxiliary["bond_existence_loss"].detach().cpu()
            ),
            "bond_existence_none_loss": float(
                auxiliary["bond_existence_none_loss"].detach().cpu()
            ),
            "bond_existence_present_loss": float(
                auxiliary["bond_existence_present_loss"].detach().cpu()
            ),
            "conditional_bond_type_loss": float(
                auxiliary["conditional_bond_type_loss"].detach().cpu()
            ),
            "bond_existence_correct": int(auxiliary["bond_existence_correct"]),
            "bond_existence_total": int(auxiliary["bond_existence_total"]),
            "conditional_bond_type_correct": int(
                auxiliary["conditional_bond_type_correct"]
            ),
            "conditional_bond_type_total": int(
                auxiliary["conditional_bond_type_total"]
            ),
            "bond_none_pairs": int(auxiliary["bond_none_pairs"]),
            "bond_present_pairs": int(auxiliary["bond_present_pairs"]),
            "bond_self_pairs": int(auxiliary["bond_self_pairs"]),
        }
        record.update({name: float(value.detach().cpu()) for name, value in loss_dict.items()})
        record["loss"] = float(total_loss.detach().cpu())
        losses.append(record)

        should_audit = step == args.max_steps or step % args.checkpoint_every == 0
        if should_audit:
            metrics = evaluate_current_model()
            evaluations.append({"step": step, "metrics": metrics})
            print(
                f"step={step}/{args.max_steps} loss={record['loss']:.6f} "
                f"atom_exact={metrics['atom_exact_molecules']}/{args.tier} "
                f"bond_exact={metrics['bond_exact_molecules']}/{args.tier} "
                f"rmsd_max={metrics['coordinate_rmsd_max_angstrom']:.6f}A",
                flush=True,
            )
        if step % args.checkpoint_every == 0 and step < args.max_steps:
            payload = _checkpoint_payload(
                model, optimizer, step, identity, losses, evaluations, initial_evaluation
            )
            _save_checkpoint(checkpoint_dir / f"step-{step:06d}.pt", payload)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final_metrics = evaluations[-1]["metrics"]
    gate = reconstruction_gate(final_metrics, args.coordinate_rmsd_limit)
    frozen_fingerprint_after = _parameter_fingerprint(model, frozen_names)
    if frozen_fingerprint_after != frozen_fingerprint_before:
        raise RuntimeError("trainable scope 之外的参数发生变化")
    last_path = output_dir / "last.pt"
    _save_checkpoint(
        last_path,
        _checkpoint_payload(
            model, optimizer, args.max_steps, identity, losses, evaluations, initial_evaluation
        ),
    )
    loss_path = output_dir / "losses.npz"
    loss_names = (
        "loss", "official_loss", "exact_aux_loss", "balanced_bond_loss",
        "bond_none_stratum_loss", "bond_present_stratum_loss",
        "bond_self_stratum_loss", "mean_coordinate_loss",
        "hard_pair_ce_loss", "hard_pair_margin_loss",
        "hard_pair_none_ce_loss", "hard_pair_present_ce_loss",
        "hard_pair_self_ce_loss", "hard_pair_none_margin_loss",
        "hard_pair_present_margin_loss", "hard_pair_self_margin_loss",
        "hierarchical_bond_loss", "bond_existence_loss",
        "bond_existence_none_loss", "bond_existence_present_loss",
        "conditional_bond_type_loss",
        "atom_loss",
        "bond_loss", "coordinate_loss", "distance_loss", "bond_distance_loss",
        "KLD", "gradient_norm_before_clip",
        "pcgrad_gradient_norm_before_clip",
    )
    count_names = (
        "hard_pair_count", "hard_pair_none_count",
        "hard_pair_present_count", "hard_pair_self_count",
        "pcgrad_conflict_pairs",
        "bond_existence_correct", "bond_existence_total",
        "conditional_bond_type_correct", "conditional_bond_type_total",
    )
    save_deterministic_npz(
        loss_path,
        {
            "step": np.asarray([record["step"] for record in losses], dtype=np.int64),
            **{
                name: np.asarray([
                    record.get(
                        name,
                        record["loss"] if name == "official_loss" else np.nan,
                    )
                    for record in losses
                ], dtype=np.float64)
                for name in loss_names
            },
            **{
                name: np.asarray(
                    [record.get(name, 0) for record in losses], dtype=np.int64
                )
                for name in count_names
            },
        },
    )
    state_fingerprint = _state_fingerprint(model.state_dict())
    report = {
        "schema_version": "1.0",
        "status": "PASS_EXECUTION",
        "purpose": "UAE-3D deterministic-mean reconstruction training gate",
        "current_reconstruction_metrics_version": RECONSTRUCTION_METRICS_VERSION,
        "gate_conclusion": "PASS_U2_TIER" if gate["passed"] else "PENDING_MORE_TRAINING",
        "tier": args.tier,
        "initial_step": initial_step,
        "steps": args.max_steps,
        "batch_size": args.batch_size,
        "identity": identity,
        "initialization": initialization,
        "initial_evaluation": initial_evaluation,
        "evaluation_history": evaluations,
        "final_evaluation": final_metrics,
        "gate": gate,
        "optimization": {
            "optimizer": "AdamW",
            "official_stochastic_vae_objective": True,
            "official_loss_weight": args.official_loss_weight,
            "exact_reconstruction_auxiliary_enabled": auxiliary_enabled,
            "exact_reconstruction_objective_version": (
                (
                    HARD_PAIR_OBJECTIVE_VERSION
                    if hard_pair_enabled
                    else (
                        HIERARCHICAL_BOND_OBJECTIVE_VERSION
                        if hierarchical_enabled
                        else EXACT_RECONSTRUCTION_OBJECTIVE_VERSION
                    )
                )
                if auxiliary_enabled else None
            ),
            "balanced_bond_aux_weight": args.balanced_bond_aux_weight,
            "mean_coordinate_aux_weight": args.mean_coordinate_aux_weight,
            "hard_pair_ce_weight": args.hard_pair_ce_weight,
            "hard_pair_margin_weight": args.hard_pair_margin_weight,
            "hard_pair_margin": args.hard_pair_margin if hard_pair_enabled else None,
            "hierarchical_bond_weight": args.hierarchical_bond_weight,
            "hierarchical_present_score": (
                args.hierarchical_present_score if hierarchical_enabled else None
            ),
            "hierarchical_structural_self": hierarchical_enabled,
            "trainable_scope": args.trainable_scope,
            "gradient_surgery": args.gradient_surgery,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": sum(
                int(parameter.numel()) for parameter in trainable_parameters
            ),
            "frozen_parameter_count": sum(
                int(parameter.numel()) for name, parameter in model.named_parameters()
                if name in set(frozen_names)
            ),
            "frozen_parameters_unchanged": True,
            "frozen_parameter_fingerprint_before": frozen_fingerprint_before,
            "frozen_parameter_fingerprint_after": frozen_fingerprint_after,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "loss_first": losses[0]["loss"],
            "loss_last": losses[-1]["loss"],
            "loss_minimum": min(record["loss"] for record in losses),
            "all_losses_and_gradients_finite": True,
        },
        "model_state_fingerprint": state_fingerprint,
        "elapsed_seconds_this_invocation": elapsed,
        "peak_cuda_memory_bytes_this_invocation": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "artifacts": {
            "losses.npz": _sha256_file(loss_path),
            "last.pt": _sha256_file(last_path),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_geometric": torch_geometric.__version__,
            "rdkit": rdBase.rdkitVersion,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{report['status']} gate={report['gate_conclusion']} report={report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--tier", type=int, choices=(1, 4, 32), required=True)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--position-std", type=float, default=OFFICIAL_GEOM_POSITION_STD)
    parser.add_argument("--coordinate-rmsd-limit", type=float, default=0.05)
    parser.add_argument("--profile", choices=("mini", "official"), default="official")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--aug-rotation", action="store_true")
    parser.add_argument("--aug-translation", action="store_true")
    parser.add_argument("--aug-translation-scale", type=float, default=0.1)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--allow-init-tier-expansion", action="store_true")
    parser.add_argument("--balanced-bond-aux-weight", type=float, default=0.0)
    parser.add_argument("--mean-coordinate-aux-weight", type=float, default=0.0)
    parser.add_argument("--hard-pair-ce-weight", type=float, default=0.0)
    parser.add_argument("--hard-pair-margin-weight", type=float, default=0.0)
    parser.add_argument("--hard-pair-margin", type=float, default=0.5)
    parser.add_argument("--hierarchical-bond-weight", type=float, default=0.0)
    parser.add_argument(
        "--hierarchical-present-score",
        choices=("max", "logsumexp"),
        default="max",
    )
    parser.add_argument("--trainable-scope", choices=("all", "bond-head"), default="all")
    parser.add_argument("--official-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--gradient-surgery", choices=("none", "pcgrad-strata"), default="none"
    )
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
