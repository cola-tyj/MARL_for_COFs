"""Train M2.2 local-rotor set head and run the frozen Tier-16 Gates."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .c0_local_recovery import _sha256
from .geometry import build_geometry_contract
from .local_rotor import (
    build_local_rotor_set_contract,
    coupled_circular_errors_degrees,
    coupled_permutation_invariant_circular_loss,
    permutation_invariant_circular_errors_degrees,
    permutation_invariant_circular_loss,
)
from .m0_oracle_reconstruction import build_oracle_targets, optimize_one_start, reconstruction_metrics
from .m1_bond_angle_training import _angle_mae_degrees
from .m2_torsion_training import _learned_decoder_targets
from .m2p1_phase_training import (
    _balanced_schedule,
    _model_setting,
    _prediction_checks,
    _reconstruction_checks,
)
from .m2p2_model import SetValuedQuotientICPredictor, freeze_m2p1_parent, load_m2p1_parent
from .orbit_ic_model import build_orbit_ic_example
from .orbit_kinematics import build_orbit_parameterization
from .overfit import _panel_samples


SCHEMA_VERSION = "pg-orbitflow-m2p2-local-rotor-set-training-v1"


def _dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "pg-orbitflow-m2p2-local-rotor-set-protocol-v1":
        raise ValueError("unsupported M2.2 protocol")
    for section in ("parent", "representation_audit"):
        for item in protocol[section].values():
            if isinstance(item, dict) and "path" in item and _sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"M2.2 frozen identity changed: {section}")
    for name, source in protocol["implementation"].items():
        if _sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"M2.2 implementation changed: {name}")
    return protocol


def _prediction_record(sample, targets, prediction, local) -> dict:
    errors = (
        coupled_circular_errors_degrees(
            prediction["torsion_sincos"], targets.torsion_target_sincos, local
        )
        if local.groups
        else permutation_invariant_circular_errors_degrees(
            prediction["torsion_sincos"], targets.torsion_target_sincos, ()
        )
    )
    angles = np.abs(np.rad2deg(np.arctan2(targets.torsion_target_sincos[:, 0], targets.torsion_target_sincos[:, 1])))
    nonplanar = np.minimum(angles, np.abs(180.0 - angles)) > 5.0
    raw_cos = np.sum(prediction["torsion_sincos"] * targets.torsion_target_sincos, axis=-1).clip(-1, 1)
    raw_sin = prediction["torsion_sincos"][:, 0] * targets.torsion_target_sincos[:, 1] - prediction["torsion_sincos"][:, 1] * targets.torsion_target_sincos[:, 0]
    raw_error = np.abs(np.rad2deg(np.arctan2(raw_sin, raw_cos)))
    return {
        "package_index": int(sample.package_index),
        "target_pg": sample.target_pg,
        "local_rotor_set_count": len(local.groups),
        "bond_orbit_mae_angstrom": float(np.mean(np.abs(prediction["bond_lengths"] - targets.bond_target_lengths))),
        "angle_orbit_mae_degrees": _angle_mae_degrees(prediction["angle_cosines"], targets.angle_target_cosines),
        "torsion_orbit_circular_mae_degrees": float(np.mean(errors)),
        "nonplanar_torsion_orbit_circular_mae_degrees": float(np.mean(errors[nonplanar])) if np.any(nonplanar) else 0.0,
        "atom_labelled_torsion_mae_diagnostic_degrees": float(np.mean(raw_error)),
    }


def _save(path, model, optimizer, step, protocol_sha, inheritance, isolation):
    import torch
    torch.save({"schema_version": SCHEMA_VERSION, "step": step, "protocol_sha256": protocol_sha,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "inheritance": inheritance, "isolation": isolation}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch

    protocol_path = args.protocol.resolve(); protocol = _load_protocol(protocol_path)
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=False)
    seed = int(protocol["seed"]); random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    tier = {"records": protocol["panel"]["records"], "evaluation": {}, "gate": {}}
    samples = _panel_samples(protocol, tier)
    examples = []
    for sample in samples:
        contract = build_geometry_contract(sample); graph, targets = build_orbit_ic_example(sample, contract)
        local = build_local_rotor_set_contract(sample, contract)
        examples.append((sample, contract, graph, targets, local))

    model = SetValuedQuotientICPredictor(**_model_setting(protocol))
    parent = torch.load(protocol["parent"]["checkpoint"]["path"], map_location="cpu", weights_only=False)
    inheritance = load_m2p1_parent(model, parent["model"]); isolation = freeze_m2p1_parent(model)
    model = model.to(args.training_device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(protocol["training"]["learning_rate"]), weight_decay=0.0)
    steps = int(protocol["training"]["optimizer_steps"])
    training_unit = protocol["training"].get("training_unit", "molecule_balanced_batch")
    schedule = (
        _balanced_schedule(examples, steps=steps, seed=seed + 17)
        if training_unit == "molecule_balanced_batch"
        else None
    )
    cached_groups = []
    if training_unit == "rotor_set_full_batch":
        model.eval()
        with torch.no_grad():
            for _, _, graph, targets, local in examples:
                model(graph, local, device=args.training_device)
                features = model._captured_torsion_input.detach().clone()
                target = torch.as_tensor(
                    targets.torsion_target_sincos,
                    dtype=torch.float32,
                    device=args.training_device,
                )
                for component, slot_maps in zip(
                    local.coupled_group_components,
                    local.coupled_component_slot_maps,
                    strict=True,
                ):
                    groups = tuple(local.groups[index] for index in component)
                    cached_groups.append((features, target, groups, slot_maps))
        if len(cached_groups) != int(protocol["training"]["rotor_set_count"]):
            raise RuntimeError("M2.2 frozen rotor-set count changed")
    elif training_unit != "molecule_balanced_batch":
        raise ValueError("unsupported M2.2 training unit")
    losses = []
    for step in range(1, steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); total = torch.zeros((), device=args.training_device)
        if training_unit == "rotor_set_full_batch":
            for features, target, groups, slot_maps in cached_groups:
                rows = []
                target_rows = []
                local_groups = []
                offset = 0
                for group in groups:
                    current = model.predict_local_rotor_group(
                        features, group, device=args.training_device
                    )
                    rows.append(current); target_rows.append(target[list(group)])
                    local_groups.append(tuple(range(offset, offset + len(group))))
                    offset += len(group)
                prediction = torch.cat(rows); local_target = torch.cat(target_rows)
                from types import SimpleNamespace
                local_contract = SimpleNamespace(
                    groups=tuple(local_groups),
                    coupled_group_components=(tuple(range(len(local_groups))),),
                    coupled_component_slot_maps=(slot_maps,),
                )
                total = total + coupled_permutation_invariant_circular_loss(
                    prediction, local_target, local_contract
                ) / len(cached_groups)
        else:
            batch = schedule[step - 1]
            for index in batch:
                _, _, graph, targets, local = examples[index]
                prediction = model(graph, local, device=args.training_device)
                target = torch.as_tensor(targets.torsion_target_sincos, dtype=torch.float32, device=args.training_device)
                total = total + permutation_invariant_circular_loss(prediction["torsion_sincos"], target, local.groups) / len(batch)
        # A balanced molecule batch can contain no local-rotor set.  The M2.1
        # parent is intentionally frozen, so attach an exact zero to the sole
        # trainable head and make such a batch a deterministic no-op update.
        total = total + sum(
            parameter.sum() * 0.0 for parameter in model.local_rotor_head.parameters()
        )
        total.backward(); gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(protocol["training"]["gradient_clip_norm"]))
        if not torch.isfinite(total) or not torch.isfinite(torch.as_tensor(gradient)): raise RuntimeError("non-finite M2.2 training")
        optimizer.step(); losses.append((step, float(total.detach()), float(gradient)))
        if step == 1 or step % int(protocol["training"]["checkpoint_every"]) == 0 or step == steps:
            model.eval(); records = []
            with torch.no_grad():
                for sample, _, graph, targets, local in examples:
                    pred = {k:v.cpu().numpy() for k,v in model(graph, local, device=args.training_device).items()}
                    records.append(_prediction_record(sample, targets, pred, local))
            values, checks, _ = _prediction_checks(records, protocol["prediction_gate"])
            print(f"step={step}/{steps} loss={float(total):.6g} bond={values['bond_orbit_mae_max_angstrom']:.6g}A angle={values['angle_orbit_mae_max_degrees']:.6g}deg torsion={values['torsion_orbit_circular_mae_max_degrees']:.6g}deg nonplanar={values['nonplanar_torsion_orbit_circular_mae_max_degrees']:.6g}deg gate={'PASS' if all(checks.values()) else 'FAIL'}", flush=True)
            _save(output_dir / f"step-{step:06d}.pt", model, optimizer, step, _sha256(protocol_path), inheritance, isolation)
    _save(output_dir / "last.pt", model, optimizer, steps, _sha256(protocol_path), inheritance, isolation)
    np.savez_compressed(output_dir / "losses.npz", step=np.asarray([x[0] for x in losses]), loss=np.asarray([x[1] for x in losses]), gradient_norm=np.asarray([x[2] for x in losses]))

    model.eval(); predictions=[]
    with torch.no_grad():
        for _,_,graph,_,local in examples:
            predictions.append({k:v.cpu().numpy() for k,v in model(graph, local, device=args.training_device).items()})
    records=[_prediction_record(s,t,p,l) for (s,_,_,t,l),p in zip(examples,predictions,strict=True)]
    values, checks, strata = _prediction_checks(records, protocol["prediction_gate"])
    np.savez_compressed(output_dir / "predictions.npz", **{f"{s.package_index}_{k}":v for (s,_,_,_,_),p in zip(examples,predictions,strict=True) for k,v in p.items()})
    if not all(checks.values()):
        report={"schema_version":SCHEMA_VERSION,"status":"FAIL_M2P2_PREDICTION_GATE_STOP_BEFORE_DECODER","passed":False,"protocol":{"path":str(protocol_path),"sha256":_sha256(protocol_path)},"prediction_records":records,"prediction_gate":{"checks":checks,"values":values,"strata":strata,"thresholds":protocol["prediction_gate"]}}
        _dump(output_dir/"report.json",report); print(f"{report['status']} report={output_dir/'report.json'}",flush=True); return

    model.to("cpu"); torch.cuda.empty_cache()
    decoder_protocol={k:protocol["decoder"][k] for k in ("initialization","optimization","energy_weights")}
    reconstruction=[]; coordinate_parts=[]; offsets=[0]
    starts=int(protocol["decoder"]["initialization"]["starts_per_molecule"]); stride=int(protocol["decoder"]["initialization"]["molecule_specific_seed_stride"])
    for (sample,contract,_,_,_),prediction in zip(examples,predictions,strict=True):
        decoder_targets=_learned_decoder_targets(prediction,contract); specification=build_orbit_parameterization(sample); candidates=[]; coords=[]
        for start in range(starts):
            candidate,coordinates=optimize_one_start(sample,contract,decoder_targets,specification,decoder_protocol,seed=seed+sample.package_index*stride+start,device=protocol["decoder"]["evaluation_device"])
            candidates.append(candidate); coords.append(coordinates)
        selected=min(range(starts),key=lambda i:candidates[i]["final_energy"]); coordinates=coords[selected]
        metrics=reconstruction_metrics(sample,contract,build_oracle_targets(sample,contract),coordinates)
        reconstruction.append({"package_index":sample.package_index,"target_pg":sample.target_pg,"selected_start_index":selected,"selected_energy":candidates[selected]["final_energy"],"finite_start_count":sum(x["all_values_finite"] for x in candidates),"all_starts_finite":all(x["all_values_finite"] for x in candidates),**metrics})
        coordinate_parts.append(coordinates.astype(np.float32)); offsets.append(offsets[-1]+len(coordinates))
        print(f"reconstruct package_index={sample.package_index} PG={sample.target_pg} torsion={metrics['torsion_circular_mae_degrees']:.6g} nonplanar={metrics['nonplanar_torsion_circular_mae_degrees']:.6g} rmsd={metrics['kabsch_rmsd_angstrom']:.6g}A",flush=True)
    rvalues,rchecks,rstrata=_reconstruction_checks(reconstruction,protocol["reconstruction_gate"]); passed=all(rchecks.values())
    status="PASS_M2P2_SET_TORSION_TIER16" if passed else "FAIL_M2P2_RECONSTRUCTION_GATE"
    np.savez_compressed(output_dir/"coordinates.npz",coordinates=np.concatenate(coordinate_parts),atom_offsets=np.asarray(offsets),package_indices=np.asarray([s.package_index for s in samples]))
    report={"schema_version":SCHEMA_VERSION,"status":status,"passed":passed,"protocol":{"path":str(protocol_path),"sha256":_sha256(protocol_path)},"training":{"steps":steps,"parameter_count":sum(p.numel() for p in model.parameters()),"trainable_parameter_count":isolation["trainable_parameter_count"],"inheritance":inheritance,"isolation":isolation},"prediction_records":records,"prediction_gate":{"passed":True,"checks":checks,"values":values,"strata":strata,"thresholds":protocol["prediction_gate"]},"reconstruction_records":reconstruction,"reconstruction_gate":{"passed":passed,"checks":rchecks,"values":rvalues,"strata":rstrata,"thresholds":protocol["reconstruction_gate"]}}
    _dump(output_dir/"report.json",report); print(f"{status} report={output_dir/'report.json'}",flush=True)


if __name__ == "__main__": main()
