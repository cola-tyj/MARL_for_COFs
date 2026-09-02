"""Freeze the M2.2 local-rotor set-head Tier-16 experiment."""

from __future__ import annotations

import argparse, json
from pathlib import Path

from ..c0_local_recovery import _sha256
from ..m2p2_model import SetValuedQuotientICPredictor, freeze_m2p1_parent, load_m2p1_parent
from ..geometry import build_geometry_contract
from ..local_rotor import build_local_rotor_set_contract
from ..overfit import _panel_samples


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--failed-v1-log",type=Path); parser.add_argument("--parent-v2-report",type=Path); parser.add_argument("--parent-v3-report",type=Path); parser.add_argument("--parent-v4-matching-report",type=Path); args=parser.parse_args()
    root=Path("generative_model/PG_OrbitFlow").resolve(); parent_protocol=root/"configs/m2p1_phase_tier16_v1.json"; parent_report=root/"runs/m2p1_phase_tier16_v1/report.json"; parent_checkpoint=root/"runs/m2p1_phase_tier16_v1/last.pt"; diagnosis=root/"reports/m2p1_phase_tier16_diagnosis_v1.json"; representation=root/"reports/m2p2_local_rotor_set_audit_v1.json"
    p=json.loads(parent_protocol.read_text()); r=json.loads(parent_report.read_text()); d=json.loads(diagnosis.read_text()); a=json.loads(representation.read_text())
    if r["status"]!="FAIL_M2P1_PREDICTION_GATE_STOP_BEFORE_DECODER" or not a["passed"]: raise ValueError("M2.2 requires frozen M2.1 failure and passed Gate A")
    settings={k:p["model"][k] for k in ("node_feature_dim","edge_feature_dim","hidden_dim","layers","predict_torsion","geometry_group_phase")}
    import torch
    model=SetValuedQuotientICPredictor(**settings); parent=torch.load(parent_checkpoint,map_location="cpu",weights_only=False); inheritance=load_m2p1_parent(model,parent["model"]); isolation=freeze_m2p1_parent(model)
    settings["parameter_count_expected"]=sum(x.numel() for x in model.parameters()); settings["trainable_parameter_count_expected"]=isolation["trainable_parameter_count"]
    tier={"records":p["panel"]["records"],"evaluation":{},"gate":{}}
    coupled_component_count=sum(len(build_local_rotor_set_contract(sample,build_geometry_contract(sample)).coupled_group_components) for sample in _panel_samples(p,tier))
    sources={name:root/file for name,file in {"m2p2_model":"m2p2_model.py","local_rotor":"local_rotor.py","runner":"m2p2_set_training.py","orbit_ic_model":"orbit_ic_model.py","solver":"m0_oracle_reconstruction.py","kinematics":"orbit_kinematics.py"}.items()}
    protocol={"schema_version":"pg-orbitflow-m2p2-local-rotor-set-protocol-v1","seed":int(p["seed"])+100,"package_dir":p["package_dir"],"canonical_manifest_sha256":p["canonical_manifest_sha256"],"panel":p["panel"],"model":settings,"parent":{"protocol":{"path":str(parent_protocol),"sha256":_sha256(parent_protocol)},"report":{"path":str(parent_report),"sha256":_sha256(parent_report)},"checkpoint":{"path":str(parent_checkpoint),"sha256":_sha256(parent_checkpoint)}},"representation_audit":{"diagnosis":{"path":str(diagnosis),"sha256":_sha256(diagnosis)},"gate_a":{"path":str(representation),"sha256":_sha256(representation)}},"implementation":{n:{"path":str(x),"sha256":_sha256(x)} for n,x in sources.items()},"initialization":{"complete_m2p1_parent_inherited":True,"m2p1_parent_frozen":True,"local_rotor_head_initialized_from_parent":True,"optimizer_reused":False,"inheritance":inheritance,"isolation":isolation},"training":{"optimizer_steps":2048,"training_unit":"rotor_set_full_batch","rotor_set_count":coupled_component_count,"torsion_view_count":int(a["summary"]["local_rotor_set_count"]),"rotor_set_exposures_each":2048,"learning_rate":1e-3,"gradient_clip_norm":5.0,"checkpoint_every":512,"loss":"terminal-sibling-coupled permutation-invariant circular torsion set loss"},"decoder":p["decoder"],"prediction_gate":p["prediction_gate"],"reconstruction_gate":p["reconstruction_gate"],"isolation":{"xyz_or_target_used_to_build_rotor_sets":False,"m2p1_parent_trainable":False,"oracle_decoder_geometry_used":False,"hard_projection_or_f02_used":False,"frozen_parent_features_cached_without_grad":True},"progression":{"if_passes":"freeze M2 Tier-16 completion and run reproducibility audit","if_prediction_fails":"diagnose set-head capacity without changing Gate","if_reconstruction_fails":"diagnose deterministic slot assignment/decoder without oracle target assignment"}}
    if args.failed_v1_log:
        failed=args.failed_v1_log.resolve(); protocol["execution_revision"]={"failed_v1_log":{"path":str(failed),"sha256":_sha256(failed)},"failure_signature":"RuntimeError: element 0 of tensors does not require grad","failure_step":2,"checkpoint_written_before_failure":True,"checkpoint_reused":False,"restart_policy":"new output directory, deterministic restart from step 1","scientific_protocol_changed":False,"sole_code_change":"attach an exact zero-gradient local-rotor-head anchor for batches containing no rotor set"}
    if args.parent_v2_report:
        previous=args.parent_v2_report.resolve(); protocol["training_revision"]={"parent_v2_report":{"path":str(previous),"sha256":_sha256(previous)},"parent_status":"FAIL_M2P2_PREDICTION_GATE_STOP_BEFORE_DECODER","checkpoint_reused":False,"restart_policy":"same M2.1 parent, new head initialization, new optimizer, restart from step 1","single_scientific_change":"replace sparse molecule-balanced batches with a full batch of the 12 rotor sets; cache the frozen parent features","reason":"v2 proved representation improvement but many molecule batches contained no trainable set and Hungarian assignments were unstable"}
    if args.parent_v3_report:
        previous=args.parent_v3_report.resolve(); protocol["training_revision"]={"parent_v3_report":{"path":str(previous),"sha256":_sha256(previous)},"parent_status":"FAIL_M2P2_RECONSTRUCTION_GATE","checkpoint_reused":False,"restart_policy":"same M2.1 parent, new head initialization, new optimizer, restart from step 1","single_scientific_change":"couple all torsion views sharing one terminal sibling atom set under a single Hungarian permutation","reason":"v3 prediction set Gate passed, but independently matched torsion views learned inconsistent slot permutations and produced incompatible decoder constraints"}
    if args.parent_v4_matching_report:
        previous=args.parent_v4_matching_report.resolve(); previous_data=json.loads(previous.read_text(encoding="utf-8"));
        if previous_data["status"]!="FAIL_M2P2_GRAPH_EQUIVALENT_RECONSTRUCTION_GATE": raise ValueError("v5 requires the frozen failed v4 matching audit")
        protocol["training_revision"]={"parent_v4_matching_report":{"path":str(previous),"sha256":_sha256(previous)},"parent_status":"FAIL_M2P2_GRAPH_EQUIVALENT_RECONSTRUCTION_GATE","checkpoint_reused":False,"restart_policy":"same M2.1 parent, new head initialization, new optimizer, restart from step 1","single_scientific_change":"couple terminal sibling groups related by canonical symmetry permutations and conjugate the shared Hungarian assignment into each local slot frame","reason":"v4 passed prediction and 15/16 matched reconstructions; package_index 1075 retained inconsistent gauges across two NH2 groups exchanged by the C2 action"}
        protocol["training"]["loss"]="symmetry-operation-linked terminal-sibling coupled permutation-invariant circular torsion set loss"
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); print(f"WROTE_M2P2_PROTOCOL {args.output.resolve()}")


if __name__=="__main__": main()
