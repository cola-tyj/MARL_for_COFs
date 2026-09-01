"""Validate the inherited baseline seed, then run the frozen M3.6 decoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .c0_local_recovery import _sha256
from .m3p6_shape_decoder_gate import main as run_decoder


def main() -> None:
    try:
        protocol_index = sys.argv.index("--protocol") + 1
        protocol_path = Path(sys.argv[protocol_index]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("M3.6 v2 requires --protocol PATH") from error
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline = protocol.get("baseline_decoder_protocol")
    if not baseline:
        raise RuntimeError("M3.6 v2 baseline decoder protocol is missing")
    baseline_path = Path(baseline["path"])
    if _sha256(baseline_path) != baseline["sha256"]:
        raise RuntimeError("baseline decoder protocol identity changed")
    baseline_protocol = json.loads(baseline_path.read_text(encoding="utf-8"))
    if int(protocol["seed"]) != int(baseline_protocol["seed"]):
        raise RuntimeError("M3.6 v2 did not inherit the baseline decoder seed")
    run_decoder()


if __name__ == "__main__":
    main()
