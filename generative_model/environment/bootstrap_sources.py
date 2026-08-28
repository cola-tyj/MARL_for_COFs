"""按 source_lock.json 获取并校验只读上游模型源码。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "generative_model/models/source_lock.json"
DEFAULT_DESTINATION = ROOT / "generative_model/external"


def _run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def bootstrap(lock_path: Path, destination: Path, model: str | None = None) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected = lock["sources"]
    if model is not None:
        selected = [entry for entry in selected if entry["id"] == model]
        if not selected:
            raise ValueError(f"source lock 中没有模型 {model!r}")

    destination.mkdir(parents=True, exist_ok=True)
    for entry in selected:
        target = destination / entry["directory"]
        if not target.exists():
            _run("git", "clone", "--filter=blob:none", entry["repository"], str(target))
        if not (target / ".git").is_dir():
            raise RuntimeError(f"目标存在但不是 Git checkout: {target}")
        remote = _run("git", "remote", "get-url", "origin", cwd=target)
        if remote.rstrip("/").removesuffix(".git") != entry["repository"].rstrip("/").removesuffix(".git"):
            raise RuntimeError(f"{entry['id']} origin 不符合 lock: {remote}")
        _run("git", "fetch", "origin", entry["commit"], cwd=target)
        _run("git", "checkout", "--detach", entry["commit"], cwd=target)
        head = _run("git", "rev-parse", "HEAD", cwd=target)
        if head != entry["commit"]:
            raise RuntimeError(f"{entry['id']} HEAD={head}，预期 {entry['commit']}")
        dirty = _run("git", "status", "--porcelain", cwd=target)
        if dirty:
            raise RuntimeError(f"{entry['id']} checkout 有未提交修改")
        print(f"{entry['id']}: {head} ({target})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--model", help="只获取指定 source id，例如 semlaflow")
    args = parser.parse_args()
    bootstrap(args.lock, args.destination, args.model)


if __name__ == "__main__":
    main()
