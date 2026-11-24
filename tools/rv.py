"""L2/L3/L4 写结果文件的唯一入口：{"_meta": …, "payload": …}，并算出 payload_sha256。

用法（在你的训练、评测脚本末尾）：
    from tools.rv import write_result
    write_result("results/rv/RV-main-001.json", payload={"mAP50": 0.812, ...},
                 kind="RV", version="RV-main-001", config="configs/main.yaml", seed=0)
"""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import ROOT, git, head, payload_sha256, sha256_file, snapshot  # noqa: E402


def write_result(path, payload, kind, version, config=None, seed=None, extra_meta=None):
    snap = snapshot()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip())
    meta = {
        "kind": kind,
        "version": version,
        "commit": head(),
        "dirty": dirty,
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_version": snap.get("DATA", {}).get("version", "-"),
        "lock": snap.get("LOCK", {}).get("version", "-"),
        "config": config,
        "config_sha256": sha256_file(ROOT / config) if config else None,
        "seed": seed,
        "payload_sha256": payload_sha256(payload),
    }
    if extra_meta:
        meta.update(extra_meta)
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"_meta": meta, "payload": payload}, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"{version} payload_sha256={meta['payload_sha256']} dirty={dirty} -> {path}")
    return meta["payload_sha256"]
