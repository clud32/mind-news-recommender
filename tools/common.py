"""权威体系 v2 的公共函数：仓库根、哈希、登记簿解析、git。只用标准库。"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "authority" / "REGISTRY.md"
AUTHORITY = ROOT / "authority" / "AUTHORITY.md"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _round(obj, digits=10):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, digits) for v in obj]
    return obj


def payload_sha256(payload):
    """结果身份值：payload 按键排序、浮点取 10 位后的 JSON 的 sha256（与运行时间、机器无关）。"""
    text = json.dumps(_round(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(*args, check=False):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败：{r.stderr.strip()}")
    return r


def head():
    return git("rev-parse", "HEAD").stdout.strip()


def status_paths():
    """工作区改动（含未跟踪文件），返回 posix 路径集合。"""
    out = git("-c", "core.quotepath=false", "status", "--porcelain", "--untracked-files=all").stdout
    paths = set()
    for line in out.splitlines():
        p = line[3:]
        if " -> " in p:
            p = p.split(" -> ", 1)[1]
        paths.add(p.strip().strip('"'))
    return paths


def table_rows(text, heading_prefix):
    """取以 heading_prefix 开头的 ## 小节里第一张表的数据行（列表的列表）。"""
    m = re.search(rf"^## {re.escape(heading_prefix)}.*?$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return []
    rows = []
    for line in m.group(1).splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: ") or cells[0] in ("对象", "卡", "规则", "编号", "决策", "版本"):
            continue
        rows.append(cells)
    return rows


def snapshot():
    """登记簿第 1 节：{对象: {"version", "identity", "status"}}。"""
    rows = table_rows(REGISTRY.read_text(encoding="utf-8"), "1.")
    return {r[0]: {"version": r[1], "identity": r[2], "status": r[3] if len(r) > 3 else ""} for r in rows if len(r) >= 3}


def registered(row):
    return row is not None and row["version"] not in ("-", "")
