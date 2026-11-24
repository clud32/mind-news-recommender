"""Verify the published G3 offline snapshot using only the Python standard library."""
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.judge.mind_final import conclusions, verify
from src.present.mind_final import plan, render_tex
from tools.common import payload_sha256, sha256_file
from tools import report


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads((ROOT / path).read_bytes())


def verify_documents(manifest):
    documents = {}
    for obj, entry in manifest["results"].items():
        path = entry["path"]
        require(path in manifest["files"], f"unlisted result file: {path}")
        doc = read(path)
        meta = doc["_meta"]
        kind = obj if obj in ("IV", "CV") else "RV"
        require(meta["version"] == entry["version"] and meta["kind"] == kind
                and meta["dirty"] is False, f"unexpected result metadata: {path}")
        require(payload_sha256(doc["payload"]) == meta["payload_sha256"] == entry["payload_sha256"],
                f"payload digest mismatch: {path}")
        documents[obj] = doc

    audits = {obj: doc for obj, doc in documents.items() if obj.startswith("RV-evaluation-audit-s")}
    require(len(audits) == 3 and all("max_individual_deviations" not in doc["payload"]["random"]
                                    for doc in audits.values()), "unsanitized diagnostic payload")
    frozen_references = read("configs/mind-final-v0.1.json")["references"]
    for obj, digest in frozen_references.items():
        require(documents[obj]["_meta"]["payload_sha256"] == digest, f"stale release config input: {obj}")
    for seed in range(3):
        final = documents[f"RV-small-dev-s{seed}"]["payload"]["references"]
        require(final == frozen_references, f"stale final evaluation references: seed {seed}")
    return documents


def verify_report(manifest, documents):
    metrics, sections = plan()
    actual_iv, actual_cv = documents["IV"], documents["CV"]
    expected_iv = [metric.calculate(documents) for metric in metrics]
    require(expected_iv == actual_iv["payload"], "IV values or provenance differ from RV recomputation")
    require(actual_cv["_meta"]["indicator_payload_sha256"] == actual_iv["_meta"]["payload_sha256"],
            "CV points at a stale IV")
    expected_cv = conclusions(expected_iv)
    require(expected_cv == actual_cv["payload"], "CV differs from conclusion recomputation")
    verify(expected_iv, expected_cv)

    source = render_tex(sections).encode("utf8")
    require(source == (ROOT / "report/g3-results.tex").read_bytes(), "frozen report source differs")
    citations = re.findall(rb"\\(?:iv|cv)\{[^}]*\}", source)
    counts = (len(expected_iv), len(expected_cv), len(citations))
    require(counts == tuple(manifest["counts"][name] for name in ("indicators", "conclusions", "citations")),
            "unexpected report counts")

    with TemporaryDirectory() as temp:
        staging = Path(temp)
        (staging / "report").mkdir()
        (staging / "results").mkdir()
        (staging / "report/g3-results.tex").write_bytes(source)
        saved = report.ROOT, report.IV, report.CV
        try:
            report.ROOT, report.IV, report.CV = staging, ROOT / "results/indicators.json", ROOT / "results/conclusions.json"
            report.macros()
            report.trace()
        finally:
            report.ROOT, report.IV, report.CV = saved
        for rel in ("report/generated/macros.tex", "results/number_trace.csv"):
            require((staging / rel).read_bytes() == (ROOT / rel).read_bytes(), f"regenerated output differs: {rel}")
    return counts


def main():
    manifest = read("release/MANIFEST.json")
    require(manifest["format"] == "mind-g3-release-v1", "unsupported release manifest")
    for rel, digest in manifest["files"].items():
        path = ROOT / rel
        require(path.is_file() and sha256_file(path) == digest, f"file digest mismatch: {rel}")
    documents = verify_documents(manifest)
    counts = verify_report(manifest, documents)
    print(f"OK: {len(manifest['files'])} file SHA-256 values and {len(documents)} published payloads")
    print(f"OK: {counts[0]} recomputed indicators, {counts[1]} conclusions, {counts[2]} report citations")


if __name__ == "__main__":
    main()
