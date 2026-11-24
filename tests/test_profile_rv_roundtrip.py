"""Run from the repository root with the project interpreter."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(1, str(Path.cwd()))
from profile_mind_small import profile
from tools.common import ROOT, payload_sha256
from tools.rv import write_result


def test_histogram_roundtrip():
    news = [f"N{i}\tc\ts\ttitle\tabstract\turl\t[]\t[]\n".encode()
            for i in range(1, 101)]
    exposures = []
    for row, count in enumerate([2, 10, 10, 100, 100, 100], start=1):
        candidates = " ".join(f"N{i}-{int(i == 1)}" for i in range(1, count + 1))
        exposures.append(f"{row}\tU1\t11/12/2019 {row}:00:00 AM\t\t{candidates}\n".encode())
    payload = profile(news, exposures, (4, 5))
    expected = {"2": 1, "10": 2, "100": 3}

    scratch = ROOT / "scratch" / "card-901"
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=scratch) as temp_dir:
        path = Path(temp_dir) / "synthetic.json"
        written = write_result(path, payload, kind="RV", version="synthetic-histogram")
        saved = json.loads(path.read_text(encoding="utf-8"))

    recorded = saved["_meta"]["payload_sha256"]
    recomputed = payload_sha256(saved["payload"])
    print(f"metadata_sha256={recorded}")
    print(f"readback_sha256={recomputed}")
    assert written == recorded == recomputed, "payload hash changed after JSON round-trip"
    assert payload["impressions"]["candidate_count_histogram"] == expected
    assert saved["payload"]["impressions"]["candidate_count_histogram"] == expected
    print(f"PASS histogram={json.dumps(expected)}")


if __name__ == "__main__":
    test_histogram_roundtrip()
