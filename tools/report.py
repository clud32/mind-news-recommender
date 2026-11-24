"""L5 报告的两件机械活：

  python tools/report.py macros   由指标包、结论包生成 report/generated/macros.tex
                                  \\iv{I-…} 展开为数字（按指标的 digits，缺省整数原样、小数 3 位），
                                  \\cv{C-…} 展开为结论的 statement_en 原文
  python tools/report.py trace    扫描 report/*.tex 中的 \\iv{} 与 \\cv{}，生成 results/number_trace.csv

报告里写 \\input{generated/macros.tex}，之后正文只用 \\iv{} 和 \\cv{}。
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import ROOT  # noqa: E402

IV, CV = ROOT / "results" / "indicators.json", ROOT / "results" / "conclusions.json"


def items(path):
    return json.loads(path.read_text(encoding="utf-8"))["payload"] if path.exists() else []


def fmt(item):
    v = item["value"]
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and "digits" not in item):
        return str(int(v))
    return f"{v:.{item.get('digits', 3)}f}"


def tex_escape(s):
    return re.sub(r"([%&#_$])", r"\\\1", s)


def macros():
    out = ROOT / "report" / "generated" / "macros.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% 由 tools/report.py macros 生成，不要手改",
             r"\makeatletter",
             r"\newcommand{\iv}[1]{\@ifundefined{iv@#1}{\textbf{??#1}}{\csname iv@#1\endcsname}}",
             r"\newcommand{\cv}[1]{\@ifundefined{cv@#1}{\textbf{??#1}}{\csname cv@#1\endcsname}}",
             r"\makeatother"]
    lines += [rf"\expandafter\def\csname iv@{i['id']}\endcsname{{{fmt(i)}}}" for i in items(IV)]
    lines += [rf"\expandafter\def\csname cv@{c['id']}\endcsname{{{tex_escape(c['statement_en'])}}}" for c in items(CV)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{out.relative_to(ROOT)}：指标 {len(items(IV))} 条，结论 {len(items(CV))} 条")


def trace():
    iv = {i["id"]: fmt(i) for i in items(IV)}
    cv = {c["id"] for c in items(CV)}
    rows, missing = [], []
    for f in sorted((ROOT / "report").glob("*.tex")):
        sec = "Title"
        for line in f.read_text(encoding="utf-8").splitlines():
            code = line.split("%", 1)[0]
            m = re.match(r"\s*\\(?:sub)*section\*?\{(.*?)\}", code)
            if m:
                sec = m.group(1)
            for kind, ref in re.findall(r"\\(iv|cv)\{([^}]*)\}", code):
                if (kind == "iv" and ref not in iv) or (kind == "cv" and ref not in cv):
                    missing.append(ref)
                rows.append([f"{f.name}：{sec}", iv.get(ref, "?") if kind == "iv" else "（结论原文）", ref, "机核" if kind == "iv" else "宏生成"])
    if missing:
        print(f"[停止] 报告引用了包里没有的编号：{sorted(set(missing))}")
        sys.exit(1)
    out = ROOT / "results" / "number_trace.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["论文位置", "数字或结论", "来源编号", "核对状态"])
        w.writerows(rows)
    print(f"{out.relative_to(ROOT)}：{len(rows)} 行")


if __name__ == "__main__":
    {"macros": macros, "trace": trace}[sys.argv[1] if len(sys.argv) > 1 else "x"]()
