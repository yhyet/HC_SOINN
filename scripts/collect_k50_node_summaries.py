#!/usr/bin/env python3
"""
从 logs/<model>/<dataset>/50/10/*.log 中抽取最后一次出现的
[HC-SOINN Node Summary | dataset end] 块，汇总到 stdout（或 -o 文件）。

仓库根目录运行:
  python scripts/collect_k50_node_summaries.py
  python scripts/collect_k50_node_summaries.py -o result_k50_nodes.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO_ROOT / "logs"

PATTERNS = [
    (re.compile(r"\[HC-SOINN Node Summary \| dataset end\] dataset=(\S+)"), "header"),
    (re.compile(r"avg_nodes_per_class \(over all classes\): (\S+)"), "avg_all"),
    (re.compile(r"total_classes: (\S+)"), "classes"),
    (re.compile(r"total_nodes: (\S+)"), "nodes"),
]


def extract_summary(text: str) -> dict[str, str] | None:
    """取文件中最后一个 summary 块的关键行。"""
    lines = text.splitlines()
    last_start = None
    for i, line in enumerate(lines):
        if "[HC-SOINN Node Summary | dataset end]" in line:
            last_start = i
    if last_start is None:
        return None
    chunk = "\n".join(lines[last_start : last_start + 20])
    out: dict[str, str] = {}
    for line in lines[last_start : last_start + 20]:
        for rx, key in PATTERNS:
            m = rx.search(line)
            if m:
                out[key] = m.group(1).strip()
                break
    return out if out else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    rows: list[str] = []
    for model in ("coda_prompt", "sema", "dualprompt"):
        base = LOG_ROOT / model
        if not base.is_dir():
            continue
        for ds_dir in sorted(base.iterdir()):
            if not ds_dir.is_dir():
                continue
            bucket = ds_dir / "50" / "10"
            if not bucket.is_dir():
                continue
            logs = sorted(bucket.glob("*.log"), key=lambda p: p.stat().st_mtime)
            if not logs:
                continue
            latest = logs[-1]
            txt = latest.read_text(encoding="utf-8", errors="ignore")
            summ = extract_summary(txt)
            if not summ:
                rows.append(f"{model}\t{ds_dir.name}\t(no HC-SOINN summary)\t{latest.name}")
                continue
            rows.append(
                f"{model}\t{ds_dir.name}\tavg={summ.get('avg_all','?')}\t"
                f"classes={summ.get('classes','?')}\tnodes={summ.get('nodes','?')}\t{latest.name}"
            )

    out = "\n".join(rows) + ("\n" if rows else "")
    if args.output:
        args.output.write_text(out, encoding="utf-8")
    print(out, end="")


if __name__ == "__main__":
    main()
