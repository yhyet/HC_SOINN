#!/usr/bin/env python3
"""
顺序跑 init_cls=50 + HC-SOINN + STAR 的 7 组实验（3 方法 × 3 数据集 减去 2 组已有数据）。

跳过（你说的已有数据）：
  - SEMA + cifar224
  - DualPrompt + imagenet-r (imagenetr)

用法（仓库根目录）:
  python scripts/run_batch_k50_hc_soinn_star.py
  python scripts/run_batch_k50_hc_soinn_star.py --from-job 3
  python scripts/run_batch_k50_hc_soinn_star.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 顺序：可按需调整
JOBS = [
    ("coda", "cifar224", REPO_ROOT / "exps/batch_k50_init/k50_coda_cifar224_hc_star.json"),
    ("coda", "imagenetr", REPO_ROOT / "exps/batch_k50_init/k50_coda_imagenetr_hc_star.json"),
    ("coda", "cub", REPO_ROOT / "exps/batch_k50_init/k50_coda_cub_hc_star.json"),
    ("sema", "imagenetr", REPO_ROOT / "exps/batch_k50_init/k50_sema_imagenetr_hc_star.json"),
    ("sema", "cub", REPO_ROOT / "exps/batch_k50_init/k50_sema_cub_hc_star.json"),
    ("dual", "cifar224", REPO_ROOT / "exps/batch_k50_init/k50_dual_cifar224_hc_star.json"),
    ("dual", "cub", REPO_ROOT / "exps/batch_k50_init/k50_dual_cub_hc_star.json"),
]

EXCLUDED = [
    "sema + cifar224 (already have)",
    "dualprompt + imagenetr (already have)",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-job", type=int, default=1, help="从第几个 job 开始（1-based）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将运行的命令")
    parser.add_argument("--device", type=str, default=None, help="覆盖 JSON 中的 device，如 0 或 0 1")
    args_ns = parser.parse_args()

    start = max(1, args_ns.from_job)
    print("Skipped (existing results):", EXCLUDED)
    print("-" * 60)

    for i, (method, dataset, cfg) in enumerate(JOBS, start=1):
        if i < start:
            continue
        if not cfg.is_file():
            print(f"[skip] missing config: {cfg}", file=sys.stderr)
            sys.exit(1)
        cmd = [sys.executable, str(REPO_ROOT / "main.py"), "--config", str(cfg)]
        if args_ns.device is not None:
            cmd.extend(["--device", *args_ns.device.split()])
        print(f"\n[{i}/{len(JOBS)}] {method} | {dataset}\n  -> {' '.join(cmd)}")
        if args_ns.dry_run:
            continue
        r = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print(f"Exit code {r.returncode}, stopped.", file=sys.stderr)
            sys.exit(r.returncode)

    print("\nDone. Per-class avg nodes: see [HC-SOINN Node Summary | dataset end] in each log.")


if __name__ == "__main__":
    main()
