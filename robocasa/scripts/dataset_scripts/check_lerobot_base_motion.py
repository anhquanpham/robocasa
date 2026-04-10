#!/usr/bin/env python3
"""
List LeRobot episodes whose logged `action` has non-zero base command (per meta/modality.json).

Uses `meta/modality.json` → `action.base_motion` for slice indices (defaults to [0:4] if absent).
Considers a timestep "base active" if any base component has magnitude > `--eps` (float noise).

Example:

    python check_lerobot_base_motion.py \\
        /path/to/.../lerobot \\
        --eps 1e-5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _load_base_slice(lerobot_root: Path) -> tuple[int, int]:
    modality_path = lerobot_root / "meta" / "modality.json"
    if not modality_path.is_file():
        print(f"Warning: {modality_path} not found; using base_motion indices [0, 4).", file=sys.stderr)
        return 0, 4
    with open(modality_path, encoding="utf-8") as f:
        modality = json.load(f)
    action_meta = modality.get("action") or {}
    base = action_meta.get("base_motion") or {}
    start = int(base.get("start", 0))
    end = int(base.get("end", 4))
    if end <= start:
        raise ValueError(f"Invalid base_motion slice in modality.json: start={start} end={end}")
    return start, end


def _episode_index_from_name(path: Path) -> int | None:
    m = re.match(r"episode_(\d+)\.parquet$", path.name)
    return int(m.group(1)) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find LeRobot episodes with non-negligible base motion in `action`."
    )
    parser.add_argument(
        "lerobot_root",
        type=Path,
        help="Path to a LeRobot dataset root (contains meta/ and data/).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Threshold: flag a frame if max(|action[base]|) > eps (default: 1e-5).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use exact non-zero (any base component != 0.0) instead of --eps.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Per flagged episode: print max |base| over all frames.",
    )
    args = parser.parse_args()

    root: Path = args.lerobot_root.expanduser().resolve()
    data_root = root / "data"
    if not data_root.is_dir():
        print(f"Error: {data_root} is not a directory.", file=sys.stderr)
        return 1

    try:
        b0, b1 = _load_base_slice(root)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading modality: {e}", file=sys.stderr)
        return 1

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(
            "Requires numpy and pandas (e.g. `pip install numpy pandas pyarrow`).",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    parquet_files = sorted(data_root.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        print(f"No episode parquet files under {data_root}", file=sys.stderr)
        return 1

    flagged: list[tuple[str, int | None, float]] = []

    for pq in parquet_files:
        df = pd.read_parquet(pq)
        if "action" not in df.columns:
            print(f"Warning: no 'action' column in {pq}, skipping.", file=sys.stderr)
            continue
        actions = np.stack(df["action"].values)
        base = actions[:, b0:b1]
        if args.strict:
            active = np.any(base != 0.0, axis=1)
            max_abs = float(np.max(np.abs(base))) if base.size else 0.0
        else:
            max_abs = float(np.max(np.abs(base))) if base.size else 0.0
            active = np.max(np.abs(base), axis=1) > args.eps

        if not np.any(active):
            continue

        ep_name = pq.stem
        ep_idx = _episode_index_from_name(pq)
        flagged.append((ep_name, ep_idx, max_abs))

    total = len(parquet_files)
    n_flag = len(flagged)

    print(f"lerobot_root: {root}")
    print(f"base_motion slice: [{b0}:{b1}] (from meta/modality.json)")
    print(f"criterion: {'any base != 0.0' if args.strict else f'max(|base|) > {args.eps}'}")
    print(f"scanned episodes: {total}")
    print(f"episodes with base motion: {n_flag}")
    print()

    flagged.sort(key=lambda x: (x[1] is not None, x[1] if x[1] is not None else -1))
    for ep_name, ep_idx, max_abs in flagged:
        if args.verbose:
            idx_str = f"index={ep_idx}" if ep_idx is not None else "index=?"
            print(f"{ep_name}  ({idx_str}, max|base|={max_abs:.6g})")
        else:
            print(ep_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
