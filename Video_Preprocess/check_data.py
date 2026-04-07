#!/usr/bin/env python3
"""
Check DANNCE train/val sample splits (.pickle) and whether they include multiple experiments.

Typical usage:

  python check_data.py \
    --train-pickle /path/to/train_samples.pickle \
    --val-pickle   /path/to/val_samples.pickle

python /home/haonan/proj/dannce-release_dev2/Video_Preprocess/check_data.py \
  --train-pickle /home/haonan/proj/dannce-release_dev2/demo/multi_sum/DANNCE/train_results/AVG/train_samples.pickle \
  --val-pickle   /home/haonan/proj/dannce-release_dev2/demo/multi_sum/DANNCE/train_results/AVG/val_samples.pickle \
  --io-yaml       /home/haonan/proj/dannce-release_dev2/demo/multi_sum/io_train.yaml

If you also provide the io yaml used for training (with `exp:` list), the script
will print the mapping exp_id -> label3d_file.
"""

from __future__ import annotations

import argparse
import collections
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple


def _to_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode(errors="ignore")
    return str(x)


def _exp_id(sample_id: str) -> str:
    # Typical format: "0_12345" where 0 is experiment index
    if "_" not in sample_id:
        return "NO_PREFIX"
    return sample_id.split("_", 1)[0]


def _load_pickle(path: str) -> List[str]:
    with open(path, "rb") as f:
        xs = pickle.load(f)
    # xs can be list / numpy array / etc.
    try:
        xs = list(xs)
    except Exception:
        xs = [xs]
    return [_to_str(x) for x in xs]


def _count_by_exp(sample_ids: Iterable[str]) -> Dict[str, int]:
    c = collections.Counter(_exp_id(s) for s in sample_ids)
    return dict(sorted(c.items(), key=lambda kv: kv[0]))


def _examples_by_exp(sample_ids: Iterable[str], k: int = 5) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = collections.defaultdict(list)
    for s in sample_ids:
        eid = _exp_id(s)
        if len(out[eid]) < k:
            out[eid].append(s)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _load_exp_mapping_from_io(io_yaml_path: str) -> List[Tuple[int, str]]:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("PyYAML not available; cannot read io.yaml") from e

    cfg = yaml.safe_load(open(io_yaml_path, "r"))
    exps = cfg.get("exp", []) or []
    out = []
    for i, d in enumerate(exps):
        if not isinstance(d, dict):
            continue
        out.append((i, d.get("label3d_file", "")))
    return out


def _print_block(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pickle", required=True, help="Path to train_samples.pickle")
    ap.add_argument("--val-pickle", default=None, help="Path to val_samples.pickle (optional)")
    ap.add_argument("--io-yaml", default=None, help="Path to io_train.yaml (optional)")
    ap.add_argument("--examples", type=int, default=5, help="How many sampleID examples per exp")
    args = ap.parse_args()

    train_path = os.path.abspath(args.train_pickle)
    val_path = os.path.abspath(args.val_pickle) if args.val_pickle else None

    train_ids = _load_pickle(train_path)
    _print_block("TRAIN")
    print("file:", train_path)
    print("num_samples:", len(train_ids))
    print("exp_id_counts:", _count_by_exp(train_ids))
    ex = _examples_by_exp(train_ids, k=args.examples)
    for eid, xs in ex.items():
        print(f"{eid} examples:", xs)

    if val_path:
        val_ids = _load_pickle(val_path)
        _print_block("VAL")
        print("file:", val_path)
        print("num_samples:", len(val_ids))
        print("exp_id_counts:", _count_by_exp(val_ids))
        exv = _examples_by_exp(val_ids, k=args.examples)
        for eid, xs in exv.items():
            print(f"{eid} examples:", xs)

    if args.io_yaml:
        _print_block("IO.YAML exp mapping")
        mapping = _load_exp_mapping_from_io(args.io_yaml)
        print("file:", os.path.abspath(args.io_yaml))
        for i, mat in mapping:
            print(f"exp[{i}] label3d_file:", mat)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

