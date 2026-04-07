#!/usr/bin/env python3
"""
从预测用 dannce .mat 中按「全局帧号」删除若干区间（例如某段 chunk 多机不同步）。

适用于 regenerate_dannce_mat / makeSync 一类：每路 sync 的每一行对应同一时间索引，
且 Camera1 的 data_frame 为 0..N-1（全局帧）。

用法示例：
  # 按全局闭区间排除（可多次）：
  python filter_dannce_mat_exclude_frame_ranges.py -i mouse_fullsync_dannce.mat \\
    -o mouse_fullsync_nobadchunk_dannce.mat --exclude 191851-291105

  # 按 chunk 文件名自动算区间（只查 Camera1 下 chunk_start.mp4 的帧数）：
  python filter_dannce_mat_exclude_frame_ranges.py -i in.mat -o out.mat \\
    --viddir ./videos --exclude-chunk 191851

训练用 mat（稀疏 labelData）一般不需要跑本脚本；仅当你希望预测 mat 与标注一致地跳过坏段时使用。
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import scipy.io as sio

try:
    import imageio
except ImportError:
    imageio = None


def _unwrap_cell(cell):
    while isinstance(cell, np.ndarray) and cell.dtype == object and cell.size == 1:
        cell = cell.flat[0]
    return cell


def _ravel_frames(df):
    x = np.asarray(df, dtype=np.float64).ravel()
    return x


def _parse_ranges(specs):
    """['191851-291105', '10-20'] -> [(191851,291105), (10,20)] inclusive."""
    out = []
    for s in specs:
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
        if not m:
            raise ValueError(f"无效区间: {s!r}，应为 START-END（闭区间）")
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        out.append((a, b))
    return out


def _chunk_range_from_videos(viddir: str, chunk_start: int, cam: str = "Camera1") -> tuple[int, int]:
    if imageio is None:
        raise RuntimeError("需要 imageio 以统计 chunk 帧数")
    mp4 = os.path.join(viddir, cam, f"{chunk_start}.mp4")
    if not os.path.isfile(mp4):
        # 允许多一层子目录（与 processing.load_expdict 类似）
        camdir = os.path.join(viddir, cam)
        if os.path.isdir(camdir):
            subs = [d for d in os.listdir(camdir) if os.path.isdir(os.path.join(camdir, d))]
            if len(subs) == 1:
                mp4 = os.path.join(camdir, subs[0], f"{chunk_start}.mp4")
    if not os.path.isfile(mp4):
        raise FileNotFoundError(f"找不到 chunk 视频: {chunk_start}.mp4（已查 Camera1）")
    r = imageio.get_reader(mp4)
    try:
        n = r.count_frames()
    finally:
        r.close()
    lo = chunk_start
    hi = chunk_start + n - 1
    return lo, hi


def _apply_keep_to_cam_struct(st, keep: np.ndarray) -> None:
    """sync/labelData 单相机 struct：时间维长度与 len(keep) 一致时按 keep 取子集。"""
    for fname in st.dtype.names:
        field = np.asarray(st[fname][0, 0])
        if fname == "data_frame":
            if field.ndim == 2 and field.shape[0] == 1:
                st[fname][0, 0] = field[:, keep]
            elif field.ndim == 2 and field.shape[1] == 1:
                st[fname][0, 0] = field[keep]
            else:
                fr = _ravel_frames(field)[keep]
                st[fname][0, 0] = np.asarray(fr).reshape(field.shape)
        elif fname == "data_sampleID":
            col = field.reshape(-1, 1)
            st[fname][0, 0] = col[keep].reshape(-1, 1)
        elif field.ndim == 2 and field.shape[0] == len(keep):
            st[fname][0, 0] = field[keep]
        elif field.ndim == 2 and field.shape[1] == len(keep):
            st[fname][0, 0] = field[:, keep]


def _subset_sync_cells(mat, keep: np.ndarray) -> None:
    if "sync" not in mat:
        raise KeyError("mat 中无 sync")
    arr = mat["sync"]
    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            st = _unwrap_cell(arr[r, c])
            if not hasattr(st, "dtype") or st.dtype.names is None:
                continue
            _apply_keep_to_cam_struct(st, keep)


def _keep_from_ranges_on_frames(g: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    keep = np.ones(g.size, dtype=bool)
    for lo, hi in ranges:
        keep &= ~((g >= lo) & (g <= hi))
    return keep


def _subset_labeldata_sparse_by_value(mat, ranges: list[tuple[int, int]]) -> None:
    """稀疏 labelData：按 data_frame 全局值删掉落在区间内的行。"""
    if "labelData" not in mat:
        return
    arr = mat["labelData"]
    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            st = _unwrap_cell(arr[r, c])
            if not hasattr(st, "dtype") or st.dtype.names is None:
                continue
            df = np.asarray(st["data_frame"][0, 0], dtype=np.float64).ravel()
            g = df.astype(np.int64)
            keep = _keep_from_ranges_on_frames(g, ranges)
            if keep.all():
                continue
            _apply_keep_to_cam_struct(st, keep)


def _subset_gt3d(mat, keep: np.ndarray) -> None:
    for k in ("gt3d_global", "gt3d", "gt3d_local"):
        if k not in mat:
            continue
        a = np.asarray(mat[k]).ravel()
        if a.size == len(keep):
            mat[k] = a[keep].reshape(-1, 1)


def _maybe_subset_labeldata(mat, keep: np.ndarray, ranges: list[tuple[int, int]]) -> None:
    if "labelData" not in mat:
        return
    ld0 = _unwrap_cell(mat["labelData"][0, 0])
    df_ld = _ravel_frames(ld0["data_frame"][0, 0])
    if df_ld.size == len(keep):
        arr = mat["labelData"]
        rows, cols = arr.shape
        for r in range(rows):
            for c in range(cols):
                st = _unwrap_cell(arr[r, c])
                if hasattr(st, "dtype") and st.dtype.names:
                    _apply_keep_to_cam_struct(st, keep)
    else:
        _subset_labeldata_sparse_by_value(mat, ranges)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="闭区间 START-END，可重复，例如 --exclude 191851-291105",
    )
    ap.add_argument(
        "--viddir",
        default=None,
        help="与 --exclude-chunk 联用：在 Camera1 下找 {chunk}.mp4 并数帧",
    )
    ap.add_argument(
        "--exclude-chunk",
        action="append",
        type=int,
        default=[],
        help="chunk 文件名前缀整数，如 191851，全局帧区间为 [191851, 191851+帧数-1]",
    )
    ap.add_argument(
        "--sync-only",
        action="store_true",
        help="只裁剪 sync（与 gt3d* 若长度一致），不修改 labelData",
    )
    args = ap.parse_args()

    ranges = _parse_ranges(args.exclude)
    if args.viddir and args.exclude_chunk:
        for ch in args.exclude_chunk:
            lo, hi = _chunk_range_from_videos(os.path.abspath(args.viddir), ch)
            ranges.append((lo, hi))
            print(f"chunk {ch}.mp4 -> 全局帧 [{lo}, {hi}]（含端点）", file=sys.stderr)
    if not ranges:
        print("未指定任何排除区间", file=sys.stderr)
        return 1

    mat = sio.loadmat(args.input, struct_as_record=False)
    # 以 Camera1 sync 的 data_frame 长度为准
    sync = mat["sync"]
    cam0 = _unwrap_cell(sync[0, 0])
    df0 = cam0["data_frame"][0, 0]
    g = _ravel_frames(df0).astype(np.int64)
    n = g.size

    # 按全局帧号（data_frame 每列/行的值）排除；与行号是否等于 0..N-1 无关
    keep = _keep_from_ranges_on_frames(g, ranges)

    n_kept = int(np.sum(keep))
    print(f"sync 总行数 {n} -> 保留 {n_kept}，排除 {n - n_kept}", file=sys.stderr)

    _subset_sync_cells(mat, keep)
    _subset_gt3d(mat, keep)
    if not args.sync_only:
        _maybe_subset_labeldata(mat, keep, ranges)

    # 去掉 matlab 元键
    out = {k: mat[k] for k in mat.keys() if not k.startswith("__")}
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sio.savemat(args.output, out, do_compression=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
