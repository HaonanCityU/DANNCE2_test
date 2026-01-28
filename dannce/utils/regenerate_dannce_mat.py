"""
重新生成 mouse_dannce.mat 文件，基于新的视频文件。

这个脚本会：
1. 从旧文件保留相机参数（params）和相机名称（camnames）
2. 基于新视频生成新的sync数据（data_frame范围匹配新视频）
3. 创建新的labelData（占位符）
4. 可选：从COM文件添加COM数据

用法:
    python regenerate_dannce_mat.py --old-mat path/to/old_mouse_dannce.mat \
                                    --viddir path/to/videos \
                                    --output path/to/new_mouse_dannce.mat \
                                    [--fps 30] \
                                    [--num-landmarks 22] \
                                    [--com-file path/to/com3d0.mat]
"""

import argparse
import os
import sys
import numpy as np
import scipy.io as sio
import imageio
from typing import List, Dict, Any

# 添加项目路径以便导入dannce模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dannce.engine.io import load_camera_params, load_camnames, load_com


def get_video_frame_count(vid_path: str) -> int:
    """获取视频文件的帧数"""
    try:
        reader = imageio.get_reader(vid_path)
        count = reader.count_frames()
        reader.close()
        return count
    except Exception as e:
        print(f"Warning: Could not get frame count for {vid_path}: {e}")
        return 0


def find_videos_in_directory(viddir: str, camname: str, extensions: List[str] = [".mp4", ".avi"]) -> List[str]:
    """在目录中查找视频文件"""
    camdir = os.path.join(viddir, camname)
    if not os.path.exists(camdir):
        return []
    
    videos = []
    for f in os.listdir(camdir):
        if any(f.endswith(ext) for ext in extensions):
            videos.append(os.path.join(camdir, f))
    
    return sorted(videos)


def generate_sync_data(viddir: str, camnames: List[str], fps: float, num_landmarks: int) -> List[Dict]:
    """基于视频文件生成sync数据"""
    sync_data = []
    
    # 首先检查所有相机的视频帧数
    cam_frame_counts = {}
    for camname in camnames:
        videos = find_videos_in_directory(viddir, camname)
        if not videos:
            raise ValueError(f"No videos found for camera {camname} in {viddir}")
        
        total_frames = 0
        for vid_path in videos:
            frame_count = get_video_frame_count(vid_path)
            total_frames += frame_count
            print(f"  {os.path.basename(vid_path)}: {frame_count} frames")
        
        cam_frame_counts[camname] = total_frames
        print(f"Camera {camname}: {total_frames} total frames")
    
    # 检查所有相机的帧数是否一致
    frame_counts = list(cam_frame_counts.values())
    if len(set(frame_counts)) > 1:
        print(f"Warning: Cameras have different frame counts: {cam_frame_counts}")
        # 使用最小帧数
        total_frames = min(frame_counts)
        print(f"Using minimum frame count: {total_frames}")
    else:
        total_frames = frame_counts[0]
    
    # 生成sync数据
    fp = 1000.0 / fps  # frame period in ms
    
    for camname in camnames:
        # Match Label3D.exportDannce format:
        # - data_sampleID: (N, 1)
        # - data_frame: (1, N) row vector
        # - data_2d: (N, 2*num_landmarks)
        # - data_3d: (N, 3*num_landmarks)
        data_frame_col = np.arange(total_frames, dtype="float64")[:, np.newaxis]  # (N,1)
        data_frame_row = data_frame_col.T  # (1,N)
        data_sampleID = (data_frame_col * fp + 1).astype("float64")  # (N,1)
        data_2d = np.zeros((total_frames, 2 * num_landmarks), dtype="float64")
        data_3d = np.zeros((total_frames, 3 * num_landmarks), dtype="float64")
        
        sync_data.append(
            {
                "data_frame": data_frame_row,
                "data_sampleID": data_sampleID,
                "data_2d": data_2d,
                "data_3d": data_3d,
            }
        )
    
    return sync_data


def create_label_data(num_frames: int, num_landmarks: int, num_cams: int, fps: float = 30.0) -> np.ndarray:
    """
    创建占位符 labelData，严格匹配 Label3D.exportDannce 输出：
    - labelData: cell(nCams,1)
    - each cell: struct with fields data_2d, data_3d, data_frame, data_sampleID
      where:
        data_2d: (N, 2*num_landmarks)
        data_3d: (N, 3*num_landmarks)
        data_frame: (1, N)
        data_sampleID: (N, 1)
    """
    label_cells = np.empty((num_cams, 1), dtype=object)
    fp = 1000.0 / fps  # frame period in ms
    
    data_frame_col = np.arange(num_frames, dtype="float64")[:, np.newaxis]  # (N,1)
    data_frame_row = data_frame_col.T  # (1,N)
    data_sampleID = (data_frame_col * fp + 1).astype("float64")  # (N,1)
    
    for cam_idx in range(num_cams):
        data_2d = np.zeros((num_frames, 2 * num_landmarks), dtype="float64")
        data_3d = np.zeros((num_frames, 3 * num_landmarks), dtype="float64")
        
        # IMPORTANT: fields are numeric arrays directly (no extra (1,1) object wrapping),
        # matching MATLAB struct fields produced by Label3D.exportDannce.
        label_struct = np.array(
            [(data_2d, data_3d, data_frame_row, data_sampleID)],
            dtype=[
                ("data_2d", "O"),
                ("data_3d", "O"),
                ("data_frame", "O"),
                ("data_sampleID", "O"),
            ],
        ).reshape(1, 1)
        label_cells[cam_idx, 0] = label_struct
    
    return label_cells


def regenerate_dannce_mat(
    old_mat_path: str,
    viddir: str,
    output_path: str,
    fps: float = 30.0,
    num_landmarks: int = 22,
    com_file: str = None,
):
    """重新生成mouse_dannce.mat文件"""
    
    print(f"Loading old mat file: {old_mat_path}")
    old_data = sio.loadmat(old_mat_path)
    
    # 保留相机参数和名称
    if "params" not in old_data:
        raise ValueError("Old mat file must contain 'params'")
    if "camnames" not in old_data:
        raise ValueError("Old mat file must contain 'camnames'")
    
    params = old_data["params"]
    camnames = load_camnames(old_mat_path)
    
    if camnames is None:
        raise ValueError("Could not load camnames from old mat file")
    
    print(f"Found {len(camnames)} cameras: {camnames}")
    
    # 生成新的sync数据
    print(f"\nGenerating sync data from videos in: {viddir}")
    sync_data = generate_sync_data(viddir, camnames, fps, num_landmarks)
    
    # 获取总帧数
    num_frames = len(sync_data[0]["data_frame"])
    print(f"\nTotal frames: {num_frames}")
    
    # 创建新的labelData（占位符）
    print("Creating labelData placeholder...")
    label_data = create_label_data(num_frames, num_landmarks, len(camnames), fps)
    
    # 构建 sync 为 MATLAB cell(nCams,1)，每个 cell 里是 1x1 struct，
    # 且每个字段直接是 numeric array（完全对齐 Label3D.exportDannce）。
    sync_cells = np.empty((len(sync_data), 1), dtype=object)
    for cam_idx, sync_dict in enumerate(sync_data):
        sync_struct = np.array(
            [
                (
                    sync_dict["data_frame"],     # (1,N)
                    sync_dict["data_sampleID"],  # (N,1)
                    sync_dict["data_2d"],        # (N,2L)
                    sync_dict["data_3d"],        # (N,3L)
                )
            ],
            dtype=[
                ("data_frame", "O"),
                ("data_sampleID", "O"),
                ("data_2d", "O"),
                ("data_3d", "O"),
            ],
        ).reshape(1, 1)
        sync_cells[cam_idx, 0] = sync_struct
    
    # 构建新的mat文件数据
    new_data = {
        "camnames": old_data["camnames"],
        "params": params,
        "labelData": label_data,
        "sync": sync_cells,
    }
    
    # 如果有COM文件，添加COM数据
    if com_file and os.path.exists(com_file):
        print(f"\nAdding COM data from: {com_file}")
        try:
            com_data = load_com(com_file)
            new_data["com"] = {
                "com3d": com_data["com3d"],
                "sampleID": com_data["sampleID"],
            }
        except Exception as e:
            print(f"Warning: Could not load COM data: {e}")
    
    # 保存新文件
    print(f"\nSaving new mat file to: {output_path}")
    sio.savemat(output_path, new_data, do_compression=True)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="重新生成mouse_dannce.mat文件，基于新的视频文件"
    )
    parser.add_argument(
        "--old-mat",
        required=True,
        help="旧的mouse_dannce.mat文件路径"
    )
    parser.add_argument(
        "--viddir",
        required=True,
        help="视频文件目录（包含Camera1, Camera2等子目录）"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="视频帧率（默认30.0）"
    )
    parser.add_argument(
        "--num-landmarks",
        type=int,
        default=22,
        help="关键点数量（默认22）"
    )
    parser.add_argument(
        "--com-file",
        default=None,
        help="可选的COM文件路径（com3d0.mat）"
    )
    
    args = parser.parse_args()
    
    regenerate_dannce_mat(
        args.old_mat,
        args.viddir,
        args.output,
        args.fps,
        args.num_landmarks,
        args.com_file,
    )


if __name__ == "__main__":
    main()
