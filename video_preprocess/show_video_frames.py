#!/usr/bin/env python3
"""
快速脚本：在rawdata目录中递归查找指定名称的视频，显示所有匹配视频的第Numb_frame帧
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_videos_by_name(root_dir, video_name):
    """
    在root_dir中递归查找所有包含video_name的视频文件
    
    Args:
        root_dir: 根目录路径（rawdata）
        video_name: 要查找的视频名称（可以是部分匹配）
    
    Returns:
        找到的视频文件路径列表
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_paths = []
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误：目录 {root_dir} 不存在")
        return video_paths
    
    # 递归查找所有视频文件
    for ext in video_extensions:
        for video_file in root_path.rglob(f"*{video_name}*{ext}"):
            if video_file.is_file():
                video_paths.append(str(video_file))
    
    return sorted(video_paths)


def extract_frame(video_path, frame_number):
    """
    从视频中提取指定帧
    
    Args:
        video_path: 视频文件路径
        frame_number: 帧编号（从0开始）
    
    Returns:
        提取的帧（numpy数组），如果失败返回None
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"警告：无法打开视频 {video_path}")
            return None
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_number >= total_frames:
            print(f"警告：视频 {video_path} 只有 {total_frames} 帧，无法提取第 {frame_number} 帧")
            cap.release()
            return None
        
        # 设置到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # 将BGR转换为RGB（matplotlib使用RGB）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            print(f"警告：无法读取视频 {video_path} 的第 {frame_number} 帧")
            return None
            
    except Exception as e:
        print(f"错误：处理视频 {video_path} 时出错: {e}")
        return None


def display_frames_grid(video_paths, frames, root_dir, frame_number):
    """
    在一张图上显示所有帧，使用三列网格布局，保持原视频清晰度
    
    Args:
        video_paths: 视频文件路径列表
        frames: 对应的帧数组列表
        root_dir: 根目录（用于计算相对路径）
        frame_number: 帧编号（用于标题显示）
    """
    n_videos = len(frames)
    if n_videos == 0:
        print("没有找到任何有效的帧")
        return
    
    # 固定为三列布局
    cols = 3
    rows = int(np.ceil(n_videos / cols))
    
    # 获取第一个有效帧的尺寸作为参考（假设所有视频分辨率相同或相近）
    first_frame = None
    for frame in frames:
        if frame is not None:
            first_frame = frame
            break
    
    if first_frame is None:
        print("错误：没有有效的帧")
        return
    
    # 获取原始帧的尺寸（高度和宽度，单位：像素）
    frame_height, frame_width = first_frame.shape[:2]
    
    # 设置高DPI以保持清晰度（更高的DPI = 更清晰的显示）
    dpi = 150
    
    # 根据原始帧尺寸和列数计算合适的figure大小
    # 每个子图的宽度 = frame_width / dpi，高度 = frame_height / dpi
    # 总宽度 = 3列 * 单个宽度 + 边距，总高度 = rows * 单个高度 + 边距
    single_width = frame_width / dpi
    single_height = frame_height / dpi
    
    # 添加标题和间距的空间
    fig_width = cols * single_width * 1.1  # 1.1倍用于间距
    fig_height = rows * single_height * 1.15  # 1.15倍用于标题和间距
    
    # 创建图形，设置高DPI以保持清晰度
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f'all matched videos frame {frame_number}', fontsize=16, fontweight='bold')
    
    # 确保axes是二维数组
    if n_videos == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 计算相对路径
    root_path = Path(root_dir).resolve()
    
    # 显示每个帧
    for idx, (video_path, frame) in enumerate(zip(video_paths, frames)):
        if frame is None:
            continue
        
        row = idx // cols
        col = idx % cols
        
        # 获取对应的子图
        if rows == 1 and cols == 1:
            ax = axes[0, 0]
        elif rows == 1:
            ax = axes[0, col]
        elif cols == 1:
            ax = axes[row, 0]
        else:
            ax = axes[row, col]
        
        # 显示图像，保持原始清晰度
        # 使用默认插值（bilinear），配合高DPI可以保持清晰度
        ax.imshow(frame, aspect='auto')
        ax.axis('off')
        
        # 计算相对路径
        video_path_obj = Path(video_path).resolve()
        try:
            rel_path = video_path_obj.relative_to(root_path)
        except ValueError:
            # 如果无法计算相对路径，使用文件名
            rel_path = video_path_obj.name
        
        # 设置标题（相对路径）
        ax.set_title(str(rel_path), fontsize=7, pad=5, wrap=True)
    
    # 隐藏多余的子图
    for idx in range(n_videos, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows == 1 and cols == 1:
            ax = axes[0, 0]
        elif rows == 1:
            ax = axes[0, col]
        elif cols == 1:
            ax = axes[row, 0]
        else:
            ax = axes[row, col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def main(videos_raw_name, Numb_frame, rawdata_dir="rawdata"):
    """
    主函数
    
    Args:
        videos_raw_name: 要查找的视频名称（可以是部分匹配）
        Numb_frame: 要提取的帧编号（从0开始）
        rawdata_dir: rawdata目录路径，默认为"rawdata"
    """
    print(f"正在查找包含 '{videos_raw_name}' 的视频文件...")
    print(f"目标帧编号: {Numb_frame}")
    print(f"搜索目录: {rawdata_dir}")
    print("-" * 60)
    
    # 查找所有匹配的视频
    video_paths = find_videos_by_name(rawdata_dir, videos_raw_name)
    
    if not video_paths:
        print(f"未找到包含 '{videos_raw_name}' 的视频文件")
        return
    
    print(f"找到 {len(video_paths)} 个匹配的视频文件:")
    for vp in video_paths:
        print(f"  - {vp}")
    print("-" * 60)
    
    # 提取每个视频的指定帧
    print("正在提取帧...")
    frames = []
    valid_paths = []
    
    for video_path in video_paths:
        frame = extract_frame(video_path, Numb_frame)
        if frame is not None:
            frames.append(frame)
            valid_paths.append(video_path)
        else:
            print(f"跳过: {video_path}")
    
    if not frames:
        print("错误：没有成功提取任何帧")
        return
    
    print(f"成功提取 {len(frames)} 个帧")
    print("-" * 60)
    
    # 显示所有帧
    print("正在显示结果...")
    display_frames_grid(valid_paths, frames, rawdata_dir, Numb_frame)


if __name__ == "__main__":
    # 从pipeline.ipynb中获取参数，或者直接在这里设置
    videos_raw_name = ""  # 在这里设置要查找的视频名称
    Numb_frame = 100      # 在这里设置要提取的帧编号
    rawdata_dir = "raw_data"  # rawdata目录路径
    
    # 如果参数为空，提示用户
    if not videos_raw_name:
        print("请设置 videos_raw_name 和 Numb_frame 参数")
        print("示例:")
        print("  videos_raw_name = 'test_video'")
        print("  Numb_frame = 100")
    else:
        main(videos_raw_name, Numb_frame, rawdata_dir)

