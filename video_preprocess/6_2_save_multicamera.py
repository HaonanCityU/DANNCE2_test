import cv2
import os
import numpy as np

def save_multi_view(video_paths, output_path):
    """
    将多个视频合并为分屏视频并保存
    
    Args:
        video_paths: 包含6个视频路径的列表
        output_path: 输出视频文件路径
    """
    # 打开所有视频
    caps = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"警告：无法打开视频 {path}")
            return None
        caps.append(cap)
    
    # 获取视频信息
    fps = caps[0].get(cv2.CAP_PROP_FPS) if caps else 30
    frame_count = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT)) if caps else 0
    
    print(f"开始处理视频，共 {frame_count} 帧，FPS: {fps}")
    
    # 读取第一帧以确定尺寸
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
    
    # 确保有6个帧
    while len(frames) < 6:
        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
    # 调整所有帧的大小到相同尺寸（以最小的为准）
    min_height = min([frame.shape[0] for frame in frames])
    min_width = min([frame.shape[1] for frame in frames])
    
    # 计算输出视频尺寸（3x2布局）
    output_width = min_width * 3
    output_height = min_height * 2
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"错误：无法创建输出视频文件 {output_path}")
        for cap in caps:
            cap.release()
        return None
    
    # 重置所有视频到开始位置
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_num = 0
    while True:
        # 读取所有视频的当前帧
        frames = []
        all_finished = True
        
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                all_finished = False
            else:
                # 如果某个视频结束了，使用黑色画面
                frames.append(np.zeros((min_height, min_width, 3), dtype=np.uint8))
        
        # 如果所有视频都结束了，退出循环
        if all_finished:
            break
        
        # 确保有6个帧
        while len(frames) < 6:
            frames.append(np.zeros((min_height, min_width, 3), dtype=np.uint8))
        
        # 调整所有帧的大小到相同尺寸
        resized_frames = []
        for i, frame in enumerate(frames):
            # 调整大小
            resized = cv2.resize(frame, (min_width, min_height))
            # 添加摄像头标签
            cv2.putText(resized, f"Camera{i+1}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            resized_frames.append(resized)
        
        # 创建3x2的布局
        # 第一行：Camera1, Camera2, Camera3
        row1 = np.hstack([resized_frames[0], resized_frames[1], resized_frames[2]])
        # 第二行：Camera4, Camera5, Camera6
        row2 = np.hstack([resized_frames[3], resized_frames[4], resized_frames[5]])
        # 合并两行
        combined = np.vstack([row1, row2])
        
        # 写入视频
        out.write(combined)
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"已处理 {frame_num} 帧...")
    
    # 释放所有资源
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"视频保存完成！共处理 {frame_num} 帧，保存到: {output_path}")

def main():
    # 配置视频路径
    base_dir = "/home/haonan/proj/dannce-release_dev2/demo/sh_exp1/rat1/videos"
    video_names = [f"Camera{i}/sync_mouse1.mp4" for i in range(1, 7)]
    
    # 构建完整路径
    video_paths = [os.path.join(base_dir, name) for name in video_names]
    
    # 检查文件是否存在
    for path in video_paths:
        if not os.path.exists(path):
            print(f"警告：视频文件不存在 {path}")
    
    # 设置输出视频路径
    output_path = os.path.join(base_dir, "multicamera_combined.mp4")
    
    # 保存多视频合并结果
    save_multi_view(video_paths, output_path)

if __name__ == "__main__":
    main()