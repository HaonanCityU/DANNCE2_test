import cv2
import os
import numpy as np

def create_multi_view(video_paths):
    """
    创建多视频分屏显示窗口
    
    Args:
        video_paths: 包含6个视频路径的列表
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
    original_fps = caps[0].get(cv2.CAP_PROP_FPS) if caps else 30
    playback_speed = 1.0  # 初始播放速度
    frame_delay = int(1000 / (original_fps * playback_speed))  # 帧延迟（毫秒）
    
    # 窗口名称
    window_name = "Multi-Camera Player - Camera1-6"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 播放控制变量
    paused = False
    
    print("多摄像头视频播放器已启动")
    print("控制键：")
    print("  空格 - 暂停/继续播放")
    print("  左方向键 - 看上一帧")
    print("  右方向键 - 看下一帧")
    print("  A键 - 减速播放（最低0.1x）")
    print("  D键 - 加速播放（最高3x）")
    print("  ESC - 退出")
    print("  r - 重置播放")
    print(f"当前播放速度: {playback_speed}x")
    
    # 初始化第一帧
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
    
    while True:
        # 保存当前帧以便暂停时使用
        current_frames = frames.copy()
        
        # 读取下一帧的条件
        should_read_next = False
        
        # 处理键盘输入
        key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF
        
        if key == 27:  # ESC键退出
            print("播放已停止")
            break
        elif key == 32:  # 空格键暂停/继续
            paused = not paused
            print(f"{'已暂停' if paused else '继续播放'}")
            should_read_next = not paused
        elif key == 81:  # 左方向键（看上一帧）
            if paused:
                for cap in caps:
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    # 确保不会回退到负数帧
                    prev_frame = max(0, current_frame - 2)  # 减2是因为读取后会自动+1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
                should_read_next = True
                print("上一帧")
        elif key == 83:  # 右方向键（看下一帧）
            if paused:
                should_read_next = True
                print("下一帧")
        elif key == ord('a') or key == ord('A'):  # A键减速
            playback_speed = max(0.1, playback_speed - 0.1)
            frame_delay = int(1000 / (original_fps * playback_speed))
            print(f"播放速度调整为: {playback_speed:.1f}x")
        elif key == ord('d') or key == ord('D'):  # D键加速
            playback_speed = min(3.0, playback_speed + 0.1)
            frame_delay = int(1000 / (original_fps * playback_speed))
            print(f"播放速度调整为: {playback_speed:.1f}x")
        elif key == ord('r'):  # 'r'键重置播放
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            paused = False
            playback_speed = 1.0
            frame_delay = int(1000 / (original_fps * playback_speed))
            print("播放已重置")
            print(f"当前播放速度: {playback_speed}x")
            should_read_next = True
        elif not paused:
            should_read_next = True
        
        # 读取帧
        if should_read_next:
            frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    # 视频播放完毕，跳回开始
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    
                if ret:
                    frames.append(frame)
                else:
                    print(f"错误：无法读取 Camera{i+1} 的帧")
                    # 使用黑色画面替代
                    height, width = int(caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT)), int(caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
                    frames.append(np.zeros((height, width, 3), dtype=np.uint8))
        else:
            # 保持当前帧
            frames = current_frames.copy()
        
        # 确保有6个帧（可能有视频无法打开的情况）
        while len(frames) < 6:
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # 调整所有帧的大小到相同尺寸（以最小的为准）
        min_height = min([frame.shape[0] for frame in frames])
        min_width = min([frame.shape[1] for frame in frames])
        
        resized_frames = []
        for i, frame in enumerate(frames):
            # 调整大小
            resized = cv2.resize(frame, (min_width, min_height))
            # 添加摄像头标签
            cv2.putText(resized, f"Camera{i+1}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 添加播放状态信息
            status_text = f"{'暂停' if paused else f'{playback_speed:.1f}x'}"
            cv2.putText(resized, status_text, (10, min_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            resized_frames.append(resized)
        
        # 创建3x2的布局
        # 第一行：Camera1, Camera2, Camera3
        row1 = np.hstack([resized_frames[0], resized_frames[1], resized_frames[2]])
        # 第二行：Camera4, Camera5, Camera6
        row2 = np.hstack([resized_frames[3], resized_frames[4], resized_frames[5]])
        # 合并两行
        combined = np.vstack([row1, row2])
        
        # 显示合并后的画面
        cv2.imshow(window_name, combined)
    
    # 释放所有资源
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

def save_multi_view(video_paths, output_path="multi_view_output.mp4"):
    """
    将6路视频分屏合成并保存到文件，不弹出播放窗口。

    Args:
        video_paths: 包含6个视频路径的列表
        output_path: 输出文件路径
    """
    caps = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"警告：无法打开视频 {path}")
            return
        caps.append(cap)

    if not caps:
        print("未找到可用的视频流")
        return

    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    total_frames = min(frame_counts) if frame_counts else 0

    # 读取第一帧以确定尺寸
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        frames.append(frame)

    min_height = min([f.shape[0] for f in frames])
    min_width = min([f.shape[1] for f in frames])
    combined_size = (min_width * 3, min_height * 2)  # (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, combined_size)
    if not writer.isOpened():
        print(f"无法创建输出文件: {output_path}")
        for cap in caps:
            cap.release()
        return

    print(f"开始合成，多摄像头分屏输出 -> {output_path}")
    for _ in range(total_frames):
        frames = []
        valid = True
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                valid = False
                print(f"读取 Camera{i+1} 失败，提前结束。")
                break
            frame = cv2.resize(frame, (min_width, min_height))
            cv2.putText(frame, f"Camera{i+1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frames.append(frame)

        if not valid:
            break

        while len(frames) < 6:
            frames.append(np.zeros((min_height, min_width, 3), dtype=np.uint8))

        row1 = np.hstack([frames[0], frames[1], frames[2]])
        row2 = np.hstack([frames[3], frames[4], frames[5]])
        combined = np.vstack([row1, row2])

        writer.write(combined)

    writer.release()
    for cap in caps:
        cap.release()
    print(f"合成完成，保存于: {output_path}")

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
    
    # 启动多视频播放
    # create_multi_view(video_paths)

    # 或直接合成保存到文件
    save_multi_view(video_paths, output_path="sync_mouse1_all.mp4")

if __name__ == "__main__":
    main()