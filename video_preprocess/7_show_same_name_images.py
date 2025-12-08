import cv2
import os
import numpy as np
import glob

def get_common_image_files(camera_dirs):
    """
    获取所有摄像头文件夹中共同存在的图片文件名
    
    Args:
        camera_dirs: 包含所有摄像头文件夹路径的列表
    
    Returns:
        sorted_common_files: 按名称排序的共同图片文件名列表
    """
    # 获取每个摄像头文件夹中的jpg文件
    camera_files = []
    for dir_path in camera_dirs:
        # 查找所有jpg文件
        files = set(glob.glob(os.path.join(dir_path, '*.jpg')))
        # 只保留文件名部分（不包含路径）
        file_names = set(os.path.basename(f) for f in files)
        camera_files.append(file_names)
    
    # 找出所有摄像头共有的文件名
    if camera_files:
        common_files = set.intersection(*camera_files)
        # 按文件名排序
        return sorted(common_files)
    return []

def show_images_in_grid(images, filenames):
    """
    以网格形式显示多个图片
    
    Args:
        images: 图片数组列表
        filenames: 文件名列表
    """
    # 确保有6个图片（可能有摄像头没有对应图片的情况）
    while len(images) < 6:
        images.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
    # 调整所有图片的大小到相同尺寸（以最小的为准）
    min_height = min([img.shape[0] for img in images])
    min_width = min([img.shape[1] for img in images])
    
    resized_images = []
    for i, img in enumerate(images):
        # 调整大小
        resized = cv2.resize(img, (min_width, min_height))
        # 添加摄像头标签
        cv2.putText(resized, f"Camera{i+1}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        resized_images.append(resized)
    
    # 创建3x2的布局
    # 第一行：Camera1, Camera2, Camera3
    row1 = np.hstack([resized_images[0], resized_images[1], resized_images[2]])
    # 第二行：Camera4, Camera5, Camera6
    row2 = np.hstack([resized_images[3], resized_images[4], resized_images[5]])
    # 合并两行
    combined = np.vstack([row1, row2])
    
    # 添加当前文件名信息
    if filenames:
        cv2.putText(combined, f"文件名: {filenames[0]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return combined

def main():
    # 基本目录设置
    base_dir = "G:\\demo\\test_shlab\\vessel_4\\videos"
    
    # 首先尝试frames目录（基于之前的信息）
    frames_dir = "G:\\demo\\test_shlab\\vessel_4\\videos\\frames"
    
    # 获取摄像头文件夹路径
    camera_dirs = []
    found = False
    
    # 检查是否存在frames目录
    if os.path.exists(frames_dir):
        # 查找frames下的子目录
        for subdir in os.listdir(frames_dir):
            subdir_path = os.path.join(frames_dir, subdir)
            if os.path.isdir(subdir_path):
                # 在每个子目录中查找Camera1~6
                for i in range(1, 7):
                    camera_path = os.path.join(subdir_path, f"Camera{i}")
                    if os.path.exists(camera_path) and os.path.isdir(camera_path):
                        camera_dirs.append(camera_path)
                        if len(camera_dirs) == 6:
                            found = True
                            break
                if found:
                    break
    
    # 如果在frames目录没找到，直接在videos目录下查找
    if not found:
        for i in range(1, 7):
            camera_path = os.path.join(base_dir, f"Camera{i}")
            if os.path.exists(camera_path) and os.path.isdir(camera_path):
                camera_dirs.append(camera_path)
    
    # 检查是否找到所有6个摄像头目录
    if len(camera_dirs) != 6:
        print(f"警告：只找到了 {len(camera_dirs)} 个摄像头目录，需要6个")
        for path in camera_dirs:
            print(f"找到: {path}")
        return
    
    print("找到所有6个摄像头目录")
    for path in camera_dirs:
        print(f"  {path}")
    
    # 获取共同的图片文件名
    common_files = get_common_image_files(camera_dirs)
    
    if not common_files:
        print("错误：未找到任何共同的图片文件")
        return
    
    print(f"找到 {len(common_files)} 组共同的图片文件")
    
    # 窗口设置
    window_name = "Multi-Camera Image Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 初始索引
    current_index = 0
    
    print("图片查看器已启动")
    print("控制键：")
    print("  A键 - 查看上一组图片")
    print("  D键 - 查看下一组图片")
    print("  ESC - 退出")
    print(f"当前图片: {common_files[current_index]} ({current_index+1}/{len(common_files)})")
    
    while True:
        # 加载当前索引的所有摄像头图片
        current_filename = common_files[current_index]
        images = []
        filenames = []
        
        for i, camera_dir in enumerate(camera_dirs):
            image_path = os.path.join(camera_dir, current_filename)
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    images.append(img)
                    filenames.append(current_filename)
                else:
                    # 如果图片读取失败，使用黑色画面替代
                    print(f"警告：无法读取图片 {image_path}")
                    images.append(np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                # 如果图片不存在，使用黑色画面替代
                print(f"警告：图片不存在 {image_path}")
                images.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # 创建网格显示
        combined_image = show_images_in_grid(images, filenames)
        
        # 显示图片信息
        info_text = f"图片 {current_index+1}/{len(common_files)}: {current_filename}"
        cv2.putText(combined_image, info_text, (10, combined_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 显示图片
        cv2.imshow(window_name, combined_image)
        
        # 处理键盘输入
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC键退出
            print("查看器已关闭")
            break
        elif key == ord('a') or key == ord('A'):  # A键查看上一组
            current_index = (current_index - 1) % len(common_files)
            print(f"切换到上一组图片: {common_files[current_index]} ({current_index+1}/{len(common_files)})")
        elif key == ord('d') or key == ord('D'):  # D键查看下一组
            current_index = (current_index + 1) % len(common_files)
            print(f"切换到下一组图片: {common_files[current_index]} ({current_index+1}/{len(common_files)})")
    
    # 释放资源
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()