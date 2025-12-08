import os
import subprocess
from pathlib import Path

def merge_videos_in_folder(folder_path):
    """
    Merge all MP4 files in a folder using ffmpeg
    Returns True if successful, False otherwise
    """
    # Get all MP4 files in the folder
    mp4_files = list(Path(folder_path).glob("*.mp4"))
    
    if not mp4_files:
        print(f"No MP4 files found in {folder_path}")
        return False
    
    # Create file list for ffmpeg
    filelist_path = os.path.join(folder_path, "filelist.txt")
    
    try:
        # Write file list
        with open(filelist_path, "w", encoding="utf-8") as f:
            for file in mp4_files:
                # Use relative path for better compatibility
                f.write(f"file '{file.name}'\n")
        
        # Output file path
        output_path = os.path.join(folder_path, "1.mp4")
        
        # Run ffmpeg command with separate audio and video codecs
        # Using -c:v copy to copy video stream
        # Using -c:a aac to re-encode audio to MP4 compatible format
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", filelist_path,
            "-c:v", "copy",          # 直接复制视频流（不重新编码）
            "-vsync", "0",           # 保留原始时间戳，不强制同步（关键参数）
            "-c:a", "aac",           # 音频重新编码为AAC（确保MP4兼容性）
            "-async", "1",           # 音频同步方式（1表示根据视频时间戳调整）
            output_path,
            "-loglevel", "error",
            "-y"                     # 覆盖现有文件
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Check if output file was created
        if os.path.exists(output_path):
            # Delete original MP4 files
            # for file in mp4_files:
            #     if file.name != "0.mp4":  # Ensure we don't delete our output
            #         os.remove(file)
            
            # # Delete the file list
            # os.remove(filelist_path)
            return True
        else:
            print(f"Output file not created in {folder_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error in {folder_path}: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error processing {folder_path}: {str(e)}")
        return False

def main():
    # Main folder containing Camera1-6 subfolders
    main_folder = "E:/demo/test_shlab/Raw_Data"
    
    # Process each Camera folder from 1 to 6
    for camera_num in range(1, 7):
        camera_folder = os.path.join(main_folder, f"24{camera_num}")
        
        if not os.path.isdir(camera_folder):
            print(f"Folder not found: {camera_folder}")
            continue
        
        print(f"Processing {camera_folder}...")
        success = merge_videos_in_folder(camera_folder)
        
        if success:
            print(f"Successfully processed {camera_folder}")
        else:
            print(f"Failed to process {camera_folder}")
    
    print("Processing complete")

if __name__ == "__main__":
    main()
    