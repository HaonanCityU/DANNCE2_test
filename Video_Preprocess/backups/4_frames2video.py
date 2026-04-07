"""
Frames to Video Converter

Usage:
    This script converts extracted frames back into video files. It reads frames from 
    directories organized by camera (Camera1, Camera2, etc.) and combines them into 
    synchronized video files.

    The script:
    1. Scans frame directories for image files with format: row_XXXXX_frame_YYYYYYY.jpg
    2. Filters frames based on CSV row number range (start_frame to end_frame)
    3. Sorts frames by CSV row number to maintain synchronization
    4. Combines frames into video files using OpenCV
    5. Processes all Camera folders (Camera1-Camera6) automatically

    Frame File Format:
        Files should be named: row_XXXXX_frame_YYYYYYY.jpg
        - XXXXX: CSV row number (5 digits, zero-padded)
        - YYYYYYY: Frame number (7 digits, zero-padded)
        Example: row_00001_frame_0001000.jpg

    Input Structure:
        frames_root/
        ├── Camera1/
        │   ├── row_00001_frame_0001000.jpg
        │   ├── row_00002_frame_0001001.jpg
        │   └── ...
        ├── Camera2/
        │   └── ...
        └── ...

    Output Structure:
        output_root/
        ├── Camera1/
        │   └── output_name.mp4
        ├── Camera2/
        │   └── output_name.mp4
        └── ...

    Example:
        Modify the parameters in the main() function:
        - FRAMES_ROOT: Root directory containing Camera folders with extracted frames
        - OUTPUT_ROOT: Root directory for output videos
        - START_FRAME: Starting CSV row number (inclusive)
        - END_FRAME: Ending CSV row number (inclusive)
        - FPS: Output video frame rate (default: 60)
        - output_name: Output video filename (e.g., "0.mp4")

    Requirements:
        - opencv-python (cv2)
        - tqdm (for progress bars)
"""

import cv2
import os
import sys
from tqdm import tqdm

def frames_to_video(frame_dir, output_path, start_frame, end_frame, fps=60):
    """
    Combine images within specified range into a video
    
    Args:
        frame_dir: Directory containing images (files with format row_XXXXX_frame_YYYYYYY.jpg)
        output_path: Output video path
        start_frame: Starting CSV row number (first number in filename)
        end_frame: Ending CSV row number (first number in filename)
        fps: Output video frame rate
    """
    # Scan directory for all matching image files
    # Filename format: row_08000_frame_008445.jpg
    print(f"Scanning image files in {frame_dir}...")
    all_frame_files = []  # Store all valid filenames with their CSV row numbers
    
    for filename in os.listdir(frame_dir):
        if filename.endswith('.jpg') and filename.startswith('row_'):
            try:
                # Parse filename: row_08000_frame_008445.jpg
                parts = filename.replace('.jpg', '').split('_')
                if len(parts) >= 4 and parts[0] == 'row' and parts[2] == 'frame':
                    csv_row = int(parts[1])  # CSV row number
                    frame_num = int(parts[3])  # Frame number
                    all_frame_files.append((csv_row, frame_num, filename))
            except (ValueError, IndexError):
                continue
    
    if not all_frame_files:
        print(f"Error: No valid image files found in {frame_dir}")
        return False
    
    # Find actual min and max CSV row numbers
    csv_rows = [row for row, _, _ in all_frame_files]
    actual_min_row = min(csv_rows)
    actual_max_row = max(csv_rows)
    
    # Adjust start_frame and end_frame if they exceed actual range
    adjusted_start = start_frame
    adjusted_end = end_frame
    
    if start_frame < actual_min_row:
        adjusted_start = actual_min_row
        print(f"⚠️  Warning: START_FRAME ({start_frame}) is less than minimum available row ({actual_min_row}). Adjusted to {actual_min_row}")
    
    if end_frame > actual_max_row:
        adjusted_end = actual_max_row
        print(f"⚠️  Warning: END_FRAME ({end_frame}) is greater than maximum available row ({actual_max_row}). Adjusted to {actual_max_row}")
    
    # Filter frames within adjusted range
    valid_frame_files = []
    for csv_row, frame_num, filename in all_frame_files:
        if adjusted_start <= csv_row <= adjusted_end:
            valid_frame_files.append((csv_row, frame_num, filename))
    
    # Sort by CSV row number
    valid_frame_files.sort(key=lambda x: x[0])
    
    if not valid_frame_files:
        print(f"Error: No valid images found in {frame_dir} within range {adjusted_start}~{adjusted_end}")
        return False
    
    print(f"Using frame range: {adjusted_start}~{adjusted_end} (actual available: {actual_min_row}~{actual_max_row})")
    
    print(f"Found {len(valid_frame_files)} valid frame images")
    
    # Get first image to determine dimensions
    first_frame_path = os.path.join(frame_dir, valid_frame_files[0][2])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create video file {output_path}")
        return False
    
    # Process each frame
    print(f"Processing valid frames in {frame_dir}")
    success_count = 0
    
    with tqdm(total=len(valid_frame_files), desc=f"Generating video {os.path.basename(output_path)}") as pbar:
        for csv_row, frame_num, filename in valid_frame_files:
            # Use complete filename
            frame_path = os.path.join(frame_dir, filename)
            
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
                success_count += 1
            else:
                print(f"Warning: Cannot read frame {frame_path}")
            
            pbar.update(1)
    
    # Release resources
    out.release()
    
    if success_count == 0:
        print(f"Error: No frames processed successfully")
        # Delete empty file
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    
    print(f"Successfully generated video: {output_path}")
    print(f"Successfully processed {success_count} frames")
    return True

def process_cameras(frames_root, output_root, start_frame, end_frame, fps=60, output_name="sync.mp4"):
    """
    Process all Camera folders
    """
    # Ensure output root directory exists
    os.makedirs(output_root, exist_ok=True)
    
    # Process Camera1 to Camera6
    for camera_num in range(1, 7):
        camera_name = f"Camera{camera_num}"
        
        # Input frame directory
        frame_dir = os.path.join(frames_root, camera_name)
        if not os.path.isdir(frame_dir):
            print(f"Warning: Directory not found {frame_dir}")
            continue
        
        # Output video directory
        output_dir = os.path.join(output_root, camera_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Output video path
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\nProcessing {camera_name}...")
        success = frames_to_video(frame_dir, output_path, start_frame, end_frame, fps)
        
        if success:
            print(f"{camera_name} processing completed!")
        else:
            print(f"{camera_name} processing failed!")

def main():
    # Configuration parameters
    RAT_NAME = "Rat2"
    video_name = "2"
    START_FRAME = 1
    END_FRAME = 100218
    project_root = "/home/haonan/proj/dannce-release_dev2/demo/hanshu_20260304_batch1"

    if video_name == "sync" or video_name == "cali":
        FRAMES_ROOT = f"{project_root}/{RAT_NAME}/frames/0.mp4/all"
    else:
        FRAMES_ROOT = f"{project_root}/{RAT_NAME}/frames/{video_name}.mp4/all"
    OUTPUT_ROOT = f"{project_root}/{RAT_NAME}/videos"
    FPS = 60
    output_name = f"{video_name}.mp4"
    
    # Print configuration information
    print("===== Frames to Video Converter =====")
    print(f"Frames directory: {FRAMES_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Frame range: {START_FRAME}~{END_FRAME}")
    print(f"Output frame rate: {FPS} fps")
    print(f"Output filename: {output_name}")
    print("====================================")
    
    # Start processing
    process_cameras(FRAMES_ROOT, OUTPUT_ROOT, START_FRAME, END_FRAME, FPS, output_name)
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()
