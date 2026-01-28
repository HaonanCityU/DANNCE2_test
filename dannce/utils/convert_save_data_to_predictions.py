"""
Convert save_data_AVG.mat to predictions.mat format.

This script converts the prediction output from DANNCE (save_data_AVG.mat) 
into the structured predictions.mat format that can be used by downstream 
analysis tools.

Usage:
    python convert_save_data_to_predictions.py [save_data_AVG.mat] [skeleton.mat] [output_file] [label3d_file (optional)]

Example:
    python convert_save_data_to_predictions.py \
        ./DANNCE/predict_results_AVG/save_data_AVG.mat \
        ./configs/mouse22_skeleton.mat \
        ./DANNCE/predict_results_AVG/predictions.mat \
        ./mouse_val_sh_dannce.mat
"""

import numpy as np
import scipy.io as sio
import sys
import os

def load_skeleton(skeleton_file):
    """Load skeleton file and extract joint names."""
    skeleton = sio.loadmat(skeleton_file)
    skeleton = {k: v for k, v in skeleton.items() if k[0] != '_'}
    
    # Extract joint names
    joint_names = skeleton.get('joint_names', None)
    if joint_names is None:
        raise ValueError("Could not find 'joint_names' in skeleton file")
    
    # Handle different formats
    while np.any(np.array(joint_names.shape) == 1):
        joint_names = np.squeeze(joint_names)
    
    # Convert to list of strings
    if isinstance(joint_names, np.ndarray):
        if joint_names.dtype.names is not None:
            joint_names = [name[0] for name in joint_names[0]]
        else:
            joint_names = [r[0] if isinstance(r, np.ndarray) else str(r) 
                          for r in np.squeeze(joint_names)]
    
    # In Matlab, we cannot keep parentheses in the marker names
    joint_names = [m.replace("(", "_") for m in joint_names]
    joint_names = [m.replace(")", "_") for m in joint_names]
    
    return joint_names

def load_camera_info(label3d_file):
    """Load camera information from label3d file (optional)."""
    try:
        import dannce.engine.io as io
        camnames = io.load_camnames(label3d_file)
        syncs = io.load_sync(label3d_file)
        cparams = io.load_camera_params(label3d_file)
        
        cameras = {}
        for i in range(len(camnames)):
            cameras[camnames[i]] = {}
            
            # Get frame information from sync
            mframes = syncs[i]
            cameras[camnames[i]]["frame"] = mframes.get("data_frame", np.array([]))
            
            # Get camera parameters
            camparams = cparams[i]
            cameras[camnames[i]]["IntrinsicMatrix"] = camparams.get("K", None)
            cameras[camnames[i]]["rotationMatrix"] = camparams.get("r", None)
            cameras[camnames[i]]["translationVector"] = camparams.get("t", None)
            cameras[camnames[i]]["TangentialDistortion"] = camparams.get("TDistort", None)
            cameras[camnames[i]]["RadialDistortion"] = camparams.get("RDistort", None)
            
            # Try to get video directory from config or use default
            cameras[camnames[i]]["video_directory"] = ""
        
        return cameras
    except Exception as e:
        print(f"Warning: Could not load camera information from {label3d_file}: {e}")
        print("Continuing without camera information...")
        return {}

def convert_save_data_to_predictions(save_data_file, skeleton_file, output_file, label3d_file=None):
    """
    Convert save_data_AVG.mat to predictions.mat format.
    
    Args:
        save_data_file: Path to save_data_AVG.mat file
        skeleton_file: Path to skeleton.mat file
        output_file: Path to output predictions.mat file
        label3d_file: Optional path to label3d_dannce.mat file for camera info
    """
    print(f"Loading predictions from: {save_data_file}")
    pred_data = sio.loadmat(save_data_file)
    
    # Extract prediction data
    if 'pred' not in pred_data:
        raise ValueError("Could not find 'pred' field in save_data_AVG.mat")
    
    pred = pred_data['pred']  # Shape: (n_samples, 3, n_keypoints)
    sampleID = np.squeeze(pred_data['sampleID'])
    
    # Get p_max if available
    p_max = pred_data.get('p_max', None)
    
    # Get metadata if available
    metadata = pred_data.get('metadata', None)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Number of samples: {pred.shape[0]}")
    print(f"Number of keypoints: {pred.shape[2]}")
    
    # Load skeleton to get joint names
    print(f"\nLoading skeleton from: {skeleton_file}")
    joint_names = load_skeleton(skeleton_file)
    
    if len(joint_names) != pred.shape[2]:
        raise ValueError(f"Number of joints in skeleton ({len(joint_names)}) "
                         f"does not match prediction keypoints ({pred.shape[2]})")
    
    print(f"Found {len(joint_names)} joint names")
    
    # Convert pred from (n_samples, 3, n_keypoints) to dictionary format
    # Each joint becomes a key with value (n_samples, 3)
    predictions = {}
    for i, joint_name in enumerate(joint_names):
        predictions[joint_name] = pred[:, :, i]  # (n_samples, 3)
    
    predictions["sampleID"] = sampleID
    
    # Load camera information if available
    cameras = {}
    if label3d_file is not None and os.path.exists(label3d_file):
        print(f"\nLoading camera information from: {label3d_file}")
        cameras = load_camera_info(label3d_file)
        if cameras:
            print(f"Loaded information for {len(cameras)} cameras")
    else:
        print("\nNo label3d file provided, skipping camera information")
    
    # Prepare output dictionary
    output_dir = os.path.dirname(output_file)
    if output_dir:
        fullpath = os.path.realpath(output_dir)
    else:
        fullpath = os.path.realpath(".")
    
    name = os.path.split(fullpath)[-1]
    date = os.path.split(os.path.dirname(fullpath))[-1] if os.path.dirname(fullpath) else name
    
    # Try to get network name from metadata
    netname = "unknown"
    if metadata is not None:
        try:
            if hasattr(metadata, 'dtype') and metadata.dtype.names is not None:
                if 'net' in metadata.dtype.names:
                    netname = str(metadata['net'][0, 0][0, 0])
        except:
            pass
    
    # Try to get model path from metadata
    weightspath = ""
    if metadata is not None:
        try:
            if hasattr(metadata, 'dtype') and metadata.dtype.names is not None:
                if 'dannce_predict_model' in metadata.dtype.names:
                    weightspath = str(metadata['dannce_predict_model'][0, 0][0, 0])
        except:
            pass
    
    # Create output dictionary
    output_dict = {
        "fullpath": fullpath,
        "name": name,
        "session": date,
        "netname": netname,
        "predictions": predictions,
        "cameras": cameras,
    }
    
    if p_max is not None:
        output_dict["p_max"] = p_max
    
    if weightspath:
        output_dict["weightspath"] = weightspath
    
    # Save to file
    print(f"\nSaving predictions to: {output_file}")
    sio.savemat(output_file, output_dict)
    
    print("Conversion complete!")
    print(f"Output file: {output_file}")
    print(f"  - {len(joint_names)} joints")
    print(f"  - {pred.shape[0]} samples")
    print(f"  - Camera info: {'Yes' if cameras else 'No'}")

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nError: Missing required arguments")
        print("Usage: python convert_save_data_to_predictions.py "
              "[save_data_AVG.mat] [skeleton.mat] [output_file] [label3d_file (optional)]")
        sys.exit(1)
    
    save_data_file = sys.argv[1]
    skeleton_file = sys.argv[2]
    output_file = sys.argv[3]
    label3d_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Validate input files
    if not os.path.exists(save_data_file):
        raise FileNotFoundError(f"Could not find save_data_AVG.mat file: {save_data_file}")
    
    if not os.path.exists(skeleton_file):
        raise FileNotFoundError(f"Could not find skeleton file: {skeleton_file}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Perform conversion
    convert_save_data_to_predictions(save_data_file, skeleton_file, output_file, label3d_file)

if __name__ == "__main__":
    main()
