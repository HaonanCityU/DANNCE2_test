"""
Evaluate DANNCE predictions against ground truth labels using Euclidean distance.

This script loads ground truth labels from a label3d file and compares them with
predictions from a predictions.mat file, calculating per-keypoint and overall
Euclidean distance errors.

Usage:
    python evaluate_predictions.py [ground_truth_file] [predictions_file] [skeleton_file] [output_dir]
    
Example:
    python evaluate_predictions.py \
      mouse_val_sh_dannce.mat \
      predictions.mat \
      mouse22_skeleton.mat \
      ./evaluation_results
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from dannce.engine import io as dio

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_ground_truth(gt_file):
    """Load ground truth labels from label3d file."""
    print("Loading ground truth from: {}".format(gt_file))
    
    # Load labelData
    labels = dio.load_labels(gt_file)
    
    # Extract 3D data from first camera (all cameras should have same 3D data)
    # labels[0]["data_3d"] shape: (n_samples, 3 * n_keypoints)
    data_3d = labels[0]["data_3d"]  # (n_samples, 3 * n_keypoints)
    n_samples = data_3d.shape[0]
    n_keypoints = data_3d.shape[1] // 3
    
    # Reshape to (n_samples, n_keypoints, 3) then transpose to (n_samples, 3, n_keypoints)
    # This matches the format used in serve_data_DANNCE.py line 126
    gt_3d = np.transpose(np.reshape(data_3d, [n_samples, n_keypoints, 3]), [0, 2, 1])
    
    # Get sampleID and data_frame from labels (not sync)
    # data_sampleID corresponds to the rows in data_3d
    sampleID = np.squeeze(labels[0]["data_sampleID"])
    data_frame = np.squeeze(labels[0]["data_frame"])
    
    # Ensure sampleID and data_frame have the same length as data_3d
    if len(sampleID) != n_samples:
        print("Warning: sampleID length ({}) != data_3d rows ({}). Using first {} elements.".format(
            len(sampleID), n_samples, n_samples))
        sampleID = sampleID[:n_samples]
    
    if len(data_frame) != n_samples:
        print("Warning: data_frame length ({}) != data_3d rows ({}). Using first {} elements.".format(
            len(data_frame), n_samples, n_samples))
        data_frame = data_frame[:n_samples]
    
    print("Ground truth shape: {}".format(gt_3d.shape))
    print("Number of samples: {}".format(n_samples))
    print("Number of keypoints: {}".format(n_keypoints))
    print("sampleID range: {} to {}".format(np.min(sampleID), np.max(sampleID)))
    print("data_frame range: {} to {}".format(np.min(data_frame), np.max(data_frame)))
    
    return gt_3d, sampleID, data_frame

def load_predictions(pred_file):
    """Load predictions from predictions.mat file."""
    print("Loading predictions from: {}".format(pred_file))
    
    data = sio.loadmat(pred_file)
    
    # Check if it's the structured format (from makeStructuredData)
    if "predictions" in data:
        pred_dict = data["predictions"]
        
        # Handle structured array format
        if isinstance(pred_dict, np.ndarray) and pred_dict.dtype.names is not None:
            # Get joint names from the predictions dict
            joint_names = [k for k in pred_dict.dtype.names if k != "sampleID"]
            n_keypoints = len(joint_names)
            
            # Get first joint to determine n_samples
            first_joint_data = pred_dict[joint_names[0]][0, 0]
            if len(first_joint_data.shape) == 2:
                n_samples = first_joint_data.shape[0]
            else:
                n_samples = first_joint_data.shape[1]
            
            # Stack predictions: (n_samples, 3, n_keypoints)
            pred_3d = np.zeros((n_samples, 3, n_keypoints))
            for i, name in enumerate(joint_names):
                joint_data = pred_dict[name][0, 0]
                if joint_data.shape[0] == 3:
                    pred_3d[:, :, i] = joint_data.T  # (n_samples, 3)
                else:
                    pred_3d[:, :, i] = joint_data  # Already (n_samples, 3)
            
            sampleID = np.squeeze(pred_dict["sampleID"][0, 0])
        else:
            # Dictionary format
            joint_names = [k for k in pred_dict.keys() if k != "sampleID"]
            n_keypoints = len(joint_names)
            n_samples = pred_dict[joint_names[0]].shape[0]
            
            pred_3d = np.zeros((n_samples, 3, n_keypoints))
            for i, name in enumerate(joint_names):
                if pred_dict[name].shape[0] == 3:
                    pred_3d[:, :, i] = pred_dict[name].T
                else:
                    pred_3d[:, :, i] = pred_dict[name]
            
            sampleID = np.squeeze(pred_dict["sampleID"])
        
    elif "pred" in data:
        # Direct prediction format (from save_data_AVG.mat)
        pred_3d = data["pred"]  # Shape: (n_samples, 3, n_keypoints)
        sampleID = np.squeeze(data["sampleID"])
    else:
        raise ValueError("Could not find 'predictions' or 'pred' field in predictions file. Available keys: {}".format(list(data.keys())))
    
    print("Predictions shape: {}".format(pred_3d.shape))
    print("Number of samples: {}".format(pred_3d.shape[0]))
    print("Number of keypoints: {}".format(pred_3d.shape[2]))
    
    return pred_3d, sampleID

def calculate_euclidean_distance(y_true, y_pred):
    """
    Calculate Euclidean distance between true and predicted 3D coordinates.
    
    Args:
        y_true: (n_samples, 3, n_keypoints) ground truth
        y_pred: (n_samples, 3, n_keypoints) predictions
    
    Returns:
        errors: (n_samples, n_keypoints) Euclidean distance per sample and keypoint
    """
    # Calculate distance: sqrt(sum((y_true - y_pred)^2, axis=1))
    # Handle NaN values: if either true or pred has NaN, the error should be NaN
    diff = y_true - y_pred
    
    # Check for NaN in either true or pred
    has_nan = np.isnan(y_true) | np.isnan(y_pred)
    # If any coordinate (x, y, z) is NaN, mark the whole keypoint as NaN
    has_nan_any = np.any(has_nan, axis=1)  # (n_samples, n_keypoints)
    
    errors = np.sqrt(np.sum(diff**2, axis=1))  # (n_samples, n_keypoints)
    
    # Set error to NaN where either true or pred has NaN
    errors[has_nan_any] = np.nan
    
    return errors

def match_samples(gt_3d, gt_sampleID, gt_data_frame, pred_3d, pred_sampleID):
    """Match ground truth and predictions based on sampleID."""
    print("\nMatching samples based on sampleID...")
    
    # Find common sampleIDs
    gt_sid = np.squeeze(gt_sampleID)
    pred_sid = np.squeeze(pred_sampleID)
    
    # Ensure they are 1D arrays
    if gt_sid.ndim > 1:
        gt_sid = gt_sid.flatten()
    if pred_sid.ndim > 1:
        pred_sid = pred_sid.flatten()
    
    # Find matching indices
    common_ids, gt_indices, pred_indices = np.intersect1d(
        gt_sid, pred_sid, return_indices=True
    )
    
    print("Ground truth unique sampleIDs: {}".format(len(np.unique(gt_sid))))
    print("Ground truth total entries: {}".format(len(gt_sid)))
    print("Ground truth data_3d rows: {}".format(gt_3d.shape[0]))
    print("Prediction samples: {}".format(len(pred_sid)))
    print("Matched sampleIDs: {}".format(len(common_ids)))
    
    if len(common_ids) == 0:
        raise ValueError("No matching sampleIDs found between ground truth and predictions!")
    
    # Check if indices are within bounds
    if np.any(gt_indices >= gt_3d.shape[0]):
        print("Warning: Some gt_indices are out of bounds. Filtering...")
        valid_mask = gt_indices < gt_3d.shape[0]
        gt_indices = gt_indices[valid_mask]
        pred_indices = pred_indices[valid_mask]
        common_ids = common_ids[valid_mask]
        print("Valid matches after filtering: {}".format(len(common_ids)))
    
    if np.any(pred_indices >= pred_3d.shape[0]):
        print("Warning: Some pred_indices are out of bounds. Filtering...")
        valid_mask = pred_indices < pred_3d.shape[0]
        gt_indices = gt_indices[valid_mask]
        pred_indices = pred_indices[valid_mask]
        common_ids = common_ids[valid_mask]
        print("Valid matches after filtering: {}".format(len(common_ids)))
    
    # Extract matched data
    gt_matched = gt_3d[gt_indices]
    pred_matched = pred_3d[pred_indices]
    
    print("Final matched samples: {}".format(len(common_ids)))
    print("GT matched shape: {}".format(gt_matched.shape))
    print("Pred matched shape: {}".format(pred_matched.shape))
    
    return gt_matched, pred_matched, common_ids


def match_samples_by_data_frame(
    gt_3d,
    gt_sampleID,
    gt_data_frame,
    pred_3d,
    pred_sampleID,
    sync_file,
):
    """
    Match ground truth and predictions by global frame index (data_frame).

    Use-case:
      - GT comes from a sparse labeled file (mouse_dannce.mat) whose sampleID can be irregular timestamps.
      - Predictions come from full-video inference (save_data_AVG.mat / predictions.mat) whose sampleID
        often follows sync sampleID (e.g. 1, 34, 67, ...).
      - In that case, matching by sampleID yields zero overlap, but matching by data_frame is valid:
          pred_sampleID --(sync_file: sync sampleID->data_frame)--> pred_frame
          gt_data_frame -------------------------------------------> gt_frame
          match pred_frame with gt_frame

    Args:
        gt_3d: (N_gt, 3, K)
        gt_sampleID: (N_gt,)
        gt_data_frame: (N_gt,)
        pred_3d: (N_pred, 3, K)
        pred_sampleID: (N_pred,)
        sync_file: fullsync dannce.mat containing sync (same timeline as predictions)

    Returns:
        gt_matched, pred_matched, matched_frames
    """
    print("\nMatching samples based on data_frame (via sync)...")
    if sync_file is None:
        raise ValueError("sync_file is required for match_samples_by_data_frame")

    # Flatten and cast
    gt_frames = np.asarray(gt_data_frame).reshape(-1).astype(int)
    pred_sid = np.asarray(pred_sampleID).reshape(-1).astype(int)

    # Load sync mapping (use camera 0; all cams share sampleID timeline)
    sync0 = dio.load_sync(sync_file)[0]
    sync_sid = np.asarray(sync0["data_sampleID"]).reshape(-1).astype(int)
    sync_df = np.asarray(sync0["data_frame"]).reshape(-1).astype(int)

    # Map prediction sampleIDs -> global frame indices using intersect
    common_sid, sync_idx, pred_idx = np.intersect1d(sync_sid, pred_sid, return_indices=True)
    print("Pred samples:", pred_sid.size)
    print("Sync sampleIDs:", sync_sid.size)
    print("Matched pred sampleIDs in sync:", common_sid.size)
    if common_sid.size == 0:
        raise ValueError("No prediction sampleIDs found in sync_file sampleIDs.")

    pred_frames = sync_df[sync_idx]  # aligned with pred_idx

    # Now match by frame number
    common_frames, gt_i, pred_f_i = np.intersect1d(gt_frames, pred_frames, return_indices=True)
    print("GT labeled frames:", gt_frames.size)
    print("Pred frames (mapped):", pred_frames.size)
    print("Matched frames:", common_frames.size)
    if common_frames.size == 0:
        raise ValueError("No matching frames found between GT data_frame and prediction frames.")

    # Subset arrays
    gt_matched = gt_3d[gt_i]
    pred_matched = pred_3d[pred_idx[pred_f_i]]

    print("GT matched shape:", gt_matched.shape)
    print("Pred matched shape:", pred_matched.shape)
    return gt_matched, pred_matched, common_frames


def match_samples_by_data_frame(
    gt_3d,
    gt_sampleID,
    gt_data_frame,
    pred_3d,
    pred_sampleID,
    sync_file,
):
    """
    Match ground truth and predictions by global frame index (data_frame).

    Use-case:
      - GT comes from a sparse labeled file (mouse_dannce.mat) whose sampleID can be irregular timestamps.
      - Predictions come from full-video inference (save_data_AVG.mat / predictions.mat) whose sampleID
        often follows sync sampleID (e.g. 1, 34, 67, ...).
      - In that case, matching by sampleID yields zero overlap, but matching by data_frame is valid:
          pred_sampleID --(sync_file: sync sampleID->data_frame)--> pred_frame
          gt_data_frame -------------------------------------------> gt_frame
          match pred_frame with gt_frame

    Args:
        gt_3d: (N_gt, 3, K)
        gt_sampleID: (N_gt,)
        gt_data_frame: (N_gt,)
        pred_3d: (N_pred, 3, K)
        pred_sampleID: (N_pred,)
        sync_file: fullsync dannce.mat containing sync (same timeline as predictions)

    Returns:
        gt_matched, pred_matched, matched_frames
    """
    print("\nMatching samples based on data_frame (via sync)...")
    if sync_file is None:
        raise ValueError("sync_file is required for match_samples_by_data_frame")

    # Flatten and cast
    gt_frames = np.asarray(gt_data_frame).reshape(-1).astype(int)
    pred_sid = np.asarray(pred_sampleID).reshape(-1).astype(int)

    # Load sync mapping (use camera 0; all cams share sampleID timeline)
    sync0 = dio.load_sync(sync_file)[0]
    sync_sid = np.asarray(sync0["data_sampleID"]).reshape(-1).astype(int)
    sync_df = np.asarray(sync0["data_frame"]).reshape(-1).astype(int)

    # Map prediction sampleIDs -> global frame indices using intersect
    common_sid, sync_idx, pred_idx = np.intersect1d(sync_sid, pred_sid, return_indices=True)
    print("Pred samples:", pred_sid.size)
    print("Sync sampleIDs:", sync_sid.size)
    print("Matched pred sampleIDs in sync:", common_sid.size)
    if common_sid.size == 0:
        raise ValueError("No prediction sampleIDs found in sync_file sampleIDs.")

    pred_frames = sync_df[sync_idx]  # aligned with pred_idx

    # Now match by frame number
    common_frames, gt_i, pred_f_i = np.intersect1d(gt_frames, pred_frames, return_indices=True)
    print("GT labeled frames:", gt_frames.size)
    print("Pred frames (mapped):", pred_frames.size)
    print("Matched frames:", common_frames.size)
    if common_frames.size == 0:
        raise ValueError("No matching frames found between GT data_frame and prediction frames.")

    # Subset arrays
    gt_matched = gt_3d[gt_i]
    pred_matched = pred_3d[pred_idx[pred_f_i]]

    print("GT matched shape:", gt_matched.shape)
    print("Pred matched shape:", pred_matched.shape)
    return gt_matched, pred_matched, common_frames

def plot_evaluation_results(errors, joint_names, output_dir):
    """Generate comprehensive evaluation plots."""
    
    n_samples, n_keypoints = errors.shape
    
    # 1. Per-keypoint error distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Box plot per keypoint using seaborn
    ax = axes[0, 0]
    # Filter out NaN values for each keypoint
    error_data = []
    valid_keypoints = []
    valid_names = []
    for i in range(n_keypoints):
        keypoint_errors = errors[:, i]
        valid_errors = keypoint_errors[~np.isnan(keypoint_errors)]
        if len(valid_errors) > 0:
            error_data.append(valid_errors)
            valid_keypoints.append(i)
            valid_names.append(joint_names[i])
    
    if len(error_data) > 0:
        # Prepare data for seaborn
        plot_data = []
        plot_labels = []
        for i, (data, name) in enumerate(zip(error_data, valid_names)):
            plot_data.extend(data)
            plot_labels.extend([name] * len(data))
        
        df_plot = pd.DataFrame({'Error (mm)': plot_data, 'Keypoint': plot_labels})
        
        # Create boxplot with seaborn
        # Boxplot shows: box (Q1, median, Q3), whiskers (1.5*IQR range), and outliers (black dots)
        sns.boxplot(data=df_plot, x='Keypoint', y='Error (mm)', ax=ax, 
                   showfliers=True)  # showfliers=True shows outliers as black dots
        ax.set_ylabel('Euclidean Distance Error (mm)', fontsize=12, fontweight='bold')
        ax.set_xlabel('', fontsize=12)
        ax.set_title('Per-Keypoint Error Distribution ({} keypoints with data)\n'
                     'Black dots = outliers (>1.5×IQR from Q1/Q3)'.format(len(valid_keypoints)), 
                     fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Per-Keypoint Error Distribution (No data)', fontsize=14)
    
    # Mean error per keypoint with error bars
    ax = axes[0, 1]
    mean_errors = np.nanmean(errors, axis=0)
    std_errors = np.nanstd(errors, axis=0)
    q25 = np.nanpercentile(errors, 25, axis=0)
    q75 = np.nanpercentile(errors, 75, axis=0)
    iqr = q75 - q25  # Interquartile Range
    
    # Count valid samples per keypoint
    valid_counts = np.sum(~np.isnan(errors), axis=0)
    
    # Only plot keypoints with valid data
    if len(valid_keypoints) > 0:
        valid_mean = mean_errors[valid_keypoints]
        valid_std = std_errors[valid_keypoints]
        valid_iqr = iqr[valid_keypoints]
        x_pos = np.arange(len(valid_keypoints))
        
        # Create bar plot with error bars using seaborn style
        bars = ax.bar(x_pos, valid_mean, yerr=valid_std, 
                     alpha=0.8, capsize=5, 
                     error_kw={'elinewidth': 2, 'ecolor': 'darkred', 'capthick': 2})
        
        # Color bars with seaborn palette
        colors = sns.color_palette("husl", len(valid_keypoints))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add sample count text annotations above the bars
        for i, (x, mean, std, count) in enumerate(zip(x_pos, valid_mean, valid_std, valid_counts[valid_keypoints])):
            # Calculate max value for positioning
            max_val = np.max(valid_mean + valid_std)
            # Display sample count above the error bar
            ax.text(x, mean + std + 0.02 * max_val, 'n={}'.format(int(count)), 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Error (mm)', fontsize=12, fontweight='bold')
        ax.set_xlabel('', fontsize=12)
        ax.set_title('Mean Error per Keypoint ({} keypoints with data)'.format(len(valid_keypoints)), 
                     fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Mean Error per Keypoint (No data)', fontsize=14)
    
    # Overall error distribution using seaborn
    ax = axes[1, 0]
    valid_errors = errors[~np.isnan(errors)]
    if len(valid_errors) > 0:
        # Use seaborn distplot for better visualization
        sns.histplot(valid_errors, bins=50, kde=True, ax=ax, 
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
        mean_err = np.nanmean(errors)
        median_err = np.nanmedian(errors)
        std_err = np.nanstd(errors)
        q25_err = np.nanpercentile(errors, 25)
        q75_err = np.nanpercentile(errors, 75)
        iqr_err = q75_err - q25_err
        ax.axvline(mean_err, color='red', linestyle='--', linewidth=2,
                   label='Mean: {:.2f} mm'.format(mean_err))
        ax.axvline(median_err, color='green', linestyle='--', linewidth=2,
                   label='Median: {:.2f} mm'.format(median_err))
        # ax.axvline(mean_err - std_err, color='purple', linestyle=':', linewidth=2,
        #            label='Mean - Std: {:.2f} mm'.format(mean_err - std_err))
        # ax.axvline(mean_err + std_err, color='purple', linestyle=':', linewidth=2,
        #            label='Mean + Std: {:.2f} mm (Std={:.2f} mm)'.format(mean_err + std_err, std_err))
        ax.axvline(q25_err, color='orange', linestyle=':', linewidth=2,
                   label='Q25: {:.2f} mm'.format(q25_err))
        ax.axvline(q75_err, color='orange', linestyle=':', linewidth=2,
                   label='Q75: {:.2f} mm (IQR={:.2f} mm)'.format(q75_err, iqr_err))
        ax.set_xlabel('Error (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Overall Error Distribution ({} valid values)'.format(len(valid_errors)), 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    else:
        ax.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Overall Error Distribution (No data)', fontsize=14)
    
    # Error over time (mean across keypoints)
    ax = axes[1, 1]
    mean_error_time = np.nanmean(errors, axis=1)
    # Use seaborn lineplot for smoother visualization
    time_df = pd.DataFrame({'Sample Index': np.arange(len(mean_error_time)), 
                            'Mean Error (mm)': mean_error_time})
    sns.lineplot(data=time_df, x='Sample Index', y='Mean Error (mm)', ax=ax, 
                linewidth=2, color='steelblue')
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Error (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Error Over Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=300)
    plt.close()
    
    # 2. Success rate plot using seaborn
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    thresholds = np.arange(0, 50, 1)  # 0 to 50mm
    success_rates = []
    for thresh in thresholds:
        success_rate = np.nanmean(errors < thresh) * 100
        success_rates.append(success_rate)
    
    # Use seaborn lineplot
    success_df = pd.DataFrame({'Error Threshold (mm)': thresholds, 
                              'Success Rate (%)': success_rates})
    sns.lineplot(data=success_df, x='Error Threshold (mm)', y='Success Rate (%)', 
                ax=ax, linewidth=3, color='steelblue')
    ax.set_xlabel('Error Threshold (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate vs Error Threshold', fontsize=14, fontweight='bold')
    
    # Mark common thresholds with styled markers
    for thresh in [5, 10, 20]:
        idx = np.argmin(np.abs(thresholds - thresh))
        ax.plot(thresh, success_rates[idx], 'o', markersize=12, 
               color='crimson', markeredgecolor='white', markeredgewidth=2,
               zorder=5)
        ax.text(thresh, success_rates[idx] + 3, 
                '{:.1f}%'.format(success_rates[idx]), 
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate.png'), dpi=300)
    plt.close()
    
    # 3. Per-keypoint statistics table
    stats = {
        'mean': np.nanmean(errors, axis=0),
        'std': np.nanstd(errors, axis=0),
        'median': np.nanmedian(errors, axis=0),
        'min': np.nanmin(errors, axis=0),
        'max': np.nanmax(errors, axis=0),
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'p95': np.nanpercentile(errors, 95, axis=0),
    }
    
    return stats

def plot_landmark_tracking_accuracy(errors, output_dir, threshold=18.0):
    """
    Plot landmark tracking accuracy: fraction of timepoints with at least n landmarks accurately tracked.
    
    Args:
        errors: Error array with shape (n_samples, n_keypoints)
        output_dir: Output directory for saving the plot
        threshold: Error threshold in mm (default 18.0 for rat experiments)
    """
    n_samples, n_keypoints = errors.shape
    
    # For each frame, count how many landmarks are accurately tracked (error < threshold)
    # Handle NaN values: if error is NaN, consider it as not accurately tracked
    accurately_tracked = (errors < threshold) & (~np.isnan(errors))  # (n_samples, n_keypoints)
    n_accurate_per_frame = np.sum(accurately_tracked, axis=1)  # (n_samples,)
    
    # For each n (0 to n_keypoints), calculate fraction of timepoints with at least n landmarks accurately tracked
    n_landmarks_range = np.arange(0, n_keypoints + 1)
    fractions = []
    
    for n in n_landmarks_range:
        # Count frames with at least n landmarks accurately tracked
        frames_with_at_least_n = np.sum(n_accurate_per_frame >= n)
        fraction = frames_with_at_least_n / n_samples if n_samples > 0 else 0.0
        fractions.append(fraction)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Use seaborn lineplot for better visualization
    accuracy_df = pd.DataFrame({
        'No. landmarks accurately tracked': n_landmarks_range,
        'Fraction of timepoints': fractions
    })
    sns.lineplot(data=accuracy_df, x='No. landmarks accurately tracked', 
                y='Fraction of timepoints', ax=ax, linewidth=3, color='steelblue', marker='o', markersize=6)
    
    # Fill area under the curve
    ax.fill_between(n_landmarks_range, fractions, alpha=0.3, color='steelblue')
    
    # Set labels and title
    ax.set_xlabel('No. landmarks accurately tracked', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of timepoints', fontsize=12, fontweight='bold')
    ax.set_title('Landmark Tracking Accuracy\n(Threshold: {:.1f} mm)'.format(threshold), 
                fontsize=14, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0, n_keypoints])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Mark some key points
    # Find the point where fraction drops below 0.5
    fractions_array = np.array(fractions)
    below_50_mask = fractions_array < 0.5
    if np.any(below_50_mask):
        idx_50 = np.argmax(below_50_mask)
        if idx_50 > 0:
            ax.plot(n_landmarks_range[idx_50], fractions[idx_50], 'o', markersize=12,
                   color='crimson', markeredgecolor='white', markeredgewidth=2, zorder=5)
            ax.text(n_landmarks_range[idx_50], fractions[idx_50] + 0.05,
                   f'{n_landmarks_range[idx_50]} landmarks\n({fractions[idx_50]:.2%})',
                   ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add text annotation for total landmarks
    ax.text(0.98, 0.02, f'Total landmarks: {n_keypoints}',
           transform=ax.transAxes, ha='right', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'landmark_tracking_accuracy.png'), dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\nLandmark Tracking Accuracy Summary (Threshold: {:.1f} mm):".format(threshold))
    print("-" * 70)
    print("Frames with all landmarks accurately tracked: {:.2%} ({}/{})".format(
        fractions[n_keypoints], int(fractions[n_keypoints] * n_samples), n_samples))
    print("Frames with at least {} landmarks accurately tracked: {:.2%} ({}/{})".format(
        n_keypoints // 2, fractions[n_keypoints // 2], 
        int(fractions[n_keypoints // 2] * n_samples), n_samples))
    print("Frames with at least 1 landmark accurately tracked: {:.2%} ({}/{})".format(
        fractions[1], int(fractions[1] * n_samples), n_samples))
    
    # Find median number of accurately tracked landmarks
    median_n_accurate = np.median(n_accurate_per_frame)
    print("Median number of accurately tracked landmarks per frame: {:.1f}".format(median_n_accurate))

def plot_missing_data_availability_heatmap(errors, sampleID, output_dir, joint_names=None):
    """
    Plot a heatmap showing data availability (missing values) across all frames in evaluation.
    This shows where NaN values occur in the error calculations, which can come from
    either ground truth labels or predictions having missing values.
    
    Args:
        errors: Error array with shape (n_samples, n_keypoints), may contain NaN
        sampleID: Sample IDs corresponding to each frame
        output_dir: Output directory for saving the plot
        joint_names: List of joint/keypoint names
    """
    n_samples, n_keypoints = errors.shape
    
    # Use joint names if available
    if joint_names is None or len(joint_names) != n_keypoints:
        joint_names = [f"Keypoint_{i}" for i in range(n_keypoints)]
    
    # Check for NaN values in errors
    # NaN in errors means either ground truth or prediction had missing values
    missing_per_landmark = np.isnan(errors)  # (n_samples, n_keypoints)
    # True means missing (has NaN), False means present (no NaN)
    
    # Create the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Convert boolean to numeric for visualization: 1 = missing (red), 0 = present (light blue)
    heatmap_data = missing_per_landmark.astype(float).T  # Transpose: (n_keypoints, n_samples)
    
    # Create custom colormap: light blue for 0 (present), red for 1 (missing)
    from matplotlib.colors import ListedColormap
    colors = ['#ADD8E6', '#FF0000']  # Light blue, Red
    cmap = ListedColormap(colors)
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, 
                   interpolation='nearest', origin='lower', vmin=0, vmax=1)
    
    # Set labels
    ax.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Landmark', fontsize=12, fontweight='bold')
    ax.set_title('Landmark Availability Across All Frames (Evaluation)\n(Red: Unavailable, Light Blue: Available)', 
                fontsize=14, fontweight='bold')
    
    # Set y-axis ticks and labels
    ax.set_yticks(range(n_keypoints))
    ax.set_yticklabels(joint_names, fontsize=9)
    
    # Set x-axis ticks (show every 10% of frames to avoid overcrowding)
    n_ticks = min(20, n_samples)
    x_tick_positions = np.linspace(0, n_samples - 1, n_ticks, dtype=int)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_positions, fontsize=8, rotation=45)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['Available', 'Unavailable'])
    cbar.set_label('Status', fontsize=10, fontweight='bold')
    
    # Calculate statistics first
    total_unavailable = np.sum(missing_per_landmark)
    total_cells = n_samples * n_keypoints
    unavailable_percentage = (total_unavailable / total_cells * 100) if total_cells > 0 else 0
    total_available = total_cells - total_unavailable
    available_percentage = 100 - unavailable_percentage
    
    # Add frame number annotations on unavailable cells
    # Find all unavailable positions (frame_idx, keypoint_idx)
    unavailable_frames_per_keypoint = {}  # {keypoint_idx: [frame_indices]}
    for frame_idx in range(n_samples):
        for kp_idx in range(n_keypoints):
            if missing_per_landmark[frame_idx, kp_idx]:
                if kp_idx not in unavailable_frames_per_keypoint:
                    unavailable_frames_per_keypoint[kp_idx] = []
                unavailable_frames_per_keypoint[kp_idx].append(frame_idx)
    
    # Add text annotations for unavailable frames
    # Use smaller font if there are many unavailable values to avoid overcrowding
    fontsize = 6 if total_unavailable > 1000 else (7 if total_unavailable > 500 else 8)
    
    for kp_idx, frame_indices in unavailable_frames_per_keypoint.items():
        for frame_idx in frame_indices:
            # In the transposed heatmap, x is frame_idx, y is kp_idx
            ax.text(frame_idx, kp_idx, str(frame_idx), 
                   ha='center', va='center', fontsize=fontsize, 
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='white', linewidth=0.5))
    
    # Add statistics text with per-keypoint availability info
    stats_text = f'Total available: {total_available}/{total_cells} ({available_percentage:.2f}%)\n'
    stats_text += f'Total unavailable: {total_unavailable}/{total_cells} ({unavailable_percentage:.2f}%)\n'
    stats_text += 'Data availability per keypoint:\n'
    
    # Calculate per-keypoint statistics
    valid_counts = np.sum(~missing_per_landmark, axis=0)
    nan_counts = np.sum(missing_per_landmark, axis=0)
    
    # Show top 5 keypoints with most unavailable data
    top_unavailable_indices = np.argsort(nan_counts)[::-1][:5]
    stats_text += 'Top 5 with most unavailable:\n'
    for idx in top_unavailable_indices:
        if nan_counts[idx] > 0:
            stats_text += f'  {joint_names[idx]}: {valid_counts[idx]} valid ({nan_counts[idx]} NaN, {100*valid_counts[idx]/n_samples:.1f}%)\n'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_availability_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save missing frames to CSV
    csv_file = os.path.join(output_dir, 'missing_landmarks.csv')
    missing_records = []
    
    # Create detailed records for each missing frame-keypoint pair
    for frame_idx in range(n_samples):
        for kp_idx in range(n_keypoints):
            if missing_per_landmark[frame_idx, kp_idx]:
                missing_records.append({
                    'Frame_Index': frame_idx,
                    'SampleID': sampleID[frame_idx] if frame_idx < len(sampleID) else frame_idx,
                    'Landmark': joint_names[kp_idx],
                    'Landmark_Index': kp_idx,
                    'Status': 'Unavailable'
                })
    
    # Also create a summary by frame
    frames_with_missing = {}
    for frame_idx in range(n_samples):
        missing_count = np.sum(missing_per_landmark[frame_idx, :])
        if missing_count > 0:
            missing_landmarks = [joint_names[kp_idx] for kp_idx in range(n_keypoints) 
                               if missing_per_landmark[frame_idx, kp_idx]]
            frames_with_missing[frame_idx] = {
                'Frame_Index': frame_idx,
                'SampleID': sampleID[frame_idx] if frame_idx < len(sampleID) else frame_idx,
                'Missing_Count': int(missing_count),
                'Missing_Landmarks': ', '.join(missing_landmarks)
            }
    
    # Save detailed CSV
    if len(missing_records) > 0:
        df_detailed = pd.DataFrame(missing_records)
        df_detailed.to_csv(csv_file, index=False)
        print(f"Missing landmarks detailed CSV saved to: {csv_file}")
        
        # Save summary CSV by frame
        summary_csv = os.path.join(output_dir, 'missing_frames_summary.csv')
        df_summary = pd.DataFrame(list(frames_with_missing.values()))
        df_summary = df_summary.sort_values('Frame_Index')
        df_summary.to_csv(summary_csv, index=False)
        print(f"Missing frames summary CSV saved to: {summary_csv}")
        
        # Print statistics about missing frames distribution
        print(f"\nMissing frames distribution:")
        print(f"  Total frames with missing data: {len(frames_with_missing)}/{n_samples}")
        if len(frames_with_missing) > 0:
            frame_indices = sorted(frames_with_missing.keys())
            print(f"  First frame with missing: {frame_indices[0]}")
            print(f"  Last frame with missing: {frame_indices[-1]}")
            print(f"  Frames with missing in first 100: {sum(1 for f in frame_indices if f < 100)}")
            print(f"  Frames with missing after 100: {sum(1 for f in frame_indices if f >= 100)}")
    else:
        print("No missing landmarks found - no CSV file created.")
    
    print(f"Landmark availability heatmap saved to: {os.path.join(output_dir, 'data_availability_heatmap.png')}")
    print(f"Available cells: {total_available}/{total_cells} ({available_percentage:.2f}%)")
    print(f"Unavailable cells: {total_unavailable}/{total_cells} ({unavailable_percentage:.2f}%)")

def save_large_error_records(errors, sampleID, joint_names, output_dir, threshold=18.0):
    """
    Save records of points with error greater than threshold to CSV.
    
    Args:
        errors: Error array with shape (n_samples, n_keypoints)
        sampleID: Sample IDs corresponding to each frame
        joint_names: List of joint/keypoint names
        output_dir: Output directory for saving the CSV
        threshold: Error threshold in mm (default 18.0)
    """
    n_samples, n_keypoints = errors.shape
    
    # Find all points with error > threshold (excluding NaN)
    large_error_records = []
    
    for frame_idx in range(n_samples):
        for kp_idx in range(n_keypoints):
            error_val = errors[frame_idx, kp_idx]
            # Check if error is valid (not NaN) and greater than threshold
            if not np.isnan(error_val) and error_val > threshold:
                large_error_records.append({
                    'Frame_Index': frame_idx,
                    'SampleID': sampleID[frame_idx] if frame_idx < len(sampleID) else frame_idx,
                    'BodyPart': joint_names[kp_idx],
                    'BodyPart_Index': kp_idx,
                    'Error_mm': error_val
                })
    
    # Save to CSV
    if len(large_error_records) > 0:
        csv_file = os.path.join(output_dir, f'large_errors_threshold_{threshold}mm.csv')
        df = pd.DataFrame(large_error_records)
        df = df.sort_values(['Frame_Index', 'Error_mm'], ascending=[True, False])
        df.to_csv(csv_file, index=False)
        
        print(f"\nLarge error records (>{threshold}mm) saved to: {csv_file}")
        print(f"Total points with error > {threshold}mm: {len(large_error_records)}")
        
        # Print statistics
        if len(large_error_records) > 0:
            unique_frames = df['Frame_Index'].nunique()
            print(f"  Unique frames with large errors: {unique_frames}/{n_samples}")
            print(f"  First frame with large error: {df['Frame_Index'].min()}")
            print(f"  Last frame with large error: {df['Frame_Index'].max()}")
            print(f"  Frames with large errors in first 100: {sum(1 for f in df['Frame_Index'].unique() if f < 100)}")
            print(f"  Frames with large errors after 100: {sum(1 for f in df['Frame_Index'].unique() if f >= 100)}")
            print(f"  Max error: {df['Error_mm'].max():.2f} mm")
            print(f"  Mean error (for large errors): {df['Error_mm'].mean():.2f} mm")
            
            # Count by body part
            bodypart_counts = df['BodyPart'].value_counts()
            print(f"\n  Large errors by body part:")
            for bodypart, count in bodypart_counts.items():
                print(f"    {bodypart}: {count} points")
    else:
        print(f"\nNo points with error > {threshold}mm found - no CSV file created.")

def save_evaluation_report(errors, joint_names, stats, output_dir):
    """Save detailed evaluation report to text file."""
    report_file = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DANNCE Prediction Evaluation Report\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write("Total samples evaluated: {}\n".format(errors.shape[0]))
        f.write("Total keypoints: {}\n".format(errors.shape[1]))
        f.write("\n")
        f.write("Overall Error Statistics:\n")
        f.write("  Mean:     {:.3f} mm\n".format(np.nanmean(errors)))
        f.write("  Median:   {:.3f} mm\n".format(np.nanmedian(errors)))
        f.write("  Std:      {:.3f} mm\n".format(np.nanstd(errors)))
        f.write("  Min:      {:.3f} mm\n".format(np.nanmin(errors)))
        f.write("  Max:      {:.3f} mm\n".format(np.nanmax(errors)))
        f.write("  Q25:      {:.3f} mm\n".format(np.nanpercentile(errors, 25)))
        f.write("  Q75:      {:.3f} mm\n".format(np.nanpercentile(errors, 75)))
        f.write("  P95:      {:.3f} mm\n".format(np.nanpercentile(errors, 95)))
        f.write("\n")
        
        # Success rates at different thresholds
        f.write("SUCCESS RATES\n")
        f.write("-"*70 + "\n")
        for thresh in [5, 10, 15, 20, 25]:
            success_rate = np.nanmean(errors < thresh) * 100
            f.write("  < {} mm: {:.2f}%\n".format(thresh, success_rate))
        f.write("\n")
        
        # Per-keypoint statistics
        f.write("PER-KEYPOINT STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write("{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}\n".format(
            "Keypoint", "Mean", "Median", "Std", "Q25", "Q75"))
        f.write("-"*70 + "\n")
        for i, name in enumerate(joint_names):
            f.write("{:<20} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}\n".format(
                name, stats['mean'][i], stats['median'][i], 
                stats['std'][i], stats['q25'][i], stats['q75'][i]))
        
        f.write("\n" + "="*70 + "\n")
    
    print("Evaluation report saved to: {}".format(report_file))

def main():
    if len(sys.argv) < 4:
        print("Usage: python evaluate_predictions.py [ground_truth_file] [predictions_file] [skeleton_file] [output_dir]")
        sys.exit(1)
    
    gt_file = sys.argv[1]
    pred_file = sys.argv[2]
    skeleton_file = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "./evaluation_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("DANNCE Prediction Evaluation")
    print("="*70)
    
    # Load data
    gt_3d, gt_sampleID, gt_data_frame = load_ground_truth(gt_file)
    pred_3d, pred_sampleID = load_predictions(pred_file)
    
    # Match samples
    gt_matched, pred_matched, matched_ids = match_samples(
        gt_3d, gt_sampleID, gt_data_frame, pred_3d, pred_sampleID
    )
    
    # Load skeleton for joint names first
    skeleton = sio.loadmat(skeleton_file)
    joint_names = skeleton['joint_names']
    if isinstance(joint_names, np.ndarray):
        if joint_names.dtype.names is not None:
            joint_names = [name[0] for name in joint_names[0]]
        else:
            joint_names = [name[0] if isinstance(name, np.ndarray) else name 
                          for name in np.squeeze(joint_names)]
    
    # Calculate errors
    print("\nCalculating Euclidean distances...")
    errors = calculate_euclidean_distance(gt_matched, pred_matched)
    
    # Ensure joint_names length matches
    if len(joint_names) != errors.shape[1]:
        print("Warning: Joint names count ({}) doesn't match keypoint count ({}). Using indices.".format(
            len(joint_names), errors.shape[1]))
        joint_names = ["Keypoint_{}".format(i+1) for i in range(errors.shape[1])]
    
    # Print statistics about NaN values
    nan_counts = np.sum(np.isnan(errors), axis=0)
    valid_counts = np.sum(~np.isnan(errors), axis=0)
    print("\nData availability per keypoint:")
    for i, name in enumerate(joint_names):
        print("  {}: {} valid samples ({} NaN, {:.1f}% valid)".format(
            name, valid_counts[i], nan_counts[i], 
            100 * valid_counts[i] / errors.shape[0] if errors.shape[0] > 0 else 0))
    
    # Save large error records to CSV
    print("\nSaving large error records (>18mm) to CSV...")
    save_large_error_records(errors, matched_ids, joint_names, output_dir, threshold=18.0)
    
    # Generate plots and report
    print("\nGenerating evaluation plots...")
    stats = plot_evaluation_results(errors, joint_names, output_dir)
    
    # Generate landmark tracking accuracy plot
    print("\nGenerating landmark tracking accuracy plot...")
    plot_landmark_tracking_accuracy(errors, output_dir, threshold=18.0)
    
    # Generate data availability heatmap
    print("\nGenerating data availability heatmap...")
    plot_missing_data_availability_heatmap(errors, matched_ids, output_dir, joint_names)

    print("Saving evaluation report...")
    save_evaluation_report(errors, joint_names, stats, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print("Overall Mean Error: {:.3f} mm".format(np.nanmean(errors)))
    print("Overall Median Error: {:.3f} mm".format(np.nanmedian(errors)))
    print("Overall Std Error: {:.3f} mm".format(np.nanstd(errors)))
    print("\nSuccess Rates:")
    for thresh in [5, 10, 20]:
        success_rate = np.nanmean(errors < thresh) * 100
        print("  < {} mm: {:.2f}%".format(thresh, success_rate))
    print("\nResults saved to: {}".format(output_dir))
    print("="*70)

if __name__ == "__main__":
    main()

