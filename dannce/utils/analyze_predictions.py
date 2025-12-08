"""
Comprehensive analysis script for DANNCE prediction results.

This script provides various metrics and visualizations to analyze prediction quality:
1. Per-keypoint error analysis
2. Prediction confidence (p_max) analysis
3. Temporal smoothness analysis
4. Spatial error distribution
5. Body part grouping analysis
6. Error statistics and distributions

Usage:
    python analyze_predictions.py [save_data_AVG.mat] [skeleton.mat] [output_dir]
    
Example:
    python analyze_predictions.py \
      ./DANNCE/predict_results/save_data_AVG.mat \
      ./configs/mouse22_skeleton.mat \
      ./analysis_results
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from collections import defaultdict

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_predictions(pred_file):
    """Load prediction results from .mat file."""
    data = sio.loadmat(pred_file)
    pred = data['pred']  # Shape: (n_samples, 3, n_keypoints)
    sampleID = np.squeeze(data['sampleID'])
    
    p_max = None
    if 'p_max' in data:
        p_max = data['p_max']  # Prediction confidence
    
    return pred, sampleID, p_max

def load_skeleton(skeleton_file):
    """Load skeleton structure."""
    data = sio.loadmat(skeleton_file)
    joint_names = data['joint_names']
    # Handle different formats
    if isinstance(joint_names, np.ndarray):
        if joint_names.dtype.names is not None:
            joint_names = [name[0] for name in joint_names[0]]
        else:
            joint_names = [name[0] if isinstance(name, np.ndarray) else name 
                          for name in np.squeeze(joint_names)]
    
    return joint_names

def calculate_per_keypoint_errors(pred, true=None):
    """
    Calculate error metrics per keypoint.
    If true labels are not available, calculate temporal smoothness instead.
    """
    n_samples, n_dims, n_keypoints = pred.shape
    
    if true is not None:
        # Calculate Euclidean distance per keypoint
        errors = np.sqrt(np.sum((pred - true)**2, axis=1))  # (n_samples, n_keypoints)
        return errors
    else:
        # Calculate temporal smoothness (velocity/acceleration)
        velocities = np.diff(pred, axis=0)  # (n_samples-1, 3, n_keypoints)
        speeds = np.sqrt(np.sum(velocities**2, axis=1))  # (n_samples-1, n_keypoints)
        return speeds

def plot_per_keypoint_errors(errors, joint_names, output_dir, title="Per-Keypoint Errors"):
    """Plot error distribution for each keypoint."""
    n_keypoints = len(joint_names)
    n_samples = errors.shape[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Prepare data for seaborn
    plot_data = []
    plot_labels = []
    for i in range(n_keypoints):
        plot_data.extend(errors[:, i])
        plot_labels.extend([joint_names[i]] * n_samples)
    
    df_plot = pd.DataFrame({'Error': plot_data, 'Keypoint': plot_labels})
    
    # Box plot using seaborn
    ax1 = axes[0]
    sns.boxplot(data=df_plot, x='Keypoint', y='Error', ax=ax1)
    ax1.set_ylabel('Error (mm)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('', fontsize=12)
    ax1.set_title('{} - Box Plot'.format(title), fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Mean error bar plot with error bars
    ax2 = axes[1]
    mean_errors = np.mean(errors, axis=0)
    std_errors = np.std(errors, axis=0)
    q25 = np.percentile(errors, 25, axis=0)
    q75 = np.percentile(errors, 75, axis=0)
    iqr = q75 - q25  # Interquartile Range
    x_pos = np.arange(n_keypoints)
    colors = sns.color_palette("husl", n_keypoints)
    
    bars = ax2.bar(x_pos, mean_errors, yerr=std_errors, alpha=0.8, 
                   capsize=5, error_kw={'elinewidth': 2, 'ecolor': 'darkred', 'capthick': 2})
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(joint_names, rotation=45, ha='right')
    ax2.set_ylabel('Mean Error (mm)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('', fontsize=12)
    ax2.set_title('Mean Error per Keypoint', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_keypoint_errors.png'), dpi=300)
    plt.close()
    
    # Save statistics
    stats = {
        'mean': mean_errors,
        'std': std_errors,
        'median': np.median(errors, axis=0),
        'min': np.min(errors, axis=0),
        'max': np.max(errors, axis=0),
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
    }
    
    return stats

def plot_confidence_analysis(p_max, output_dir, joint_names=None):
    """Analyze and plot prediction confidence."""
    if p_max is None:
        print("Warning: p_max not available in predictions")
        return
    
    n_samples, n_keypoints = p_max.shape
    
    # Use joint names if available, otherwise use indices
    if joint_names is None or len(joint_names) != n_keypoints:
        joint_names = [f"Keypoint_{i}" for i in range(n_keypoints)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall distribution using seaborn
    ax = axes[0, 0]
    # Auto-determine bins based on data range
    data_range = p_max.flatten()
    n_bins = min(50, int(np.sqrt(len(data_range))))
    sns.histplot(data_range, bins=n_bins, kde=True, ax=ax, 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
    mean_conf = np.mean(data_range)
    median_conf = np.median(data_range)
    q25_conf = np.percentile(data_range, 25)
    q75_conf = np.percentile(data_range, 75)
    iqr_conf = q75_conf - q25_conf
    ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
               label='Mean: {:.3f}'.format(mean_conf))
    ax.axvline(median_conf, color='green', linestyle='--', linewidth=2,
               label='Median: {:.3f}'.format(median_conf))
    ax.axvline(q25_conf, color='orange', linestyle=':', linewidth=2,
               label='Q25: {:.3f}'.format(q25_conf))
    ax.axvline(q75_conf, color='orange', linestyle=':', linewidth=2,
               label='Q75: {:.3f} (IQR={:.3f})'.format(q75_conf, iqr_conf))
    ax.set_xlabel('Confidence (p_max)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Mean confidence per keypoint
    ax = axes[0, 1]
    mean_conf = np.mean(p_max, axis=0)
    std_conf = np.std(p_max, axis=0)
    q25_conf = np.percentile(p_max, 25, axis=0)
    q75_conf = np.percentile(p_max, 75, axis=0)
    iqr_conf = q75_conf - q25_conf  # Interquartile Range
    x_pos = np.arange(n_keypoints)
    colors = sns.color_palette("husl", n_keypoints)
    
    bars = ax.bar(x_pos, mean_conf, yerr=std_conf, alpha=0.8, 
                  capsize=3, error_kw={'elinewidth': 1.5, 'ecolor': 'darkred', 'capthick': 1.5})
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(joint_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Confidence', fontsize=12, fontweight='bold')
    ax.set_xlabel('', fontsize=12)
    ax.set_title('Mean Confidence per Keypoint', fontsize=14, fontweight='bold')
    
    # Confidence over time using seaborn
    ax = axes[1, 0]
    mean_conf_time = np.mean(p_max, axis=1)
    time_df = pd.DataFrame({'Sample Index': np.arange(n_samples), 
                           'Mean Confidence': mean_conf_time})
    sns.lineplot(data=time_df, x='Sample Index', y='Mean Confidence', ax=ax, 
                linewidth=2, color='steelblue')
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Over Time', fontsize=14, fontweight='bold')
    
    # Low confidence samples
    ax = axes[1, 1]
    low_conf_threshold = np.percentile(p_max.flatten(), 25)  # Use 25th percentile as threshold
    low_conf_samples = np.sum(p_max < low_conf_threshold, axis=1)
    low_conf_df = pd.DataFrame({'Sample Index': np.arange(n_samples),
                                'Low Confidence Count': low_conf_samples})
    sns.lineplot(data=low_conf_df, x='Sample Index', y='Low Confidence Count', 
                ax=ax, linewidth=2, color='crimson')
    ax.axhline(y=n_keypoints * 0.1, color='orange', linestyle='--', linewidth=2,
               label='10% threshold')
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Low-Confidence Keypoints', fontsize=12, fontweight='bold')
    ax.set_title('Low Confidence Keypoints Over Time\n(Threshold: {:.3f})'.format(low_conf_threshold), 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), dpi=300)
    plt.show()
    plt.close()

def plot_temporal_smoothness(pred, output_dir):
    """Analyze temporal smoothness of predictions."""
    n_samples, n_dims, n_keypoints = pred.shape
    
    # Calculate velocities and accelerations
    velocities = np.diff(pred, axis=0)  # (n_samples-1, 3, n_keypoints)
    speeds = np.sqrt(np.sum(velocities**2, axis=1))  # (n_samples-1, n_keypoints)
    
    accelerations = np.diff(velocities, axis=0)  # (n_samples-2, 3, n_keypoints)
    accelerations_mag = np.sqrt(np.sum(accelerations**2, axis=1))  # (n_samples-2, n_keypoints)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Speed distribution using seaborn with adaptive bins
    ax = axes[0, 0]
    speed_data = speeds.flatten()
    n_bins = min(50, int(np.sqrt(len(speed_data))))
    # Remove outliers for better visualization (top 1%)
    p99 = np.percentile(speed_data, 99)
    speed_data_filtered = speed_data[speed_data <= p99]
    sns.histplot(speed_data_filtered, bins=n_bins, kde=True, ax=ax, 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
    mean_speed = np.mean(speed_data)
    median_speed = np.median(speed_data)
    q25_speed = np.percentile(speed_data, 25)
    q75_speed = np.percentile(speed_data, 75)
    iqr_speed = q75_speed - q25_speed
    ax.axvline(mean_speed, color='red', linestyle='--', linewidth=2,
               label='Mean: {:.2f}'.format(mean_speed))
    ax.axvline(median_speed, color='green', linestyle='--', linewidth=2,
               label='Median: {:.2f}'.format(median_speed))
    ax.axvline(q25_speed, color='orange', linestyle=':', linewidth=2,
               label='Q25: {:.2f}'.format(q25_speed))
    ax.axvline(q75_speed, color='orange', linestyle=':', linewidth=2,
               label='Q75: {:.2f} (IQR={:.2f})'.format(q75_speed, iqr_speed))
    ax.set_xlabel('Speed (mm/frame)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Speed Distribution (filtered to 99th percentile)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Acceleration distribution with adaptive bins
    ax = axes[0, 1]
    accel_data = accelerations_mag.flatten()
    n_bins = min(50, int(np.sqrt(len(accel_data))))
    # Remove outliers for better visualization (top 1%)
    p99 = np.percentile(accel_data, 99)
    accel_data_filtered = accel_data[accel_data <= p99]
    sns.histplot(accel_data_filtered, bins=n_bins, kde=True, ax=ax, 
                color='orange', alpha=0.7, edgecolor='black', linewidth=1.2)
    mean_accel = np.mean(accel_data)
    median_accel = np.median(accel_data)
    q25_accel = np.percentile(accel_data, 25)
    q75_accel = np.percentile(accel_data, 75)
    iqr_accel = q75_accel - q25_accel
    ax.axvline(mean_accel, color='red', linestyle='--', linewidth=2,
               label='Mean: {:.2f}'.format(mean_accel))
    ax.axvline(median_accel, color='green', linestyle='--', linewidth=2,
               label='Median: {:.2f}'.format(median_accel))
    ax.axvline(q25_accel, color='purple', linestyle=':', linewidth=2,
               label='Q25: {:.2f}'.format(q25_accel))
    ax.axvline(q75_accel, color='purple', linestyle=':', linewidth=2,
               label='Q75: {:.2f} (IQR={:.2f})'.format(q75_accel, iqr_accel))
    ax.set_xlabel('Acceleration (mm/frame²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Acceleration Distribution (filtered to 99th percentile)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Speed over time (mean across keypoints) using seaborn
    ax = axes[1, 0]
    mean_speed = np.mean(speeds, axis=1)
    speed_df = pd.DataFrame({'Sample Index': np.arange(len(mean_speed)), 
                            'Mean Speed (mm/frame)': mean_speed})
    sns.lineplot(data=speed_df, x='Sample Index', y='Mean Speed (mm/frame)', ax=ax, 
                linewidth=2, color='steelblue')
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Speed (mm/frame)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Speed Over Time', fontsize=14, fontweight='bold')
    
    # Acceleration over time using seaborn
    ax = axes[1, 1]
    mean_accel = np.mean(accelerations_mag, axis=1)
    accel_df = pd.DataFrame({'Sample Index': np.arange(len(mean_accel)), 
                            'Mean Acceleration (mm/frame²)': mean_accel})
    sns.lineplot(data=accel_df, x='Sample Index', y='Mean Acceleration (mm/frame²)', ax=ax, 
                linewidth=2, color='orange')
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Acceleration (mm/frame²)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Acceleration Over Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_smoothness.png'), dpi=300)
    plt.show()
    plt.close()
    
    return speeds, accelerations_mag

def plot_spatial_error_distribution(pred, output_dir, joint_names=None):
    """Analyze spatial distribution of predictions."""
    n_samples, n_dims, n_keypoints = pred.shape
    
    # Use joint names if available
    if joint_names is None or len(joint_names) != n_keypoints:
        joint_names = [f"Keypoint_{i}" for i in range(n_keypoints)]
    
    # Calculate center of mass trajectory
    com = np.mean(pred, axis=2)  # (n_samples, 3)
    
    # Calculate distances from COM for each keypoint
    distances_from_com = np.sqrt(np.sum((pred - com[:, :, np.newaxis])**2, axis=1))
    # (n_samples, n_keypoints)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # COM trajectory (3D projection) using seaborn
    ax = axes[0, 0]
    com_df = pd.DataFrame({'X (mm)': com[:, 0], 'Y (mm)': com[:, 1]})
    sns.scatterplot(data=com_df, x='X (mm)', y='Y (mm)', ax=ax, 
                   alpha=0.6, s=20, color='steelblue')
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Center of Mass Trajectory (XY projection)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    
    # Distance from COM distribution using seaborn with adaptive bins
    ax = axes[0, 1]
    dist_data = distances_from_com.flatten()
    n_bins = min(50, int(np.sqrt(len(dist_data))))
    sns.histplot(dist_data, bins=n_bins, kde=True, ax=ax, 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
    mean_dist = np.mean(dist_data)
    median_dist = np.median(dist_data)
    q25_dist = np.percentile(dist_data, 25)
    q75_dist = np.percentile(dist_data, 75)
    iqr_dist = q75_dist - q25_dist
    ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2,
               label='Mean: {:.2f}'.format(mean_dist))
    ax.axvline(median_dist, color='green', linestyle='--', linewidth=2,
               label='Median: {:.2f}'.format(median_dist))
    ax.axvline(q25_dist, color='orange', linestyle=':', linewidth=2,
               label='Q25: {:.2f}'.format(q25_dist))
    ax.axvline(q75_dist, color='orange', linestyle=':', linewidth=2,
               label='Q75: {:.2f} (IQR={:.2f})'.format(q75_dist, iqr_dist))
    ax.set_xlabel('Distance from COM (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distance from COM Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Mean distance from COM per keypoint
    ax = axes[1, 0]
    mean_dist = np.mean(distances_from_com, axis=0)
    std_dist = np.std(distances_from_com, axis=0)
    q25_dist = np.percentile(distances_from_com, 25, axis=0)
    q75_dist = np.percentile(distances_from_com, 75, axis=0)
    iqr_dist = q75_dist - q25_dist  # Interquartile Range
    x_pos = np.arange(n_keypoints)
    colors = sns.color_palette("husl", n_keypoints)
    
    bars = ax.bar(x_pos, mean_dist, yerr=std_dist, alpha=0.8, 
                  capsize=3, error_kw={'elinewidth': 1.5, 'ecolor': 'darkred', 'capthick': 1.5})
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(joint_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Distance from COM (mm)', fontsize=12, fontweight='bold')
    ax.set_xlabel('', fontsize=12)
    ax.set_title('Mean Distance from COM per Keypoint', fontsize=14, fontweight='bold')
    
    # 3D scatter of keypoint positions (sample) using seaborn
    ax = axes[1, 1]
    sample_idx = n_samples // 2
    sample_pred = pred[sample_idx]  # (3, n_keypoints)
    keypoint_df = pd.DataFrame({'X (mm)': sample_pred[0], 'Y (mm)': sample_pred[1]})
    sns.scatterplot(data=keypoint_df, x='X (mm)', y='Y (mm)', ax=ax, 
                   s=100, alpha=0.7, color='steelblue')
    ax.scatter(com[sample_idx, 0], com[sample_idx, 1], s=300, 
              marker='*', color='crimson', label='COM', edgecolors='white', linewidths=2)
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Keypoint Positions (Sample {})'.format(sample_idx), fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_distribution.png'), dpi=300)
    plt.show()
    plt.close()

def check_and_visualize_nan(pred, sampleID, output_dir, joint_names=None):
    """
    Check for NaN values in predictions and visualize their locations.
    
    Args:
        pred: Prediction array with shape (n_samples, 3, n_keypoints)
        sampleID: Sample IDs corresponding to each frame
        output_dir: Output directory for saving visualizations
        joint_names: List of joint/keypoint names
    """
    n_samples, n_dims, n_keypoints = pred.shape
    
    # Use joint names if available
    if joint_names is None or len(joint_names) != n_keypoints:
        joint_names = [f"Keypoint_{i}" for i in range(n_keypoints)]
    
    # Check for NaN values
    nan_mask = np.isnan(pred)  # Shape: (n_samples, 3, n_keypoints)
    
    # Count NaN values
    total_nans = np.sum(nan_mask)
    nan_per_frame = np.sum(nan_mask, axis=(1, 2))  # NaN count per frame
    nan_per_keypoint = np.sum(nan_mask, axis=(0, 1))  # NaN count per keypoint
    nan_per_coord = np.sum(nan_mask, axis=(0, 2))  # NaN count per coordinate (x, y, z)
    
    # Find specific locations of NaN values
    nan_locations = []
    for frame_idx in range(n_samples):
        for coord_idx in range(n_dims):
            for kp_idx in range(n_keypoints):
                if nan_mask[frame_idx, coord_idx, kp_idx]:
                    coord_name = ['X', 'Y', 'Z'][coord_idx]
                    nan_locations.append({
                        'frame': frame_idx,
                        'sampleID': sampleID[frame_idx] if frame_idx < len(sampleID) else frame_idx,
                        'keypoint': joint_names[kp_idx],
                        'keypoint_idx': kp_idx,
                        'coordinate': coord_name,
                        'coord_idx': coord_idx
                    })
    
    # Print summary
    print("\n" + "="*60)
    print("NaN Value Detection Summary")
    print("="*60)
    print(f"Total NaN values: {total_nans}")
    print(f"Frames with NaN: {np.sum(nan_per_frame > 0)} / {n_samples}")
    print(f"Keypoints with NaN: {np.sum(nan_per_keypoint > 0)} / {n_keypoints}")
    print(f"Coordinates with NaN: X={nan_per_coord[0]}, Y={nan_per_coord[1]}, Z={nan_per_coord[2]}")
    
    if total_nans > 0:
        print(f"\nNaN locations found: {len(nan_locations)}")
        print("\nFirst 20 NaN locations:")
        for i, loc in enumerate(nan_locations[:20]):
            print(f"  Frame {loc['frame']} (sampleID: {loc['sampleID']}), "
                  f"{loc['keypoint']}, {loc['coordinate']} coordinate")
        if len(nan_locations) > 20:
            print(f"  ... and {len(nan_locations) - 20} more")
    else:
        print("\n✓ No NaN values found in predictions!")
        return
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Heatmap: NaN per frame and keypoint
    ax1 = fig.add_subplot(gs[0, 0])
    nan_per_frame_keypoint = np.sum(nan_mask, axis=1)  # (n_samples, n_keypoints)
    if np.sum(nan_per_frame_keypoint) > 0:
        im1 = ax1.imshow(nan_per_frame_keypoint.T, aspect='auto', cmap='Reds', 
                        interpolation='nearest', origin='lower')
        ax1.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Keypoint', fontsize=11, fontweight='bold')
        ax1.set_title('NaN Count per Frame-Keypoint\n(Sum across X, Y, Z)', 
                     fontsize=12, fontweight='bold')
        ax1.set_yticks(range(n_keypoints))
        ax1.set_yticklabels(joint_names, fontsize=8)
        plt.colorbar(im1, ax=ax1, label='NaN Count')
    else:
        ax1.text(0.5, 0.5, 'No NaN values', ha='center', va='center', 
                fontsize=14, transform=ax1.transAxes)
        ax1.set_title('NaN Count per Frame-Keypoint', fontsize=12, fontweight='bold')
    
    # 2. Heatmap: NaN per coordinate and keypoint
    ax2 = fig.add_subplot(gs[0, 1])
    nan_per_coord_keypoint = np.sum(nan_mask, axis=0)  # (3, n_keypoints)
    if np.sum(nan_per_coord_keypoint) > 0:
        im2 = ax2.imshow(nan_per_coord_keypoint, aspect='auto', cmap='Reds',
                        interpolation='nearest', origin='lower')
        ax2.set_xlabel('Keypoint', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Coordinate', fontsize=11, fontweight='bold')
        ax2.set_title('NaN Count per Coordinate-Keypoint\n(Sum across all frames)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(range(n_keypoints))
        ax2.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=8)
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(['X', 'Y', 'Z'], fontsize=10)
        plt.colorbar(im2, ax=ax2, label='NaN Count')
    else:
        ax2.text(0.5, 0.5, 'No NaN values', ha='center', va='center',
                fontsize=14, transform=ax2.transAxes)
        ax2.set_title('NaN Count per Coordinate-Keypoint', fontsize=12, fontweight='bold')
    
    # 3. Bar plot: NaN count per keypoint
    ax3 = fig.add_subplot(gs[0, 2])
    if np.sum(nan_per_keypoint) > 0:
        colors = ['red' if count > 0 else 'gray' for count in nan_per_keypoint]
        bars = ax3.bar(range(n_keypoints), nan_per_keypoint, color=colors, alpha=0.7)
        ax3.set_xlabel('Keypoint', fontsize=11, fontweight='bold')
        ax3.set_ylabel('NaN Count', fontsize=11, fontweight='bold')
        ax3.set_title('Total NaN Count per Keypoint\n(All frames and coordinates)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xticks(range(n_keypoints))
        ax3.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=8)
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, nan_per_keypoint)):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{int(count)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No NaN values', ha='center', va='center',
                fontsize=14, transform=ax3.transAxes)
        ax3.set_title('Total NaN Count per Keypoint', fontsize=12, fontweight='bold')
    
    # 4. Line plot: NaN count over time (frames)
    ax4 = fig.add_subplot(gs[1, 0])
    if np.sum(nan_per_frame) > 0:
        ax4.plot(range(n_samples), nan_per_frame, 'r-', linewidth=2, marker='o', markersize=3)
        ax4.fill_between(range(n_samples), nan_per_frame, alpha=0.3, color='red')
        ax4.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
        ax4.set_ylabel('NaN Count per Frame', fontsize=11, fontweight='bold')
        ax4.set_title('NaN Count Over Time\n(All keypoints and coordinates)', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        # Highlight frames with NaN
        nan_frames = np.where(nan_per_frame > 0)[0]
        if len(nan_frames) > 0:
            ax4.scatter(nan_frames, nan_per_frame[nan_frames], 
                       color='darkred', s=50, zorder=5, label='Frames with NaN')
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No NaN values', ha='center', va='center',
                fontsize=14, transform=ax4.transAxes)
        ax4.set_title('NaN Count Over Time', fontsize=12, fontweight='bold')
    
    # 5. Bar plot: NaN count per coordinate
    ax5 = fig.add_subplot(gs[1, 1])
    coord_names = ['X', 'Y', 'Z']
    colors_coord = ['red' if count > 0 else 'gray' for count in nan_per_coord]
    bars = ax5.bar(coord_names, nan_per_coord, color=colors_coord, alpha=0.7)
    ax5.set_xlabel('Coordinate', fontsize=11, fontweight='bold')
    ax5.set_ylabel('NaN Count', fontsize=11, fontweight='bold')
    ax5.set_title('Total NaN Count per Coordinate\n(All frames and keypoints)', 
                 fontsize=12, fontweight='bold')
    for bar, count in zip(bars, nan_per_coord):
        if count > 0:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Scatter plot: NaN locations in frame-keypoint space
    ax6 = fig.add_subplot(gs[1, 2])
    if len(nan_locations) > 0:
        frame_indices = [loc['frame'] for loc in nan_locations]
        kp_indices = [loc['keypoint_idx'] for loc in nan_locations]
        coord_indices = [loc['coord_idx'] for loc in nan_locations]
        
        # Color by coordinate
        colors_map = {0: 'red', 1: 'green', 2: 'blue'}
        scatter_colors = [colors_map[idx] for idx in coord_indices]
        
        scatter = ax6.scatter(frame_indices, kp_indices, c=scatter_colors, 
                            alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        ax6.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Keypoint Index', fontsize=11, fontweight='bold')
        ax6.set_title('NaN Locations in Frame-Keypoint Space\n(Color by coordinate)', 
                     fontsize=12, fontweight='bold')
        ax6.set_yticks(range(n_keypoints))
        ax6.set_yticklabels(joint_names, fontsize=8)
        ax6.grid(True, alpha=0.3)
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='X coordinate'),
                          Patch(facecolor='green', label='Y coordinate'),
                          Patch(facecolor='blue', label='Z coordinate')]
        ax6.legend(handles=legend_elements, fontsize=9)
    else:
        ax6.text(0.5, 0.5, 'No NaN values', ha='center', va='center',
                fontsize=14, transform=ax6.transAxes)
        ax6.set_title('NaN Locations in Frame-Keypoint Space', fontsize=12, fontweight='bold')
    
    # 7. Detailed table view (if not too many)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    if len(nan_locations) <= 100:
        # Create a table
        table_data = []
        for loc in nan_locations:
            table_data.append([
                loc['frame'],
                loc['sampleID'],
                loc['keypoint'],
                loc['coordinate']
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Frame', 'SampleID', 'Keypoint', 'Coordinate'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style cells with NaN
        for i in range(1, len(table_data) + 1):
            for j in range(4):
                table[(i, j)].set_facecolor('#f1f1f2')
        
        ax7.set_title('Detailed NaN Locations (First 100)', 
                     fontsize=12, fontweight='bold', pad=20)
    else:
        # Show summary statistics instead
        summary_text = f"""
        NaN Detection Summary
        
        Total NaN values: {total_nans}
        Unique frames with NaN: {np.sum(nan_per_frame > 0)} / {n_samples}
        Unique keypoints with NaN: {np.sum(nan_per_keypoint > 0)} / {n_keypoints}
        
        Top 10 keypoints with most NaN values:
        """
        top_kp_indices = np.argsort(nan_per_keypoint)[::-1][:10]
        for idx in top_kp_indices:
            if nan_per_keypoint[idx] > 0:
                summary_text += f"\n  {joint_names[idx]}: {int(nan_per_keypoint[idx])} NaN values"
        
        summary_text += f"\n\nTop 10 frames with most NaN values:"
        top_frame_indices = np.argsort(nan_per_frame)[::-1][:10]
        for idx in top_frame_indices:
            if nan_per_frame[idx] > 0:
                summary_text += f"\n  Frame {idx} (sampleID: {sampleID[idx] if idx < len(sampleID) else idx}): {int(nan_per_frame[idx])} NaN values"
        
        summary_text += f"\n\n(Showing summary - {len(nan_locations)} total NaN locations found)"
        
        ax7.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax7.set_title('NaN Detection Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('NaN Value Detection and Visualization', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(os.path.join(output_dir, 'nan_detection.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Save NaN locations to file
    if len(nan_locations) > 0:
        nan_file = os.path.join(output_dir, 'nan_locations.txt')
        with open(nan_file, 'w') as f:
            f.write("NaN Locations in Predictions\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total NaN values: {total_nans}\n")
            f.write(f"Frames with NaN: {np.sum(nan_per_frame > 0)} / {n_samples}\n")
            f.write(f"Keypoints with NaN: {np.sum(nan_per_keypoint > 0)} / {n_keypoints}\n\n")
            f.write("Detailed NaN Locations:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Frame':<8} {'SampleID':<12} {'Keypoint':<20} {'Coordinate':<10}\n")
            f.write("-"*60 + "\n")
            for loc in nan_locations:
                f.write(f"{loc['frame']:<8} {loc['sampleID']:<12} {loc['keypoint']:<20} {loc['coordinate']:<10}\n")
        print(f"\nNaN locations saved to: {nan_file}")

def plot_missing_landmarks_heatmap(pred, sampleID, output_dir, joint_names=None):
    """
    Plot a heatmap showing missing landmarks across all frames.
    
    Args:
        pred: Prediction array with shape (n_samples, 3, n_keypoints)
        sampleID: Sample IDs corresponding to each frame
        output_dir: Output directory for saving the plot
        joint_names: List of joint/keypoint names
    """
    n_samples, n_dims, n_keypoints = pred.shape
    
    # Use joint names if available
    if joint_names is None or len(joint_names) != n_keypoints:
        joint_names = [f"Keypoint_{i}" for i in range(n_keypoints)]
    
    # Check for NaN values in any coordinate (X, Y, or Z)
    # If any coordinate is NaN, mark the landmark as missing
    nan_mask = np.isnan(pred)  # Shape: (n_samples, 3, n_keypoints)
    missing_per_landmark = np.any(nan_mask, axis=1)  # (n_samples, n_keypoints)
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
    ax.set_title('Landmark Availability Across All Frames\n(Red: Unavailable, Light Blue: Available)', 
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
    missing_frames_per_keypoint = {}  # {keypoint_idx: [frame_indices]}
    for frame_idx in range(n_samples):
        for kp_idx in range(n_keypoints):
            if missing_per_landmark[frame_idx, kp_idx]:
                if kp_idx not in missing_frames_per_keypoint:
                    missing_frames_per_keypoint[kp_idx] = []
                missing_frames_per_keypoint[kp_idx].append(frame_idx)
    
    # Add text annotations for unavailable frames
    # Use smaller font if there are many unavailable values to avoid overcrowding
    fontsize = 6 if total_unavailable > 1000 else (7 if total_unavailable > 500 else 8)
    
    for kp_idx, frame_indices in missing_frames_per_keypoint.items():
        for frame_idx in frame_indices:
            # In the transposed heatmap, x is frame_idx, y is kp_idx
            ax.text(frame_idx, kp_idx, str(frame_idx), 
                   ha='center', va='center', fontsize=fontsize, 
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='white', linewidth=0.5))
    
    # Add statistics text
    stats_text = f'Total available: {total_available}/{total_cells} ({available_percentage:.2f}%)\n'
    stats_text += f'Total unavailable: {total_unavailable}/{total_cells} ({unavailable_percentage:.2f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_landmarks_heatmap.png'), dpi=300, bbox_inches='tight')
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
    
    print(f"Landmark availability heatmap saved to: {os.path.join(output_dir, 'missing_landmarks_heatmap.png')}")
    print(f"Available cells: {total_available}/{total_cells} ({available_percentage:.2f}%)")
    print(f"Unavailable cells: {total_unavailable}/{total_cells} ({unavailable_percentage:.2f}%)")

def plot_error_statistics(errors, output_dir):
    """Plot overall error statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall error distribution using seaborn with adaptive bins
    ax = axes[0, 0]
    error_data = errors.flatten()
    n_bins = min(50, int(np.sqrt(len(error_data))))
    # Remove extreme outliers for better visualization (top 1%)
    p99 = np.percentile(error_data, 99)
    error_data_filtered = error_data[error_data <= p99]
    sns.histplot(error_data_filtered, bins=n_bins, kde=True, ax=ax, 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
    mean_err = np.mean(error_data)
    median_err = np.median(error_data)
    q25_err = np.percentile(error_data, 25)
    q75_err = np.percentile(error_data, 75)
    iqr_err = q75_err - q25_err
    ax.axvline(mean_err, color='red', linestyle='--', linewidth=2,
               label='Mean: {:.2f} mm'.format(mean_err))
    ax.axvline(median_err, color='green', linestyle='--', linewidth=2,
               label='Median: {:.2f} mm'.format(median_err))
    ax.axvline(q25_err, color='orange', linestyle=':', linewidth=2,
               label='Q25: {:.2f} mm'.format(q25_err))
    ax.axvline(q75_err, color='orange', linestyle=':', linewidth=2,
               label='Q75: {:.2f} mm (IQR={:.2f} mm)'.format(q75_err, iqr_err))
    ax.set_xlabel('Error (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Overall Error Distribution (filtered to 99th percentile)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Cumulative error distribution using seaborn
    ax = axes[0, 1]
    sorted_errors = np.sort(error_data)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    cumul_df = pd.DataFrame({'Error (mm)': sorted_errors, 'Cumulative Probability': cumulative})
    sns.lineplot(data=cumul_df, x='Error (mm)', y='Cumulative Probability', ax=ax, 
                linewidth=2, color='steelblue')
    ax.set_xlabel('Error (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
    
    # Error over time (mean) using seaborn
    ax = axes[1, 0]
    mean_error_time = np.mean(errors, axis=1)
    error_time_df = pd.DataFrame({'Sample Index': np.arange(len(mean_error_time)), 
                                 'Mean Error (mm)': mean_error_time})
    sns.lineplot(data=error_time_df, x='Sample Index', y='Mean Error (mm)', ax=ax, 
                linewidth=2, color='steelblue')
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Error (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Error Over Time', fontsize=14, fontweight='bold')
    
    # Error statistics summary with better formatting
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = """
    Error Statistics Summary
    
    Mean:     {:.2f} mm
    Median:   {:.2f} mm
    Std:      {:.2f} mm
    Min:      {:.2f} mm
    Max:      {:.2f} mm
    Q25:      {:.2f} mm
    Q75:      {:.2f} mm
    IQR:      {:.2f} mm
    
    Samples:  {}
    Keypoints: {}
    """.format(
        np.mean(error_data),
        np.median(error_data),
        np.std(error_data),
        np.min(error_data),
        np.max(error_data),
        np.percentile(error_data, 25),
        np.percentile(error_data, 75),
        np.percentile(error_data, 75) - np.percentile(error_data, 25),
        errors.shape[0],
        errors.shape[1]
    )
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_statistics.png'), dpi=300)
    plt.show()
    plt.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_predictions.py [pred_file] [skeleton_file] [output_dir]")
        sys.exit(1)
    
    pred_file = sys.argv[1]
    skeleton_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./analysis_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading predictions from: {}".format(pred_file))
    pred, sampleID, p_max = load_predictions(pred_file)
    
    print("Loading skeleton from: {}".format(skeleton_file))
    joint_names = load_skeleton(skeleton_file)
    
    print("Prediction shape: {}".format(pred.shape))
    print("Number of keypoints: {}".format(len(joint_names)))
    
    # Check for NaN values first
    print("\nChecking for NaN values...")
    check_and_visualize_nan(pred, sampleID, output_dir, joint_names)
    
    # Plot missing landmarks heatmap
    print("\nGenerating missing landmarks heatmap...")
    plot_missing_landmarks_heatmap(pred, sampleID, output_dir, joint_names)
    
    # Calculate temporal smoothness (since we don't have ground truth)
    print("\nCalculating temporal smoothness...")
    speeds = calculate_per_keypoint_errors(pred, true=None)
    
    # Plot analyses
    print("\nGenerating visualizations...")
    
    # 1. Per-keypoint analysis (using speeds as proxy for quality)
    print("  - Per-keypoint analysis...")
    stats = plot_per_keypoint_errors(speeds, joint_names, output_dir, 
                                     title="Per-Keypoint Speed (Temporal Smoothness)")
    
    # 2. Confidence analysis
    if p_max is not None:
        print("  - Confidence analysis...")
        plot_confidence_analysis(p_max, output_dir, joint_names)
    
    # 3. Temporal smoothness
    print("  - Temporal smoothness analysis...")
    plot_temporal_smoothness(pred, output_dir)
    
    # 4. Spatial distribution
    print("  - Spatial distribution analysis...")
    plot_spatial_error_distribution(pred, output_dir, joint_names)
    
    # 5. Overall statistics
    print("  - Error statistics...")
    plot_error_statistics(speeds, output_dir)
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Prediction Analysis Statistics\n")
        f.write("="*50 + "\n\n")
        f.write("Overall Statistics:\n")
        f.write("  Mean speed: {:.2f} mm/frame\n".format(np.mean(speeds)))
        f.write("  Median speed: {:.2f} mm/frame\n".format(np.median(speeds)))
        f.write("  Std speed: {:.2f} mm/frame\n".format(np.std(speeds)))
        f.write("\nPer-Keypoint Statistics:\n")
        for i, name in enumerate(joint_names):
            f.write("  {}: mean={:.2f}, std={:.2f}, median={:.2f}\n".format(
                name, stats['mean'][i], stats['std'][i], stats['median'][i]))
    
    print("\nAnalysis complete! Results saved to: {}".format(output_dir))
    print("Generated files:")
    print("  - nan_detection.png (NaN value visualization)")
    if np.any(np.isnan(pred)):
        print("  - nan_locations.txt (Detailed NaN locations)")
    print("  - per_keypoint_errors.png")
    if p_max is not None:
        print("  - confidence_analysis.png")
    print("  - temporal_smoothness.png")
    print("  - spatial_distribution.png")
    print("  - error_statistics.png")
    print("  - statistics.txt")

if __name__ == "__main__":
    main()

