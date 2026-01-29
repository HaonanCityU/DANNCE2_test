"""
Example script for visualizing Parkinson DB data. 3D joint predictions are projected into each
camera view, plotted into a figure, and written into a video.

To load the correct video file, the input mocap data structure filename must contain the subject
number and recording day number as 'mocap-s{subject#}-d{day#}.mat'

Usage:
    python plot2DProjection2.py [path_to_label3d_dannce.mat] [path_to_save_data_AVG.mat] [path_to_video_file]
                               [path_to_skeleton.mat] [path_to_save_video] [start_sample (int)] [max_samples (int)]
    Optional: [fps (int)] [com3d_file] [cam_index (int)]
    Use "none" or "-" for com3d_file to skip COM plotting when passing cam_index.
    cam_index: 0-based (0=Camera1, 1=Camera2, ...). Must match the video. Default: 0.
"""

import numpy as np
import scipy.io as sio
import os
import sys
import imageio

import dannce.engine.ops as dops
import dannce.engine.io as dio

  
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter

dannceMat_filepath = sys.argv[1]
preditcions_filepath = sys.argv[2]
videofle_path = sys.argv[3]
skeleton_path = sys.argv[4]
video_save_path = sys.argv[5]
start_sample = int(sys.argv[6])
max_samples = int(sys.argv[7])

if len(sys.argv) > 8:
  fps_setting = int(sys.argv[8])
else:
  fps_setting = 30

if len(sys.argv) > 9:
  _arg9 = sys.argv[9].strip().lower()
  if _arg9 in ("none", "-", ""):
    com3d_file = None
    print("Com3d file not specified. COM will not be plotted")
  else:
    com3d_file = sys.argv[9]
    print("Com3d file specified. COM will be plotted")
else:
  com3d_file = None
  print("Com3d file not specified. COM will not be plotted")

if len(sys.argv) > 10:
  cam_index = int(sys.argv[10])
  print("Using cam_index = {} (Camera{}).".format(cam_index, cam_index + 1))
else:
  cam_index = 0
  print("Using default cam_index = 0 (Camera1). Pass 10th arg for Camera2, etc.")

COLOR_DICT = [
        (1.0000,    0,              0,    0.5000),
        (0.9565,    1.0000,         0,    0.5000),
        (1.0000,    0.5217,         0,    0.5000),
        (0.6957,    1.0000,         0,    0.5000),
        (1.0000,    0.2609,         0,    0.5000),
        (1.0000,    0.7826,         0,    0.5000),
        (0.4348,    1.0000,         0,    0.5000),
        (0.1739,    1.0000,         0,    0.5000),
        (0,         1.0000,    0.0870,    0.5000),
        (0,         1.0000,    0.3478,    0.5000),
        (0,         1.0000,    0.6087,    0.5000),
        (0,         1.0000,    0.8696,    0.5000),
        (0,         0.6087,    1.0000,    0.5000),
        (0,         0.8696,    1.0000,    0.5000),
        (0,         0.3478,    1.0000,    0.5000),
        (0.4348,         0,    1.0000,    0.5000),
        (0.1739,         0,    1.0000,    0.5000),
        (0.6957,         0,    1.0000,    0.5000),
        (1.0000,         0,    0.7826,    0.5000),
        (0.9565,         0,    1.0000,    0.5000),
        (1.0000,         0,    0.5217,    0.5000),
        (1.0000,         0,    0.2609,    0.5000),
    ]

"""### Load Data"""

def get_data(dannceMat_filepath: str, preditcions_filepath: str, skeleton_path: str, com3d_filepath = None):
  """
    # Function to load data from different files, read them into suitable data structures and return. 
    # Entry point where validity of file paths are checked.

    dannceMat_filepath: label3d_dannce.mat file path
    preditcions_filepath: save_data_AVG.mat file path
    skeleton_path: file path for skeletons file
    com3d_filepath: com3d_used.mat filepath
  """
  if dannceMat_filepath == None or preditcions_filepath == None or skeleton_path == None:
    print("One or more file paths is missing. Please provide all file paths.")
    raise ValueError("Missing required file paths")
  elif os.path.exists(dannceMat_filepath) and os.path.exists(preditcions_filepath) and os.path.exists(skeleton_path) :
    cam_names = dio.load_camnames(dannceMat_filepath)
    sync = dio.load_sync(dannceMat_filepath)
    params = dio.load_camera_params(dannceMat_filepath)
    
    skeleton = sio.loadmat(skeleton_path)
    skeleton = {k:v for k, v in skeleton.items() if k[0] != '_'}
    skeleton["joint_names"] = skeleton["joint_names"][0]
    
    predictions = sio.loadmat(preditcions_filepath)
    predictions = {k:v for k, v in predictions.items() if k[0] != '_'}

    if com3d_filepath is not None and os.path.exists(com3d_filepath) :
      # com_3d = sio.loadmat(os.path.join(preditcions_filepath, 'com3d_used.mat'))['com']
      com_3d = sio.loadmat(com3d_filepath)['com']
    else:
      print ("No filepath specified for com_3d. Returning None.")
      com_3d = None
      
    return cam_names, sync, params, skeleton, predictions, com_3d
  else:
    print("Enter valid os path for dannceMat, predictions and skeleton files")
    print(f"  dannceMat_filepath exists: {os.path.exists(dannceMat_filepath)} - {dannceMat_filepath}")
    print(f"  preditcions_filepath exists: {os.path.exists(preditcions_filepath)} - {preditcions_filepath}")
    print(f"  skeleton_path exists: {os.path.exists(skeleton_path)} - {skeleton_path}")
    raise FileNotFoundError("One or more required files do not exist. Please check the file paths above.")

def get_camParams(params, skeleton, exclude_joints):
  """
    # Method to extract camera params and links from params and skeleton 

    params: dict of params read from .mat file or otherwise
    skeleton: dict of fields from the skeleton file read from .mat file or otherwise. 
              Contains joint names and their indices.
    exclude_joints: integer list of joints to exclude from plotting
  """
  cameraParams = []
  rot = []
  trans = []
  mirror = []

  for i in range(len(params)):
    cameraParams.append({'IntrinsicMatrix': params[i]["K"],
                        'RadialDistortion': params[i]["RDistort"],
                        'TangentialDistortion': params[i]["TDistort"]})
    rot.append(params[i]['r'])
    trans.append(params[i]['t'])
    if 'm' in params[i].keys():
      mirror.append(params[i]['m'])
    else:
      mirror.append(0)  # Default to no mirroring if 'm' field is missing

  links = skeleton["joints_idx"]

  goodmarks = list(range(1,len(skeleton["joint_names"])+1))
  for i in exclude_joints:
    goodmarks.remove(i) # Drop specific joint from goodmarks
  
  return cameraParams, rot, trans, mirror, links, goodmarks

def get_projected_points(predictions: dict, 
                         params: dict, 
                         cameraParams: list, 
                         rot: list, 
                         trans: list, 
                         mirror = None, 
                         com_3d = None ):
  """
    ### Takes the predictions, com3d and camera parameters, and returns the projected points.
    ### This assumes that cameras are in an indexed list (they are so in pdb data), and iterates over the length of params list

    predictions: dict of predictions read from .mat file or otherwise
    params: dict of params read from .mat file or otherwise
    cameraParams: list of dicts of camera parameters (Intrinsic Matrix, Radial Distort, and Tangential Distort)
    rot: list of arrays of camera rotations
    trans: list of arrays of camera translations
    mirror: list of single element lists indicating whether the particular camera view is mirrored
    com_3d: list of COM locations for each sample and view.

  """
  pose_3d = np.transpose(predictions["pred"],(0,2,1))
  n_samples = pose_3d.shape[0]
  
  # If com_3d points need to be plotted, then they need to be added to imagePoints
  if com_3d is None:
    tpred_bulk = pose_3d
  else:
    com_3d = np.expand_dims(com_3d, 1)
    tpred_bulk = np.concatenate((pose_3d,com_3d), axis=1)
  
  n_channels = tpred_bulk.shape[1]
  
  tpred_bulk = np.reshape(tpred_bulk,(-1,3))
  
  imagePoints_agg = []
  com_2d_agg = []

  # Calculate projections for each cam view for all of the predicted points
  for ncam in range (len(params)):
        camParam = cameraParams[ncam]
        rotation = rot[ncam]
        translation = trans[ncam]

        imagePoints_blk = dops.project_to2d(tpred_bulk, 
                                       camParam["IntrinsicMatrix"],
                                       rotation, 
                                       translation)[:, :2]

        imagePoints_blk = dops.distortPoints(imagePoints_blk, 
                                        camParam["IntrinsicMatrix"],
                                        np.squeeze(camParam["RadialDistortion"]), 
                                        np.squeeze(camParam["TangentialDistortion"])).T

        if mirror!=None :
          if mirror[ncam] == 1:
            imagePoints_blk[:,1] = 2048 - imagePoints_blk[:,1]
        else:
          imagePoints_blk = imagePoints_blk.T
        
        imagePoints_blk = np.reshape(imagePoints_blk, (-1, n_channels, 2))
        
        if not (com_3d is None):
          com_2d_blk = imagePoints_blk[:,-1:,:]
          imagePoints_blk = imagePoints_blk[:, :n_channels-1, :]
          com_2d_agg.append(com_2d_blk)

        imagePoints_agg.append(imagePoints_blk)
        

  
  return imagePoints_agg, com_2d_agg

def plot_projected_points(predictions, 
                          sync, 
                          params, 
                          imagePoints_agg,
                          com_2d_agg, 
                          goodmarks, 
                          links, 
                          color_dict,
                          videofle_path,
                          video_save_path,
                          start_sample = 0, 
                          max_samples = 1000, 
                          fps = 30,
                          cam_index = 0,
                          ):
  """
    # Plots the projected points and saves them to the locations specified in video_save_path
    # Both videofle_path and video_save_path except full paths with filename and extension.
    # This method is called from driver with all the related arguments passed.

    predictions: dict of predictions
    sync: dict required to sync frames from the video with the preductions. 
          This is necessary to determine which predictions correspons to which sample
    params: dict of params loaded from .mat file
    imagePoints_agg: list of lists of projection points for each camera view
    com_2d_agg: list of lists of projected Center of Mass for each camera view
    goodmarks: List of joint indices to consider while plotting
    links: List specifying which joint indices are connected to which with a bone
    color_dict: List of tupples mentioning colors for each bone
    videofle_path: Path from where to read video file
    video_save_path: Path to save videos to
    start_sample: SampleID to start reading frames from.
                  Default: 0
    max_samples: Max number of frames to read from video
                  Default: 1000
    cam_index: 0-based camera index (0=Camera1, 1=Camera2, ...). Must match videofle_path.
  """
  n_cams = len(imagePoints_agg)
  if cam_index < 0 or cam_index >= n_cams:
    raise ValueError("cam_index must be in [0, {}). Got {}.".format(n_cams, cam_index))
  movie_reader = imageio.get_reader(videofle_path)

  metadata = dict(title='dannce_visualization', artist='Matplotlib')
  writer = FFMpegWriter(fps=fps, metadata=metadata)

  fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

  if not os.path.exists(os.path.dirname(video_save_path)):
    os.makedirs(os.path.dirname(video_save_path))

  # Flatten data_frame and data_sampleID to handle both (N,) and (1,N) shapes
  data_frame_flat = np.squeeze(sync[0]["data_frame"]).flatten()
  data_sampleID_flat = np.squeeze(sync[0]["data_sampleID"]).flatten()
  
  with writer.saving(fig, video_save_path, dpi=300):

    for i in range(start_sample, start_sample + max_samples):
      # revised by haonan 20260128
      # frame should be taken from sync[0]["data_frame"] from an index where data_sampleID from sync[0] matches sampleID at i-th index from predictions
      # using np.where for this gives a nested numpy array containing a single element(the index), so use squeeze     
      matching_indices = np.where(data_sampleID_flat == predictions["sampleID"][0][i])[0]
      if len(matching_indices) == 0:
        print(f"Warning: No matching frame found for sampleID {predictions['sampleID'][0][i]} at index {i}. Skipping.")
        continue
      # Take the first match if multiple matches exist
      frame_idx = int(matching_indices[0])
      if frame_idx >= len(data_frame_flat):
        print(f"Warning: Frame index {frame_idx} is out of bounds (max: {len(data_frame_flat)-1}). Skipping.")
        continue
      fr = data_frame_flat[frame_idx]
      
      # Handle both scalar and array cases
      if np.isscalar(fr):
        frame_num = int(fr)
      else:
        frame_num = int(fr[0] if len(fr) > 0 else fr)
      
      frame = movie_reader.get_data(frame_num)
      print("Sample: ", i)
    
      axes.imshow(frame)      
      
      # Use selected camera's projections (cam_index: 0=Camera1, 1=Camera2, ...)
      imagePoints = imagePoints_agg[cam_index][i]
      if com_2d_agg != None:
        com = com_2d_agg[cam_index][i]
        axes.scatter(com[:,0], com[:,1], marker='.', color='red', linewidths=1)
      
      for mm in range(len(links)):
        if links[mm][0] in goodmarks and links[mm][1] in goodmarks:
          xx = [imagePoints[links[mm][0]-1,0],
                imagePoints[links[mm][1]-1,0]]
          yy = [imagePoints[links[mm][0]-1,1],
                imagePoints[links[mm][1]-1,1]]

          axes.scatter(xx, yy, marker = '.', color='white', linewidths=0.5)
          axes.plot(xx,yy, c=color_dict[mm % len(color_dict)], lw=2)
      
      axes.axis("off")
      axes.set_title(str(i))
        
      writer.grab_frame()
      axes.clear()

def driver(dannceMat_filepath : str, 
           preditcions_filepath : str, 
           videofle_path : str, 
           skeleton_path : str, 
           com3d_filepath : str,
           exclude_joints: list, 
           video_save_path: str, 
           start_sample = 0, 
           max_samples = 1000,
           fps=30,
           cam_index=0,
           ):
  print("Com3d in drive = ", com3d_filepath)
  cam_names, sync, params, skeleton, predictions, com_3d = get_data(dannceMat_filepath=dannceMat_filepath, 
                                                                    preditcions_filepath=preditcions_filepath, 
                                                                    skeleton_path=skeleton_path,
                                                                    com3d_filepath=com3d_filepath)
  
  cameraParams, rot, trans, mirror, links, goodmarks = get_camParams(params = params, skeleton = skeleton, 
                                                                     exclude_joints = exclude_joints)
  
  imagePoints_agg, com_2d_agg = get_projected_points(predictions, params, cameraParams, rot, trans, mirror, com_3d)

  if com_2d_agg == [] :
    com_2d_agg = None
  
  plot_projected_points(predictions, 
                        sync, 
                        params, 
                        imagePoints_agg,
                        com_2d_agg, 
                        goodmarks, 
                        links, 
                        COLOR_DICT,
                        videofle_path,
                        video_save_path,
                        start_sample = start_sample,
                        max_samples = max_samples,
                        fps=fps,
                        cam_index=cam_index)

driver(dannceMat_filepath, preditcions_filepath, videofle_path, skeleton_path, com3d_file,
        exclude_joints = [], 
        video_save_path = video_save_path, 
        start_sample = start_sample, 
        max_samples = max_samples,
        fps=fps_setting,
        cam_index=cam_index)