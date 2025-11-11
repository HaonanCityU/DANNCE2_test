%% Example setup for Label3D
% Label3D is a GUI for manual labeling of 3D keypoints in multiple cameras. 
% 
% Its main features include:
% 1. Simultaneous viewing of any number of camera views. 
% 2. Multiview triangulation of 3D keypoints.
% 3. Point-and-click and draggable gestures to label keypoints. 
% 4. Zooming, panning, and other default Matlab gestures
% 5. Integration with Animator classes. 
% 6. Support for editing prelabeled data.
% 
% Instructions:
% right: move forward one frameRate
% left: move backward one frameRate
% up: increase the frameRate
% down: decrease the frameRate
% t: triangulate points in current frame that have been labeled in at least two images and reproject into each image
% r: reset gui to the first frame and remove Animator restrictions
% u: reset the current frame to the initial marker positions
% z: Toggle zoom state
% p: Show 3d animation plot of the triangulated points. 
% backspace: reset currently held node (first click and hold, then
%            backspace to delete)
% pageup: Set the selectedNode to the first node
% tab: shift the selected node by 1
% shift+tab: shift the selected node by -1
% h: print help messages for all Animators
% shift+s: Save the data to a .mat file
clear all
close all;
addpath(genpath('deps'))
addpath(genpath('skeletons'))
danncePath = '/home/zhoulab/lijiejue/dannce/';

%% Load in the calibration parameter data
projectFolder = fullfile('/home/zhoulab/lijiejue/dannce_Data/dataset20220112flatview_2/');
cd(projectFolder)
calibPaths = collectCalibrationPaths(projectFolder);
params = cellfun(@(X) {load(X)}, calibPaths);

%% Load the videos into memory
vidName = '0.mp4';
vidPaths = collectVideoPaths(projectFolder,vidName);
videos = cell(numel(vidPaths),1);
sync = collectSyncPaths(projectFolder, '*.mat');
sync = cellfun(@(X) {load(X)}, sync);

% In case the demo folder uses the dannce.mat data format. 
% if isempty(sync)
%     dannce_file = dir(fullfile(projectFolder, '*dannce.mat'));
%     dannce = load(fullfile(dannce_file(1).folder, dannce_file(1).name));
%     sync = dannce.sync;
%     params = dannce.params;
% end

% framesToLabel = 13001:14000; %change as labeling proceeds
% for nVid = 1:numel(vidPaths)
%     frameInds = sync{nVid}.data_frame(framesToLabel);
%     videos{nVid} = readFrames(vidPaths{nVid}, frameInds+1);
% end
startframe=1;
endframe=1000;

for nVid=1:numel(vidPaths)
v=VideoReader(vidPaths{nVid});
videos{nVid} = read(v,[startframe endframe]);
end
%% Get the skeleton
skeleton = load('skeletons/mouse22.mat');
% skeleton = load('com');

%% Start Label3D
close all
labelGui = Label3D(params, videos, skeleton);
% labelGui = Label3D('/media/zhoulab/JJ Data/Videos/dataset20220106_overexpose_whitebright2/20220111_223351_Label3D.mat',videos);
% labelGui = Label3D(params, videos, skeleton, 'sync', sync, 'framesToLabel', framesToLabel);
% pts3d = load('/home/zhoulab/lijiejue/dannce_Data/dataset20220106_overexpose_whitebright2/DANNCE/predict_results/save_data_AVG0.mat','pred');
% pts3d = pts3d.pred;
% labelGui.loadFrom3D(pts3d(1:100,:,:))
  

%% Save to DANNCE format
labelGui.exportDannce('framesToLabel',startframe:endframe)

%% Check the camera positions
labelGui.plotCameras     

%% If you just wish to view labels, use View 3D
close all
viewGui = View3D(params, videos, skeleton);
pts3d = load('/home/zhoulab/lijiejue/dannce_Data/dataset20220112flatview_2/DANNCE/predict_results/save_data_MAX0.mat','pred');
pts3d = pts3d.pred;
viewGui.loadFrom3D(pts3d(10001:11000,:,:))
%% You can load both in different ways
close all;
View3D()

%% Generate dannce.mat to train DANNCE network
com=load('./COM/predict_results/com3d.mat','com','sampleID');
com=renameStructField(com,'com','com3d');
