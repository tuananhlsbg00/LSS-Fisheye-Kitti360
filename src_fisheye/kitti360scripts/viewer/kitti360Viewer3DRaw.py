#!/usr/bin/python
# -*- coding: utf-8 -*-


#################
## Import modules
#################
import sys
# walk directories
import glob
# access to OS functionality
import os
# copy things
import copy
# numpy
import numpy as np
# matplotlib for colormaps
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# struct for reading binary ply files
import struct
from kitti360scripts.helpers.csHelpers import Rodrigues
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid

CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        from kitti360scripts.helpers import curlVelodyneData
    except:
        CSUPPORT = False
        print('CSUPPORT is required for unwrapping the velodyne data!')
        print('Run ``CYTHONIZE_EVAL= python setup.py build_ext --inplace`` to build with cython')
        sys.exit(-1)


# the main class that loads raw 3D scans
class Kitti360Viewer3DRaw(object):

    # Constructor
    def __init__(self, seq=0, mode='velodyne'):

        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')

        if mode=='velodyne':
            self.sensor_dir='velodyne_points'
        elif mode=='sick':
            self.sensor_dir='sick_points'
        else:
            raise RuntimeError('Unknown sensor type!')

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath  = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')

        self.kitti360Path = kitti360Path
        self.sequence = sequence
        self.loadPoses()
        self.loadExtrinsics()

    # poses are required to unwrap velodyne points to compensate for ego-motion
    def loadPoses(self):
        # load poses
        filePoses = os.path.join(self.kitti360Path, 'data_poses', self.sequence, 'poses.txt')
        poses = np.loadtxt(filePoses)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,3,4])
        self.Tr_pose_world = {}
        self.frames = frames
        for frame, pose in zip(frames, poses): 
            pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
            self.Tr_pose_world[frame] = pose

    def loadExtrinsics(self):
        # cam_0 to velo
        fileCameraToVelo = os.path.join(self.kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
        TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

        # all cameras to system center 
        fileCameraToPose = os.path.join(self.kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
        TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)
  
        self.TrVeloToPose = TrCamToPose['image_00'] @ np.linalg.inv(TrCam0ToVelo)

        # velodyne to all cameras
        self.TrVeloToCam = {}
        for k, v in TrCamToPose.items():
            # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
            TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
            TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
            # Tr(velo -> cam_k)
            self.TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,4])
        return pcd 

    def loadSickData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,2])
        pcd = np.concatenate([np.zeros_like(pcd[:,0:1]), -pcd[:,0:1], pcd[:,1:2]], axis=1)
        return pcd 

    def curlParameterFromPoses(self, frame):
        Tr_pose_pose = np.eye(4)

        if frame in self.Tr_pose_world.keys():
            if frame==1:
                if frame+1 in self.Tr_pose_world.keys():
                    Tr_pose_pose = np.linalg.inv(self.Tr_pose_world[frame+1]) @ self.Tr_pose_world[frame]
            else:
                if frame-1 in self.Tr_pose_world.keys():
                    Tr_pose_pose = np.linalg.inv(self.Tr_pose_world[frame]) @ self.Tr_pose_world[frame-1]
        Tr_delta = np.linalg.inv(self.TrVeloToPose) @ Tr_pose_pose @ self.TrVeloToPose
        
        r = Rodrigues(Tr_delta[0:3,0:3])
        t = Tr_delta[0:3,3]
        return r.flatten(),t


    def curlVelodyneData(self, frame, pcd):
        pcd=pcd.astype(np.float64)
        pcd_curled = np.copy(pcd) 
        # get curl parameters 
        r,t = self.curlParameterFromPoses(frame)
        # unwrap points to compensate for ego motion
        pcd_curled = curlVelodyneData.cCurlVelodyneData(pcd, pcd_curled, r, t)
        return pcd_curled.astype(np.float32)
        

def projectVeloToImage(cam_id=0, seq=0):
    from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
    from PIL import Image
    import matplotlib.pyplot as plt

    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')
    
    sequence = '2013_05_28_drive_%04d_sync'%seq

    # perspective camera
    if cam_id in [0,1]:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye camera
    elif cam_id in [2,3]:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
    else:
        raise RuntimeError('Unknown camera ID!')

    # object for parsing 3d raw data 
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq)
    
    # take the rectification into account for perspective cameras
    if cam_id==0 or cam_id == 1:
        TrVeloToRect = np.matmul(camera.R_rect, velo.TrVeloToCam['image_%02d' % cam_id])
    else:
        TrVeloToRect = velo.TrVeloToCam['image_%02d' % cam_id]

    # color map for visualizing depth map
    cm = plt.get_cmap('jet')

    # visualize a set of frame
    # for each frame, load the raw 3D scan and project to image plane
    for frame in range(0, 1000, 50):
        points = velo.loadVelodyneData(frame)
        # curl velodyne
        points = velo.curlVelodyneData(frame, points)

        points[:,3] = 1

        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:,:3]
        # project to image space
        u,v, depth= camera.cam2image(pointsCam.T)
        u = u.astype(int)
        v = v.astype(int)

        # prepare depth map for visualization
        depthMap = np.zeros((camera.height, camera.width))
        depthImage = np.zeros((camera.height, camera.width, 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
        # visualize points within 30 meters
        mask = np.logical_and(np.logical_and(mask, depth>0), depth<30)
        depthMap[v[mask],u[mask]] = depth[mask]
        layout = (2,1) if cam_id in [0,1] else (1,2)
        sub_dir = 'data_rect' if cam_id in [0,1] else 'data_rgb'
        fig, axs = plt.subplots(*layout, figsize=(18,12))

        # load RGB image for visualization
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir, '%010d.png' % frame)
        if not os.path.isfile(imagePath):
            raise RuntimeError('Image file %s does not exist!' % imagePath)

        colorImage = np.array(Image.open(imagePath)) / 255.
        depthImage = cm(depthMap/depthMap.max())[...,:3]
        colorImage[depthMap>0] = depthImage[depthMap>0]

        axs[0].imshow(depthMap, cmap='jet', interpolation='none')
        axs[0].title.set_text('Projected Depth')
        axs[0].axis('off')
        axs[1].imshow(colorImage)
        axs[1].title.set_text('Projected Depth Overlaid on Image')
        axs[1].axis('off')
        plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (seq, cam_id, frame))
        plt.show()

if __name__=='__main__':

    visualizeIn2D = True
   
    # sequence index
    seq = 0
    # set it to 0 or 1 for projection to perspective images
    #           2 or 3 for projecting to fisheye images
    cam_id = 2
    
    # visualize raw 3D velodyne scans in 2D
    if visualizeIn2D:
        projectVeloToImage(seq=seq, cam_id=cam_id)

    # visualize raw 3D scans in 3D
    else:
        # open3d
        try:
            import open3d
        except:
            raise ValueError("Open3d required to visualize raw 3D scans in 3D!")
        mode = 'velodyne'
        frame = 0

        v = Kitti360Viewer3DRaw(mode=mode)
        if mode=='velodyne':
            points = v.loadVelodyneData(frame)
        elif mode=='sick':
            points = v.loadSickData(frame)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        
        open3d.visualization.draw_geometries([pcd])

