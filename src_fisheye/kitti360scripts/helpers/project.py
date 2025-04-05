import os
import numpy as np
import re
import yaml
import sys
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose
import math
import cv2
def readYAMLFile(fileName):
    '''make OpenCV YAML file compatible with python'''
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut, Loader=yaml.SafeLoader)
    return ret

class Camera:
    def __init__(self):
        
        # load intrinsics
        self.load_intrinsics(self.intrinsic_file)

        # load poses
        poses = np.loadtxt(self.pose_file)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,3,4])
        self.cam2world = {}
        self.frames = frames
        for frame, pose in zip(frames, poses): 
            pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
            # consider the rectification for perspective cameras
            if self.cam_id==0 or self.cam_id==1:
                self.cam2world[frame] = np.matmul(np.matmul(pose, self.camToPose),
                                                  np.linalg.inv(self.R_rect))
            # fisheye cameras
            elif self.cam_id==2 or self.cam_id==3:
                self.cam2world[frame] = np.matmul(pose, self.camToPose)
            else:
                raise RuntimeError('Unknown Camera ID!')


    def world2cam(self, points, R, T, inverse=False):
        assert (points.ndim==R.ndim)
        assert (T.ndim==R.ndim or T.ndim==(R.ndim-1)) 
        ndim=R.ndim
        if ndim==2:
            R = np.expand_dims(R, 0) 
            T = np.reshape(T, [1, -1, 3])
            points = np.expand_dims(points, 0)
        if not inverse:
            points = np.matmul(R, points.transpose(0,2,1)).transpose(0,2,1) + T
        else:
            points = np.matmul(R.transpose(0,2,1), (points - T).transpose(0,2,1))

        if ndim==2:
            points = points[0]

        return points

    def cam2image(self, points):
        raise NotImplementedError

    def load_intrinsics(self, intrinsic_file):
        raise NotImplementedError
    
    def project_vertices(self, vertices, frameId, inverse=True):

        # current camera pose
        curr_pose = self.cam2world[frameId]
        T = curr_pose[:3,  3]
        R = curr_pose[:3, :3]

        # convert points from world coordinate to local coordinate 
        points_local = self.world2cam(vertices, R, T, inverse)

        # perspective projection
        u,v,depth = self.cam2image(points_local)

        return (u,v), depth 

    def __call__(self, obj3d, frameId):

        vertices = obj3d.vertices

        uv, depth = self.project_vertices(vertices, frameId)

        obj3d.vertices_proj = uv
        obj3d.vertices_depth = depth 
        obj3d.generateMeshes()


class CameraPerspective(Camera):

    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync', cam_id=0):
        # perspective camera ids: {0,1}, fisheye camera ids: {2,3}
        assert (cam_id==0 or cam_id==1)

        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        calib_dir = os.path.join(root_dir, 'calibration')
        self.pose_file = os.path.join(pose_dir, "poses.txt")
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
        self.cam_id = cam_id
        super(CameraPerspective, self).__init__()

    def load_intrinsics(self, intrinsic_file):
        ''' load perspective intrinsics '''
    
        intrinsic_loaded = False
        width = -1
        height = -1
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_%02d:' % self.cam_id:
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3,4])
                intrinsic_loaded = True
            elif line[0] == 'R_rect_%02d:' % self.cam_id:
                R_rect = np.eye(4) 
                R_rect[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)
            elif line[0] == "S_rect_%02d:" % self.cam_id:
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert(intrinsic_loaded==True)
        assert(width>0 and height>0)
    
        self.K = K
        self.width, self.height = width, height
        self.R_rect = R_rect

    def cam2image(self, points):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(self.K[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth

class CameraFisheye(Camera):
    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync', cam_id=2):
        # perspective camera ids: {0,1}, fisheye camera ids: {2,3}
        assert (cam_id==2 or cam_id==3)

        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        calib_dir = os.path.join(root_dir, 'calibration')
        self.pose_file = os.path.join(pose_dir, "poses.txt")
        self.intrinsic_file = os.path.join(calib_dir, 'image_%02d.yaml' % cam_id)
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
        self.cam_id = cam_id
        super(CameraFisheye, self).__init__()

    def load_intrinsics(self, intrinsic_file):
        ''' load fisheye intrinsics '''

        intrinsics = readYAMLFile(intrinsic_file)

        self.width, self.height = intrinsics['image_width'], intrinsics['image_height']
        self.fi = intrinsics
    def cam2image(self, points):
        ''' camera coordinate to image plane '''
        points = points.T
        norm = np.linalg.norm(points, axis=1)
        # print('******* Start cam2image',points.shape, '************\n',points,'\n*********** End cam2image***********')
        x = points[:,0] / norm
        y = points[:,1] / norm
        z = points[:,2] / norm

        x /= z+self.fi['mirror_parameters']['xi']
        y /= z+self.fi['mirror_parameters']['xi']

        k1 = self.fi['distortion_parameters']['k1']
        k2 = self.fi['distortion_parameters']['k2']
        gamma1 = self.fi['projection_parameters']['gamma1']
        gamma2 = self.fi['projection_parameters']['gamma2']
        u0 = self.fi['projection_parameters']['u0']
        v0 = self.fi['projection_parameters']['v0']

        ro2 = x*x + y*y
        u = x*(1 + k1*ro2 + k2*ro2*ro2)
        v = y*(1 + k1*ro2 + k2*ro2*ro2)

        Px = gamma1*u + u0
        Py = gamma2*v + v0

        return Px, Py, norm * points[:,2] / np.abs(points[:,2])

    def cam2image_ocv2(self, points):
        points = points.reshape(-1, 1, 3)
        xi = self.fi['mirror_parameters']['xi']

        k1 = self.fi['distortion_parameters']['k1']
        k2 = self.fi['distortion_parameters']['k2']
        p1 = self.fi['distortion_parameters']['p1']
        p2 = self.fi['distortion_parameters']['p2']
        gamma1 = self.fi['projection_parameters']['gamma1']
        gamma2 = self.fi['projection_parameters']['gamma2']
        u0 = self.fi['projection_parameters']['u0']
        v0 = self.fi['projection_parameters']['v0']

        K = np.array([[gamma1, 0, u0],
                      [0, gamma2, v0],
                      [0,      0,  1]])

        D = np.array([k1, k2, p1, p2])
        rvec = np.zeros((3, 1), dtype=np.float64)  # Zero rotation vector: identity rotation
        tvec = np.zeros((3, 1), dtype=np.float64)  # Zero translation vector
        projected_point, _ = cv2.omnidir.projectPoints(objectPoints=points,
                                                       rvec=rvec, tvec=tvec, K=K,
                                                       D=D, xi=xi)

        return projected_point

    def image2cam_cv2(self, points, Z=500):
        points = points.reshape(-1, 1, 2)
        Z = np.array(Z).reshape(-1, 1, 1)
        xi = np.array([self.fi['mirror_parameters']['xi']], dtype=np.float64)


        k1 = self.fi['distortion_parameters']['k1']
        k2 = self.fi['distortion_parameters']['k2']
        p1 = self.fi['distortion_parameters']['p1']
        p2 = self.fi['distortion_parameters']['p2']
        gamma1 = self.fi['projection_parameters']['gamma1']
        gamma2 = self.fi['projection_parameters']['gamma2']
        u0 = self.fi['projection_parameters']['u0']
        v0 = self.fi['projection_parameters']['v0']

        K = np.array([[gamma1, 0, u0],
                      [0, gamma2, v0],
                      [0,      0,  1]])

        D = np.array([k1, k2, p1, p2])

        R = np.eye(3)

        undistorted_points_norm = cv2.omnidir.undistortPoints(points,K,D,xi,R)

        undistorted_points = undistorted_points_norm * Z
        return np.concatenate((undistorted_points, Z), axis=2)
    def image2cam(self, u, v, z=50):
        """
        Approximate unprojection of pixel (u,v) to a 3D ray [X, Y, Z] in camera coordinates.
        This is a simplified approach for demonstration.
        """
        # 1) Normalize pixel around principal point
        mx = (u - self.fi['projection_parameters']['u0']) / self.fi['projection_parameters']['gamma1']
        my = (v - self.fi['projection_parameters']['v0']) / self.fi['projection_parameters']['gamma2']

        # 2) Undo radial distortion (k1, k2) with a simple "one-step" iteration
        #    Real MEI might need a more robust method (like Newton-Raphson).
        k1 = self.fi['distortion_parameters']['k1']
        k2 = self.fi['distortion_parameters']['k2']

        # We'll guess an undistorted radius "ru" from the distorted radius "rd"
        rd = math.sqrt(mx * mx + my * my)
        # invert polynomial: ru ~= rd / (1 + k1*rd^2 + k2*rd^4)
        if rd > 1e-12:
            factor = 1.0 / (1.0 + k1 * rd * rd + k2 * rd ** 4)
            ru = rd * factor
            mx_u = mx * (ru / rd)
            my_u = my * (ru / rd)
        else:
            mx_u = mx
            my_u = my

        # 3) Apply the MEI mirror parameter xi
        xi = self.fi['mirror_parameters']['xi']
        # For MEI: x' = X / (Z + xi * norm3D)
        # we approximate norm3D ~ sqrt(X^2+Y^2+Z^2). We'll do a small assumption:
        # Let Z=1 for an initial guess, so norm3D ~ sqrt(X^2 + Y^2 + 1^2).
        # Then X = x' * (Z + xi*norm3D). We'll do a small iteration for Z.
        # For simplicity, do a single pass:

        # Start guess:
        Z = 1.0
        for _ in range(2):  # do 2 small iterations
            r2 = mx_u * mx_u + my_u * my_u
            # norm3D ~ sqrt(X^2 + Y^2 + Z^2), but X=mx_u*(Z + xi*norm3D)
            # We'll skip fully correct iteration. Just do a small approach:
            norm_approx = math.sqrt(r2 + Z * Z)
            Z = 1.0 / (xi) - norm_approx  # a naive guess

        # After iteration, compute final X, Y using the found Z
        norm_approx = math.sqrt(r2 + Z * Z)
        denom = (Z + xi * norm_approx)
        if abs(denom) < 1e-12:
            return np.array([0, 0, -1], dtype=np.float32)  # behind camera

        X = mx_u * denom
        Y = my_u * denom

        # Return direction vector. If you prefer a unit vector, normalize:
        direction = np.array([X, Y, Z], dtype=np.float32)
        direction_norm = direction / np.linalg.norm(direction)
        return direction_norm

if __name__=="__main__":
    import cv2
    import matplotlib.pyplot as plt
    from labels import id2label

    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')
    
    seq = 3
    cam_id = 2
    sequence = '2013_05_28_drive_%04d_sync'%seq
    # perspective
    if cam_id == 0 or cam_id == 1:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye
    elif cam_id == 2 or cam_id == 3:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
        print(camera.fi)
    else:
        raise RuntimeError('Invalid Camera ID!')

    # loop over frames
    for frame in camera.frames:
        # perspective
        if cam_id == 0 or cam_id == 1:
            image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rect', '%010d.png'%frame)
        # fisheye
        elif cam_id == 2 or cam_id == 3:
            image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rgb', '%010d.png'%frame)
        else:
            raise RuntimeError('Invalid Camera ID!')
        if not os.path.isfile(image_file):
            print('Missing %s ...' % image_file)
            continue


        print(image_file)
        image = cv2.imread(image_file)
        plt.imshow(image[:,:,::-1])

        # 3D bbox
        from annotation import Annotation3D
        label3DBboxPath = os.path.join(kitti360Path, 'data_3d_bboxes')
        annotation3D = Annotation3D(label3DBboxPath, sequence)

        points = []
        depths = []
        for k,v in annotation3D.objects.items():
            if len(v.keys())==1 and (-1 in v.keys()): # show static only
                obj3d = v[-1]
                if not id2label[obj3d.semanticId].name=='building': # show buildings only
                    continue
                camera(obj3d, frame)
                vertices = np.asarray(obj3d.vertices_proj).T
                points.append(np.asarray(obj3d.vertices_proj).T)
                depths.append(np.asarray(obj3d.vertices_depth))
                for line in obj3d.lines:
                    v = [obj3d.vertices[line[0]]*x + obj3d.vertices[line[1]]*(1-x) for x in np.arange(0,1,0.01)]
                    uv, d = camera.project_vertices(np.asarray(v), frame)
                    mask = np.logical_and(np.logical_and(d>0, uv[0]>0), uv[1]>0)
                    mask = np.logical_and(np.logical_and(mask, uv[0]<image.shape[1]), uv[1]<image.shape[0])
                    plt.plot(uv[0][mask], uv[1][mask], 'r.')

        plt.pause(0.5)
        plt.clf()
