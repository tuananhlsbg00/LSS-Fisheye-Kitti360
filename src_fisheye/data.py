"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob
import time

from tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx
from kitti360scripts.viewer.BEVSegmentation import (load_poses, transform_points,
                                                    world_to_bev_indices, fill_polygon,
                                                    get_bottom_face, assign_color,
                                                    draw_fisheye_coverage, draw_ego_vehicle)
from kitti360scripts.helpers.project import CameraFisheye
from kitti360scripts.helpers.annotation import Annotation3D, local2global, global2local

class KittiData(torch.utils.data.Dataset):
    def __init__(self, is_train, data_aug_conf, grid_conf, is_aug=False):
        # Set paths (adjust as needed)
        if 'KITTI360_DATASET' in os.environ:
            self.kitti360Path = os.environ['KITTI360_DATASET']
        else:
            raise RuntimeError("Please set KITTI360_DATASET in your environment.")


        self.bboxPath = os.path.join(self.kitti360Path, 'data_3d_bboxes')
        self.posesPath = os.path.join(self.kitti360Path, 'data_poses')
        self.imagesPath = os.path.join(self.kitti360Path, 'data_2d_raw')

        self.is_train = is_train
        self.is_aug = is_aug
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.sequences = self.get_sequences()
        self.ixes = self.prepro()

        self.bboxes = self.get_bboxes(self.sequences)

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        self.bev_min = self.bx[0] - self.dx[0]/2
        self.bev_max = self.bev_min+self.dx[0]*self.nx[0]
        self.z_max   =  self.bx[2] - self.dx[2]/2 - self.dx[2]*self.nx[2]
        self.cams = self.get_cams()
        self.T_additional = self.shift_origin()
        print(self)


    def shift_origin(self,x=-0.81, y=-0.32, z=-0.9):

        T_additional = np.eye(4)
        T_additional[0, 3] = x  # Shift in X (backward)
        T_additional[1, 3] = y  # Shift in Y (left)
        T_additional[2, 3] = z  # Shift in Z (down)

        return T_additional

    def get_sequences(self, val_idxs=[8]):
        # filter by scene split
        # self.all_sequences = [                             '2013_05_28_drive_0000_sync',
        #                       '2013_05_28_drive_0003_sync','2013_05_28_drive_0002_sync',
        #                       '2013_05_28_drive_0005_sync','2013_05_28_drive_0004_sync',
        #                       '2013_05_28_drive_0007_sync','2013_05_28_drive_0006_sync',
        #                       '2013_05_28_drive_0009_sync',                             ]
        #
        # self.val_sequence =   ['2013_05_28_drive_0010_sync']
        sequences = sorted(os.listdir(self.imagesPath))
        val_sequence = [ sequences.pop(idx) for idx in val_idxs ]
        if self.is_train:

            return sequences

        return val_sequence

    def prepro(self):
        samples = []
        sample_dtype = np.dtype(
            [
                ('sequence','U26'),
                ('frame', 'U14'),
                ('pose', 'float32', (16,))
            ]
        )
        for sequence in self.sequences:

            posesPath = os.path.join(self.posesPath, sequence, 'poses.txt')

            if sequence=='2013_05_28_drive_0002_sync':
                poses = load_poses(posesPath, start=4335, max_poses=13668) #missing images in sequence 2
            else:
                poses = load_poses(posesPath, start=0, max_poses=None)
            pad = np.zeros((poses.shape[0], 4))
            pad[:, -1] = 1
            poses = np.hstack((poses, pad))
            frame = np.char.add(np.char.zfill(poses[:, 0].astype(int).astype(str), 10), ".png")
            temp_samples = np.array(list(zip([sequence]*len(poses), frame, poses[:, 1:])),
                                    dtype=sample_dtype)
            samples.append(temp_samples)

        return np.concatenate(samples)

    def get_bboxes(self, sequences):

        bboxes = {}

        for sequence in sequences:

            annotation3D = Annotation3D(self.bboxPath, sequence)
            bboxes[sequence] = annotation3D.objects

        return bboxes

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_aug_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                         resize=resize,
                                                         resize_dims=resize_dims,
                                                         crop=crop,
                                                         flip=flip,
                                                         rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrinsics = []
        distortions = []
        xis = []
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        for cam in cams.values():
            imagePath = os.path.join(self.imagesPath,
                                     rec['sequence'],
                                     cam.fi['camera_name'],
                                     'data_rgb',
                                     rec['frame'])

            img = Image.open(imagePath).resize((fW, fH))
            ratio = max(fH/H, fW/W)

            extrinsic = cam.camToPose
            rot = extrinsic[:3, :3]
            tran = extrinsic[:3, 3]

            intrinsic = np.array(
                [cam.fi['projection_parameters']['gamma1']*ratio,
                 cam.fi['projection_parameters']['gamma2']*ratio,
                 cam.fi['projection_parameters']['u0']*ratio,
                 cam.fi['projection_parameters']['v0']*ratio]
            )

            distortion = np.array(
                [cam.fi['distortion_parameters']['k1'],
                 cam.fi['distortion_parameters']['k2'],
                 cam.fi['distortion_parameters']['p1'],
                 cam.fi['distortion_parameters']['p2']]
            )

            xi = np.array(
                [cam.fi['mirror_parameters']['xi']]
            )

            imgs.append(normalize_img(img))
            rots.append(torch.from_numpy(rot))
            trans.append(torch.from_numpy(tran))
            intrinsics.append(torch.from_numpy(intrinsic))
            distortions.append(torch.from_numpy(distortion))
            xis.append(torch.from_numpy(xi))

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans), torch.stack(intrinsics), torch.stack(distortions), torch.stack(xis))


    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        T_imu_to_world = rec['pose'].reshape(4,4)
        frame = int(rec['frame'][0:-4])

        T_world_to_imu = np.linalg.inv(T_imu_to_world)
        T_total = self.T_additional @ T_world_to_imu  # Transformation from world to BEV (ego-car)

        bev_map = np.zeros((self.nx[0], self.nx[1]), dtype=np.uint8)

        for globalId, bbox_dict in self.bboxes[rec['sequence']].items():
            for obj in bbox_dict.values():
                # For dynamic objects, process only if the object's timestamp matches the current frame.
                if obj.timestamp != -1 and int(obj.timestamp) != frame:
                    continue

                # Transform vertices to IMU coordinates
                vertices_world = obj.vertices
                vertices_imu = transform_points(vertices_world, T_total)

                # Extract only the bottom face (4 points)
                bottom4 = get_bottom_face(vertices_imu)
                bottom4_xy = bottom4[:, :2]  # (4,2) shape
                bottom_z = bottom4[:, 2]

                # Ensure bounding box is within the BEV area
                if (bottom4_xy[:, 0].max() < self.bev_min or bottom4_xy[:, 0].min() > self.bev_max or
                    bottom4_xy[:, 1].max() < self.bev_min or bottom4_xy[:, 1].min() > self.bev_max or
                                                                     bottom_z.min() < self.z_max):
                    continue

                # Convert (x,y) to pixel indices
                indices = world_to_bev_indices(bottom4_xy, self.bev_min, self.bev_max, self.nx[0])
                # Fill the polygon using 4 bottom face points
                fill_polygon(bev_map, indices)

        binimg = np.rot90(bev_map, k=1)

        return torch.Tensor(binimg.copy()).unsqueeze(0)

    def get_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']

            cams_data = {}
            for cam in cams:
                cams_data[cam] = CameraFisheye(root_dir=self.kitti360Path, cam_id=int(cam[-1]))
        return cams_data

    def __str__(self):
        return f"""Kitti360Data: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(KittiData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

    def get_colored_binimg(self, rec, cams):
        T_imu_to_world = rec['pose'].reshape(4, 4)
        frame = int(rec['frame'][0:-4])

        T_world_to_imu = np.linalg.inv(T_imu_to_world)
        T_total = self.T_additional @ T_world_to_imu  # Transformation from world to BEV (ego-car)

        bev_map = np.zeros((self.nx[0], self.nx[1], 3), dtype=np.uint8)

        cam_02 = cams['image_02']
        cam_03 = cams['image_03']

        draw_fisheye_coverage(bev_map=bev_map, bev_min=self.bev_min,
                              bev_max=self.bev_max, bev_resolution=self.nx[0],
                              camera=cam_02, color=(162, 181, 224))

        draw_fisheye_coverage(bev_map=bev_map, bev_min=self.bev_min,
                              bev_max=self.bev_max, bev_resolution=self.nx[0],
                              camera=cam_03, color=(162, 181, 224))
        draw_ego_vehicle(bev_map)

        for globalId, bbox_dict in self.bboxes[rec['sequence']].items():
            for obj in bbox_dict.values():
                # For dynamic objects, process only if the object's timestamp matches the current frame.
                if obj.timestamp != -1 and int(obj.timestamp) != frame:
                    continue

                # Transform vertices to IMU coordinates
                vertices_world = obj.vertices
                vertices_imu = transform_points(vertices_world, T_total)

                # Extract only the bottom face (4 points)
                bottom4 = get_bottom_face(vertices_imu)
                bottom4_xy = bottom4[:, :2]  # (4,2) shape
                bottom_z   = bottom4[:, 2]

                # Ensure bounding box is within the BEV area
                if (bottom4_xy[:, 0].max() < self.bev_min or bottom4_xy[:, 0].min() > self.bev_max or   #limiting x axes
                    bottom4_xy[:, 1].max() < self.bev_min or bottom4_xy[:, 1].min() > self.bev_max or   #limiting y axes
                                                                     bottom_z.min() < self.z_max)    :  #limiting z axes
                    continue

                # Convert (x,y) to pixel indices
                indices = world_to_bev_indices(bottom4_xy, self.bev_min, self.bev_max, self.nx[0])
                r, g, b = assign_color(globalId)  # Fetch color from labels.py
                color = (b, g, r)
                # Fill the polygon using 4 bottom face points
                fill_polygon(bev_map, indices, color)

        colored_binimg = np.rot90(bev_map, k=1)

        return torch.Tensor(colored_binimg.copy()).unsqueeze(0)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.cams

        binimg = self.get_binimg(rec)

        colored_binimg = self.get_colored_binimg(rec, cams)

        # if self.is_aug:
        #     imgs, rots, trans, intrins, post_rots, post_trans = self.get_aug_image_data(rec, cams)
        #
        #     return imgs, rots, trans, intrins, post_rots, post_trans, binimg

        imgs, rots, trans, K, D, xi = self.get_image_data(rec, cams)

        return imgs, rots, trans, K, D, xi, binimg, colored_binimg


class SegmentationData(KittiData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.cams

        binimg = self.get_binimg(rec)

        # if self.is_aug:
        #     imgs, rots, trans, intrins, post_rots, post_trans = self.get_aug_image_data(rec, cams)
        #
        #     return imgs, rots, trans, intrins, post_rots, post_trans, binimg

        imgs, rots, trans, K, D, xi = self.get_image_data(rec, cams)


        return imgs, rots, trans, K, D, xi, binimg

def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(data_aug_conf, grid_conf, is_aug, bsz,
                 nworkers, parser_name):
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    traindata = parser(is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf, is_aug=is_aug)
    valdata = parser(is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf, is_aug=is_aug)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
if __name__ == '__main__':
    from tools import denormalize_img
    H = 1400
    W = 1400
    resize_lim = (0.193, 0.225)
    final_dim = (512, 512)
    bot_pct_lim = (0.0, 0.22)
    rot_lim = (-5.4, 5.4)
    rand_flip = True
    ncams = 2

    xbound = [-10.0, 10.0, 0.1]
    ybound = [-10.0, 10.0, 0.1]
    zbound = [  2.0, -2.0, 4.0]
    dbound = [  1.0, 14.0, 0.325]

    dataset = VizData(False,
                        data_aug_conf = {
                                        'resize_lim': resize_lim,
                                        'final_dim': final_dim,
                                        'rot_lim': rot_lim,
                                        'H': H, 'W': W,
                                        'rand_flip': rand_flip,
                                        'bot_pct_lim': bot_pct_lim,
                                        'cams': ['image_02', 'image_03'],
                                        'Ncams': ncams,
                                        },
                        grid_conf = {
                                    'xbound': xbound,
                                    'ybound': ybound,
                                    'zbound': zbound,
                                    'dbound': dbound,
                                    }
                        )



    print(dataset.sequences)
    print(dataset.ixes[10500:10500+20])
    print('sequence: ', dataset.ixes['sequence'].shape)
    print('frame: ', dataset.ixes['frame'].shape)
    print('pose: ', dataset.ixes['pose'].shape)
    print(len(dataset))
    print(dataset.ixes['pose'][-1].reshape(4, -1))
    print(dataset.cams['image_02'].fi, '\n', dataset.cams['image_03'].fi)
    print(dataset.cams['image_02'].camToPose, '\n', dataset.cams['image_03'].camToPose)



    for i in range(0,len(dataset),1):
        imgs, rots, trans, K, D, xi, binimg, colored_binimg = dataset[i]

        print(i)#, imgs.shape, rots.shape, trans.shape, K.shape, D.shape, xi.shape, binimg.shape)
        binimg = colored_binimg
        # Convert PIL Image to numpy array
        imgL_np = np.array(denormalize_img(imgs[0]))
        imgR_np = np.array(denormalize_img(imgs[1]))

        # Define the radius
        radius = 512//2

        u0L, v0L = K[0, 2:]
        u0R, v0R = K[1, 2:]

        # Draw the circle
        # Syntax: cv2.circle(image, center, radius, color, thickness)
        cv2.circle(imgL_np, center=(int(u0L), int(v0L)), radius=radius, color=(0, 255, 0), thickness=2)
        cv2.circle(imgR_np, center=(int(u0R), int(v0R)), radius=radius, color=(0, 255, 0), thickness=2)

        img_np = np.hstack((imgL_np, imgR_np))

        H, W, _ = img_np.shape
        # img_np = cv2.resize(img_np, (W//2, H//2))
        # # Convert RGB to BGR for cv2.imshow (if needed)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)



        cv2.imshow('img', img_bgr)
        # If the tensor is on GPU, first call .cpu(), then .numpy().
        binimg_np = colored_binimg.cpu().numpy().squeeze() .astype(np.uint8) # Now a NumPy array
        H, W, _ = binimg_np.shape
        binimg_np = cv2.resize(binimg_np, (W*2, H*2))

        # binimg_np = np.clip(binimg_np, 0, 255).astype(np.uint8)

        cv2.imshow('bev_map', binimg_np)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
