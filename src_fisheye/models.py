"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, is_aug=False):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.is_aug = is_aug


        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )

        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        self.mask = None

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        # B, N, _ = trans.shape
        #
        # # undo post-transformation
        # # B x N x D x H x W x 3
        # points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        # points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        #
        # # cam_to_ego
        # points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
        #                     points[:, :, :, :, :, 2:3]
        #                     ), 5)
        # combine = rots.matmul(torch.inverse(intrins))
        # points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # points += trans.view(B, N, 1, 1, 1, 3)
        #
        # return points
        pass

    def undistort_points_pytorch(self, points, K, D, xi, iterations=20):
        """
        Differentiable implementation of MEI-model unprojection.
        points: Tensor of shape (N, 2) representing distorted pixel coordinates.
        K: Tensor of shape (3, 3) containing intrinsic parameters.
        D: Tensor of shape (4,) with distortion coefficients: [k1, k2, p1, p2].
        xi: Scalar (tensor) for the mirror parameter.
        Returns: Tensor of shape (N, 3) representing unit ray directions.
        """
        B, N = K.shape[0], K.shape[1]

        # Expand K components to shape [B, N, 1, 1, 1] for broadcasting:
        gamma1 = K[..., 0].view(B, N, 1, 1, 1)
        gamma2 = K[..., 1].view(B, N, 1, 1, 1)
        u0 = K[..., 2].view(B, N, 1, 1, 1)
        v0 = K[..., 3].view(B, N, 1, 1, 1)

        # Expand distortion parameters D to shape [B, N, 1, 1, 1]:
        k1 = D[..., 0].view(B, N, 1, 1, 1)
        k2 = D[..., 1].view(B, N, 1, 1, 1)
        p1 = D[..., 2].view(B, N, 1, 1, 1)
        p2 = D[..., 3].view(B, N, 1, 1, 1)

        # Ensure xi is of shape [B, N, 1, 1, 1]:
        xi = xi.view(B, N, 1, 1, 1)

        # Normalize pixel coordinates to the idealized image plane.
        pp = torch.stack([(points[..., 0] - u0) / gamma1, (points[..., 1] - v0) / gamma2], dim=-1)  # [B, N, D, H, W, 2]
        pu = pp.clone()  # initial guess for undistorted points


        # Iteratively remove distortion:
        for _ in range(iterations):
            # Use ... to index the last dimension
            r2 = pu[..., 0] ** 2 + pu[..., 1] ** 2  # [B, N, D, H, W]
            r4 = r2 ** 2  # [B, N, D, H, W]
            denom = 1 + k1 * r2 + k2 * r4  # broadcast to [B, N, D, H, W]
            pu_x = (pp[..., 0] - 2 * p1 * pu[..., 0] * pu[..., 1] - p2 * (r2 + 2 * pu[..., 0] ** 2)) / denom
            pu_y = (pp[..., 1] - 2 * p2 * pu[..., 0] * pu[..., 1] - p1 * (r2 + 2 * pu[..., 1] ** 2)) / denom
            pu = torch.stack([pu_x, pu_y], dim=-1)  # [B, N, D, H, W, 2]

            # Compute quadratic coefficients for Zs (MEI model)
        r2 = pu[..., 0] ** 2 + pu[..., 1] ** 2  # [B, N, D, H, W]
        a = r2 + 1
        b = 2 * xi * r2
        c = (xi ** 2) * r2 - 1
        disc = torch.sqrt(b ** 2 - 4 * a * c)
        Zs = (-b + disc) / (2 * a)  # [B, N, D, H, W]

        # Form the 3D ray:
        X = pu[..., 0] * (Zs + xi)  # [B, N, D, H, W]
        Y = pu[..., 1] * (Zs + xi)  # [B, N, D, H, W]
        Z = Zs                      # [B, N, D, H, W]
        points_3d = torch.stack([X, Y, Z], dim=-1) # [B, N, D, H, W, 3]

        # Normalize to unit vector (optional if you want just the direction)
        points_3d = nn.functional.normalize(points_3d, p=2, dim=-1)
        return points_3d

    def build_fisheye_circle_mask(self, B, N, H, W, K):
        """
        K: shape [B, N, 4], each row [gamma1, gamma2, u0, v0].
        Returns: mask shape [B, N, H, W], dtype=bool
        """
        device = K.device  # to ensure everything is on the same device

        # 1) Create a meshgrid for H, W, shape [H, W]
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        # shape of yy, xx is [H, W]

        # 2) Expand them to [1, 1, H, W], so we can broadcast over B,N
        yy = yy.unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]
        xx = xx.unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]

        # 3) Expand them to [B,N,H,W]
        xx = xx.expand(B, N, H, W)
        yy = yy.expand(B, N, H, W)

        # 4) Extract and expand u0,v0
        # (u0, v0) are originally [B, N], but we also used "floor div" by self.downsample
        # so let's do that carefully:
        u0 = torch.div(K[..., 2], self.downsample, rounding_mode='floor')  # shape [B,N]
        v0 = torch.div(K[..., 3], self.downsample, rounding_mode='floor')  # shape [B,N]

        # expand them to [B,N,H,W]
        u0 = u0.unsqueeze(-1).unsqueeze(-1).expand(B, N, H, W)
        v0 = v0.unsqueeze(-1).unsqueeze(-1).expand(B, N, H, W)

        # 5) For simplicity, radius = min(H,W)//2
        radius = (min(H, W) - 1 ) // 2

        # 6) Compute squared distance from center
        dist2 = (xx - u0) ** 2 + (yy - v0) ** 2  # shape [B,N,H,W]

        # 7) Compare with radius^2
        mask = dist2 >= (radius ** 2)  # shape [B,N,H,W], True if out-of-range

        # window = mask[0, 0, :, :] #chi de test thoi, dung co uncomment
        #
        # for line in window:
        #     print(' '.join(map(lambda x: str(round(x, 1)), line.tolist())))


        return mask.unsqueeze(2)

    def get_geometry_fisheye(self, rots, trans,
                             # For fisheye, we need additional parameters:
                             # K, D, xi can be embedded in 'intrins' or passed separately.
                             K, D, xi):
        """
        Determine the (x,y,z) locations (in the ego frame) for fisheye images using
        a differentiable MEI unprojection.
        Returns a tensor of shape: [B, N, D, H_down, W_down, 3]
        """
        B, N, _ = trans.shape  # batch, number of cameras

        # Assume self.frustum is of shape [D, H, W, 3] and was computed based on final_dim.
        # Expand frustum to [B, N, D, H, W, 3]:
        frustum = self.frustum.unsqueeze(0).unsqueeze(0)  # now [1,1,D,H,W,3]
        frustum = frustum.expand(B, N, -1, -1, -1, -1)  # [B, N, D, H, W, 3]

        # For each point in the frustum, the first two channels are pixel coordinates.
        # We reshape them into a 2D tensor to apply our differentiable MEI unprojection.
        # Note: Since we ignore augmentation, we don't subtract post_trans or undo any rotation.
        # Extract pixel coordinates and depth candidates
        pixel_coords = frustum[..., :2]  # [B, N, D, H, W, 2]
        depth_candidates = frustum[..., 2:3]  # [B, N, D, H, W, 1]

        # Unproject pixel coordinates to 3D rays
        unit_rays = self.undistort_points_pytorch(pixel_coords, K, D, xi)  # [B, N, D, H, W, 3]

        ## Scale by depth to get camera-frame 3D points
        points_cam = unit_rays * depth_candidates  # [B, N, D, H, W, 3]

        # Transform from camera to ego coordinates
        # Expand rots and trans for broadcast:
        rots_expanded = rots.view(B, N, 1, 1, 1, 3, 3)  # [B, N, 1, 1, 1, 3, 3]
        trans_expanded = trans.view(B, N, 1, 1, 1, 3)  # [B, N, 1, 1, 1, 3]

        # Add an extra dimension to points_cam for matrix multiplication:
        points_cam = points_cam.unsqueeze(-1)  # [B, N, D, H, W, 3, 1]

        # Rotate points: [B, N, D, H, W, 3, 1] = rots_expanded x points_cam
        points_ego = torch.matmul(rots_expanded, points_cam).squeeze(-1)

        # Add the camera translation to obtain final points in ego coordinates:
        points_ego += trans_expanded  # shape: [B, N, D, H, W, 3]

        B, N, D, H, W, _ = points_ego.shape

        if (self.mask is None):
            self.mask = self.build_fisheye_circle_mask(B, N, H, W, K).expand(-1, -1, points_ego.shape[2], -1, -1)
        elif self.mask.shape[0] != B:
            self.mask = self.build_fisheye_circle_mask(B, N, H, W, K).expand(-1, -1, points_ego.shape[2], -1, -1)

        points_ego[self.mask, :] = -1e3
        points_ego[..., 0] *= -1
        return points_ego  # [B, N, D, H, W, 3]

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        x[self.mask, :] = 0

        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, K, D, xi):
        geom = self.get_geometry_fisheye(rots, trans, K, D, xi)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC=1, is_aug=False):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC, is_aug)


if __name__ == '__main__':
    from data import SegmentationData
    import numpy as np
    import open3d as o3d
    import matplotlib.pyplot as plt
    import cv2
    from tools import SimpleLoss, get_val_info



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
    zbound = [ 2.0, -2.0, 4.0]
    dbound = [1.0, 14.0, 0.325]

    bsz = 4
    nworkers = 10
    pos_weight = 2.13

    gpuid = 0

    Model = compile_model(grid_conf={
                                   'xbound': xbound,
                                   'ybound': ybound,
                                   'zbound': zbound,
                                   'dbound': dbound,
                                    },
                          data_aug_conf={
                                    'resize_lim': resize_lim,
                                    'final_dim': final_dim,
                                    'rot_lim': rot_lim,
                                    'H': H, 'W': W,
                                    'rand_flip': rand_flip,
                                    'bot_pct_lim': bot_pct_lim,
                                    'cams': ['image_02', 'image_03'],
                                    'Ncams': ncams,
                                    }
                          )

    dataset = SegmentationData(False,
                               data_aug_conf={
                                   'resize_lim': resize_lim,
                                   'final_dim': final_dim,
                                   'rot_lim': rot_lim,
                                   'H': H, 'W': W,
                                   'rand_flip': rand_flip,
                                   'bot_pct_lim': bot_pct_lim,
                                   'cams': ['image_02', 'image_03'],
                                   'Ncams': ncams,
                                    },
                               grid_conf={
                                   'xbound': xbound,
                                   'ybound': ybound,
                                   'zbound': zbound,
                                   'dbound': dbound,
                                    }
                               )

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    valloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                            shuffle=False,
                                            num_workers=nworkers)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                            shuffle=False,
                                            num_workers=nworkers)
    loss_fn = SimpleLoss(pos_weight=pos_weight)

    points=[]

    for batchi, (imgs, rots, trans, K, D, xi, binimgs) in enumerate(valloader):
        # val_info = get_val_info(Model, valloader, loss_fn, device)

        B, N, _ = trans.shape
        frustum = Model.frustum.unsqueeze(0).unsqueeze(0)
        frustum = frustum.expand(B, N, -1, -1, -1, -1)
        pixel_coords = frustum[..., :2]
        print(frustum.shape, imgs.shape, K.shape, D.shape, xi.shape, pixel_coords.shape)
        print(Model.undistort_points_pytorch(pixel_coords, K, D, xi).shape)
        geom = Model.get_geometry_fisheye(rots, trans, K, D, xi)
        window = geom[0, 0, 5, :, :, 0]
        points = geom[0, :].view(-1, 3).numpy()
        break


    # for batchi, (imgs, rots, trans, K, D, xi, binimgs) in enumerate(trainloader):
    #
    #     B, N, _ = trans.shape
    #     frustum = Model.frustum.unsqueeze(0).unsqueeze(0)
    #     frustum = frustum.expand(B, N, -1, -1, -1, -1)
    #     pixel_coords = frustum[..., :2]
    #     print(frustum.shape, imgs.shape, K.shape, D.shape, xi.shape, pixel_coords.shape)
    #     print(Model.undistort_points_pytorch(pixel_coords, K, D, xi).shape)
    #     geom = Model.get_geometry_fisheye(rots, trans, K, D, xi)
    #     window = geom[0, 0, 5, :, :, 0]
    #     points = geom[0, :].view(-1, 3).numpy()
    #     break

    B, N, Depth, H, W, _ = geom.shape

    print(imgs.shape)

    Bevgrid = torch.sum(Model.get_voxels(imgs, rots, trans, K, D, xi), dim=1, keepdim=False)[0].detach().cpu().numpy()

    H, W = Bevgrid.shape
    Bevgrid = np.abs(cv2.resize(Bevgrid, (W * 4, H * 4)))

    for line in Bevgrid[:25,:25]:
        print(' '.join(map(lambda x : str(round(x, 1)),line.tolist())))

    img_normalized = cv2.normalize(Bevgrid, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    for line in img_normalized[:25,:25]:
        print(' '.join(map(lambda x : str(round(x, 1)),line.tolist())))

    cv2.imshow('BEvgrid', img_normalized.astype(np.uint8))
    cv2.imwrite('./evaluate/Grid.png', img_normalized.astype(np.uint8))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    print(Bevgrid.shape)



    # print(points.shape)
    # for line in window:
    #     print(' '.join(map(lambda x : str(round(x, 1)),line.tolist())))
    # # Debugging prints
    # print("Points shape:", points.shape)
    #
    # # Compute Euclidean distances for coloring
    # distances = np.linalg.norm(points, axis=1)
    # distances_norm = (distances - distances.min()) / (distances.max() - distances.min())
    #
    # # Map distances to colors
    # colors = plt.cm.plasma(distances_norm)[:, :3]
    #
    # # Convert to float32 (Open3D requirement)
    # points = points.astype(np.float32)
    # colors = colors.astype(np.float32)
    #
    # # Create Open3D point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    #
    # o3d.visualization.draw_geometries([pcd])