import torch
import numpy as np
import os

from models import compile_model
from data import VizData
from tools import SimpleLoss, get_batch_iou, denormalize_img
from tqdm import tqdm
import cv2


def get_val_info_n_save_fig(model, valloader, loss_fn, device, use_tqdm=False, confidence = 0.5, overlay = 0.5):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')
    ixes = valloader.dataset.ixes['frame']
    valloader.dataset.cams_coverage = True
    valloader.dataset.ego = True
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch, index in zip(iter(loader), ixes):
            imgs, rots, trans, K, D, xi, binimgs, colored_binimgs = batch
            preds   = model(imgs.to(device), rots.to(device),
                            trans.to(device), K.to(device), D.to(device),
                            xi.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect    += intersect
            total_union        += union
            # saving eval prediction
            preds  = torch.sigmoid(preds)

            pred            = preds[0][0].detach().cpu().numpy()
            binimg          = binimgs[0][0].detach().cpu().numpy()
            colored_binimgs = colored_binimgs[0][0].detach().cpu().numpy()
            imgL            = imgs[0][0]
            imgR            = imgs[0][1]

            pred[pred <= confidence] = 0.0

            W, H = pred.shape

            # Define the output directory
            output_dir = f'./evaluation/'
            os.makedirs(output_dir, exist_ok=True)

            pred   = np.abs(cv2.resize(pred, (W * 3, H * 3)))
            binimg = np.abs(cv2.resize(binimg, (W * 3, H * 3)))

            imgL            = cv2.resize(np.array(denormalize_img(imgL)), (W * 3, H * 3))
            imgL            = cv2.cvtColor(imgL, cv2.COLOR_RGB2BGR)
            imgR            = cv2.resize(np.array(denormalize_img(imgR)), (W * 3, H * 3))
            imgR            = cv2.cvtColor(imgR, cv2.COLOR_RGB2BGR)
            colored_binimgs = cv2.resize(colored_binimgs, (W * 3, H * 3)).astype(np.uint8)

            pred      = cv2.normalize(pred,   None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            pred      = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            binimg    = cv2.normalize(binimg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            binimg    = cv2.cvtColor(binimg.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            overlayed = cv2.addWeighted(pred, overlay, colored_binimgs , 1, 0)

            ver_strip = np.ones((H*3, 3, 3)) * 255 #vertical white strip for better visualization
            hor_strip = np.ones((3, W * 3 * 3 + 6, 3)) * 255
            
            imgLR        = np.hstack((imgL,ver_strip, imgR, ver_strip, colored_binimgs)).astype(np.uint8)
            evalimg      = np.hstack((pred, ver_strip, binimg, ver_strip, overlayed)).astype(np.uint8)

            finalimg = np.vstack((imgLR, hor_strip, evalimg)).astype(np.uint8)
            
            save_path = os.path.join(output_dir, index)
            cv2.imwrite(save_path, finalimg)
            print(f'Saved: {save_path}')

    return {
        'loss': total_loss / len(valloader.dataset),
        'iou': total_intersect / total_union,
    }

def draw_fisheye_overlay(Px, Py, img, bev_img, overlay=1, background=0.85):
    H, W = img.shape[:2]
    canvas = np.zeros((W + 50, H + 50, 3), dtype=np.uint8)
    bev_img = np.transpose(bev_img, (1, 0, 2))[:, ::-1, ::-1]

    canvas[Py, Px, :]         = bev_img
    canvas[Py - 1, Px, :]     = bev_img
    canvas[Py, Px - 1, :]     = bev_img
    canvas[Py - 1, Px - 1, :] = bev_img
    canvas[Py, Px + 1, :]     = bev_img
    canvas[Py + 1, Px, :]     = bev_img
    canvas[Py + 1, Px + 1, :] = bev_img
    canvas[Py - 1, Px + 1, :] = bev_img
    canvas[Py + 1, Px - 1, :] = bev_img

    canvas = canvas[:H, :W, :]

    out_img = cv2.cvtColor(cv2.addWeighted(canvas, overlay, img, background, 0), cv2.COLOR_RGB2BGR)

    return out_img

def visualization(model, valloader, device, use_tqdm=False, confidence=0.5, overlay=0.5):
    model.eval()
    print('running visualization...')
    ixes = valloader.dataset.ixes['frame']
    valloader.dataset.cams_coverage = False
    valloader.dataset.ego = True
    loader = tqdm(valloader) if use_tqdm else valloader
    shift_matrix = valloader.dataset.shift_origin(x=0.81, y=0.32, z=0.9)
    fH, fW       = valloader.dataset.data_aug_conf['final_dim']
    H, W         = valloader.dataset.data_aug_conf['H'], valloader.dataset.data_aug_conf['H']
    bevH, bevW   = valloader.dataset.nx[:2]
    bev_max      = valloader.dataset.bev_max
    bev_min      = valloader.dataset.bev_min
    ratio        = max(fH/H, fW/W)

    cam_02 = valloader.dataset.cams['image_02']
    cam_03 = valloader.dataset.cams['image_03']


    xs = np.linspace(bev_min, bev_max, bevW)
    ys = np.linspace(bev_min, bev_max, bevH)

    xv, yv = np.meshgrid(xs, ys)
    zv     = np.zeros_like(xv)
    ones   = np.ones_like(xv)

    idxes_imu  = np.stack([xv, yv, zv, ones], axis=-1)
    idxes_imuL = idxes_imu[:bevW// 2, :, :].reshape(-1, 4).T
    idxes_imuR = idxes_imu[bevW// 2:, :, :].reshape(-1, 4).T
    idxes_egoL = shift_matrix @ idxes_imuL
    idxes_egoR = shift_matrix @ idxes_imuR

    idxes_cam_02 = np.linalg.inv(cam_02.camToPose) @ idxes_egoL
    idxes_cam_03 = np.linalg.inv(cam_03.camToPose) @ idxes_egoR
    idxes_cam_02 = idxes_cam_02[:3]  # Normalize homogeneous coordinates
    idxes_cam_03 = idxes_cam_03[:3]

    Px_cam_02, Py_cam_02, _ = cam_02.cam2image(idxes_cam_02)
    Px_cam_03, Py_cam_03, _ = cam_03.cam2image(idxes_cam_03)
    Px_cam_02, Py_cam_02    = Px_cam_02*ratio, Py_cam_02*ratio
    Px_cam_03, Py_cam_03    = Px_cam_03*ratio, Py_cam_03*ratio

    Px_cam_02, Py_cam_02    = Px_cam_02.astype(np.uint16).reshape(bevH // 2, bevW), Py_cam_02.astype(np.uint16).reshape(bevH // 2, bevW)
    Px_cam_03, Py_cam_03    = Px_cam_03.astype(np.uint16).reshape(bevH // 2, bevW), Py_cam_03.astype(np.uint16).reshape(bevH // 2, bevW)


    with torch.no_grad():
        for batch, index in zip(iter(loader), ixes):
            imgs, rots, trans, K, D, xi, binimgs, colored_binimgs = batch

            preds = model(imgs.to(device), rots.to(device),
                          trans.to(device), K.to(device), D.to(device),
                          xi.to(device))

            preds = torch.sigmoid(preds)

            pred = preds[0][0].detach().cpu().numpy()
            colored_binimgs = colored_binimgs[0][0].detach().cpu().numpy().squeeze()
            img_02 = np.array(denormalize_img(imgs[0][0]))
            img_03 = np.array(denormalize_img(imgs[0][1]))

            pred[pred <= confidence] = 0.0

            pred = cv2.normalize(pred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            pred = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            out_img_02_colored = draw_fisheye_overlay(Px_cam_02, Py_cam_02, img_02, colored_binimgs[:, :bevW//2, :])
            out_img_03_colored = draw_fisheye_overlay(Px_cam_03, Py_cam_03, img_03, colored_binimgs[:, bevW//2:, :])
            out_img_02_pred    = draw_fisheye_overlay(Px_cam_02, Py_cam_02, img_02, pred[:, :bevW//2, :], overlay=0.4, background=1)
            out_img_03_pred    = draw_fisheye_overlay(Px_cam_03, Py_cam_03, img_03, pred[:, bevW//2:, :], overlay=0.4, background=1)

            # Define the output directory
            output_dir = f'./visualization/'
            os.makedirs(output_dir, exist_ok=True)

            pred               = cv2.resize(pred, (bevW * 3, bevH * 3))
            colored_binimgs    = cv2.resize(colored_binimgs, (bevW * 3, bevH * 3)).astype(np.uint8)
            out_img_02_colored = cv2.resize(out_img_02_colored, (bevW * 3, bevH * 3))
            out_img_03_colored = cv2.resize(out_img_03_colored, (bevW * 3, bevH * 3))
            out_img_02_pred    = cv2.resize(out_img_02_pred, (bevW * 3, bevH * 3))
            out_img_03_pred    = cv2.resize(out_img_03_pred, (bevW * 3, bevH * 3))


            overlayed = cv2.addWeighted(pred, overlay, colored_binimgs, 1, 0)

            ver_strip = np.ones((bevH * 3, 3, 3)) * 255
            hor_strip = np.ones((3, bevW * 3 * 3 + 6, 3)) * 255

            imgLR = np.hstack((out_img_02_colored, ver_strip, out_img_03_colored, ver_strip, colored_binimgs)).astype(np.uint8)
            evalimg = np.hstack((out_img_02_pred, ver_strip, out_img_03_pred, ver_strip, overlayed)).astype(np.uint8)



            finalimg = np.vstack((imgLR, hor_strip, evalimg)).astype(np.uint8)

            save_path = os.path.join(output_dir, index)
            cv2.imwrite(save_path, finalimg)
            print(f'\rSaved: {save_path}', end='', flush=True)

    return
def make_video(image_folder, output_name, fps, scale = 1):

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_name, fourcc, fps, (width * scale, height * scale))

    for i, image in enumerate(images):
        img_path = os.path.join(image_folder, image)
        print(f'\rSaving video ... Loading: {img_path}', end='', flush=True)

        frame = cv2.imread(img_path)
        upscaled_frame = cv2.resize(frame, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
        video_writer.write(upscaled_frame)

    video_writer.release()
    print("Video saved as", output_name)

if __name__ == '__main__':
    nepochs=10,
    gpuid=0

    H=1400
    W=1400
    resize_lim=(0.193, 0.225)
    final_dim=(512, 512)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True
    ncams=2
    max_grad_norm=5.0
    pos_weight=2.13
    logdir='./runs'

    xbound=[-10.0, 10.0, 0.1]
    ybound=[-10.0, 10.0, 0.1]
    zbound=[  2.0, -2.0, 4.0]
    dbound=[1.0, 14.0, 0.325]

    is_aug=False

    bsz=1
    nworkers=10
    lr=5e-4
    weight_decay=1e-7

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['image_02', 'image_03'],
                    'Ncams': ncams,
                }
    valdata = VizData(is_train=False, data_aug_conf=data_aug_conf,
                     grid_conf=grid_conf, is_aug=is_aug)

    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    modelf = os.path.join(logdir, 'model61500.pt')


    model = compile_model(grid_conf, data_aug_conf, outC=1, is_aug=is_aug)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))


    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    # val_info = get_val_info_n_save_fig( model, valloader, loss_fn, device, False)
    # print('VAL', val_info)

    visualization(model, valloader, device, False)
    make_video('./visualization/', 'Visualization.mp4', 6)
