import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from models import compile_model
from data import VizData
from tools import SimpleLoss, get_batch_iou, denormalize_img
from tqdm import tqdm
import cv2


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False, confidence = 0.5, overlay = 0.5):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')
    ixes = valloader.dataset.ixes['frame']
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
            output_dir = f'./visualization/'
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
            
            imgLR        = np.hstack((imgL, imgR, colored_binimgs)).astype(np.uint8)
            evalimg      = np.hstack((pred, binimg, overlayed)).astype(np.uint8)
            # eval_colered = np.hstack((colored_binimgs, overlayed)).astype(np.uint8)


            finalimg = np.vstack((imgLR, evalimg)).astype(np.uint8)
            
            save_path = os.path.join(output_dir, index)
            cv2.imwrite(save_path, finalimg)
            print(f'Saved: {save_path}')
            # cv2.imshow('Evaluation', finalimg)
            # if cv2.waitKey(250) & 0xFF == ord('q'):
            #     break

    return {
        'loss': total_loss / len(valloader.dataset),
        'iou': total_intersect / total_union,
    }

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
    logdir='./runs_2'

    xbound=[-10.0, 10.0, 0.1]
    ybound=[-10.0, 10.0, 0.1]
    zbound=[-2.0, 5.0, 7.0]
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

    modelf = os.path.join(logdir, 'model84500.pt')


    model = compile_model(grid_conf, data_aug_conf, outC=1, is_aug=is_aug)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))


    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    val_info = get_val_info(model, valloader, loss_fn, device, False)
    print('VAL', val_info)
