"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from models import compile_model
from data import compile_data
from tools import SimpleLoss, get_batch_iou, get_val_info


def train(nepochs=10,
            gpuid=0,

            H=1400, W=1400,
            resize_lim=(0.193, 0.225),
            final_dim=(512, 512),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=2,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',

            xbound=[-10.0, 10.0, 0.1],
            ybound=[-10.0, 10.0, 0.1],
            zbound=[-2.0, 5.0, 7.0],
            dbound=[1.0, 14.0, 0.325],

            is_aug=False,

            bsz=8,
            nworkers=10,
            lr=5e-4,
            weight_decay=1e-7,
            ):
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
    trainloader, valloader = compile_data(data_aug_conf=data_aug_conf, is_aug=is_aug,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1, is_aug=is_aug)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 500

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, K, D, xi, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    K.to(device),
                    D.to(device),
                    xi.to(device),
                    )
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(f"Epoch #{epoch} Step #{counter} Loss : {loss.item()}")
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device, counter=counter)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

if __name__ == '__main__':
    train()