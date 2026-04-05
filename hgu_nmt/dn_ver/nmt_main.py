# -*- coding: utf-8 -*-

# H. Choi, hchoi@handong.edu
from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import copy
import random
import numpy as np
import shutil
import gc
from distutils.dir_util import copy_tree

import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from libs.utils import timeSince, ids2words, save_checkpoint, CountDown, load_metrics,\
                         load_checkpoint
from libs.opts import ScheduledOptim, InverseRootSquareScheduler, optimizer_to

import nmt_const as Const
from nmt_model_tm import NormalNMT
from nmt_trans import multi_validation
from nmt_ensemble import Ensemble
import neptune # version 1.2.0


def train(model, optimizer, x_data, x_mask, y_data, y_mask, args,\
        update_flag, update_step):

    def get_grad(model):
        avg_grad = torch.tensor([torch.norm(p.grad).cpu().item() for p in \
                                model.parameters() if p.requires_grad])
        return torch.mean(avg_grad)

    total_trans_loss = 0

    # Bilingual Process
    model.train()
    (loss) = model(x_data, x_mask, y_data, y_mask, 'training')
    loss /= update_step
    loss.backward()

    total_trans_loss += loss.item()

    if update_flag:
        if args.grad_clip > 0.0:
            nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.zero_grad()
        optimizer.zero_grad()

    return total_trans_loss


def train_main(args, train_iter, valid_iter, rank, neptune_project):
    if rank == 0 and args.neptune == 1:
        if args.resume == 1:
            print("Load neptune exp of {}".format(args.load_neptune_exp_id))
            nep_proj = neptune.init_run(project=neptune_project,\
                                         with_id=args.load_neptune_exp_id)
        else:
            neptune_project_name = neptune_project
            nep_proj = neptune.init_run(project=neptune_project_name,\
                        source_files=['*.py','*.sh'])
            nep_proj['sys/name'] = args.subdir

    # model 
    if args.model == 'normal_nmt':
        model = NormalNMT(args=args)

    if args.opt_scheduled:
        optim_adam = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        optimizer = InverseRootSquareScheduler(optim_adam, args.lr, args.opt_start)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr) 

    if args.resume == 1:
        model, optimizer = load_checkpoint(args.save_dir, args.subdir, model, optimizer,\
                                            args.opt_scheduled, load_latest=True)

    print( sum( p.numel() for p in model.parameters() if p.requires_grad ) )
  
    model.cuda()
    optimizer_to(optimizer._optimizer, torch.device("cuda:{}".format(rank)))
    model_ddp = DDP(model, device_ids=[rank], dim=0) #, find_unused_parameters=True)

    dist.barrier()   
 
    best_bleu = -1
    trans_loss = 0  # Reset every args.print_every
    iloop = 0

    n_patience = torch.tensor(0).cuda()
    early_stop = False

    xy = args.src_lang + '2' + args.trg_lang

    start = time.time()

    print_every = args.print_every * args.update_step
    valid_every = args.valid_every * args.update_step
    valid_start = args.valid_every * args.update_step

    n_update = 0
    print("Training Starts..")
    for iloop, [x_data, x_mask, y_data, y_mask, idxes] in enumerate(train_iter):
        if early_stop == True:
            break

        x_data, x_mask, y_data, y_mask = x_data[0], x_mask[0], y_data[0], y_mask[0]

        if iloop % args.update_step == 0:
            update_flag = True
            n_update += 1
        else:
            update_flag = False

        tmp_trans_loss = train(model_ddp, optimizer, x_data, x_mask, y_data, y_mask, args,\
                     update_flag, args.update_step)

        trans_loss += tmp_trans_loss

        if rank == 0 and iloop % print_every == 0 and iloop > 0:
            trans_loss_avg = trans_loss/print_every
            trans_loss = 0  # Reset every args.print_every

            print("model : {} : {} iters | {} patience - {}".format(\
                                args.subdir, iloop, n_patience, timeSince(start)))
            print("Trans Loss  : {:.4f}".format(trans_loss_avg))

            if args.neptune == 1: nep_proj['train/'+xy+'_trans_loss'].log(trans_loss_avg)

        if iloop % valid_every == 0 and iloop >= valid_start:
            if rank == 0:
                if args.opt_scheduled:
                    save_list = {'iloop':iloop,\
                        'state_dict':model_ddp.module.state_dict(),\
                        'scheduler':optimizer.state_dict(),\
                        'optimizer':optim_adam.state_dict()} 
                else:
                    save_list = {'iloop':iloop,\
                        'state_dict':model_ddp.module.state_dict(),\
                         'optimizer':optimizer.state_dict()} 
                save_checkpoint(args.save_dir, args.subdir, save_list, args.n_checkpoint)

            dist.barrier()
            # Make ensemble model
            ensemble_save_list = Ensemble(args)

            # Load
            if args.model == 'normal_nmt':
                valid_model = NormalNMT(args=args, mean_batch=valid_iter.avg_rank_batch_size)

            valid_model.load_state_dict(ensemble_save_list['state_dict'], strict=False)
            valid_model.cuda()
            valid_model_ddp = DDP(valid_model, device_ids=[rank], dim=0)

            if iloop >= valid_start:
                bleu, valid_loss, _ = multi_validation(valid_model_ddp, valid_iter, args, rank)
                if rank == 0:
                    if args.neptune == 1: nep_proj['valid/'+xy+'_bleu'].log(bleu)
                    if args.neptune == 1: nep_proj['valid/'+xy+'_valid_loss'].log(valid_loss)

                    if bleu > best_bleu:
                        best_bleu = bleu
                        torch.save(ensemble_save_list, args.save_dir+args.subdir\
                                                        +'/'+xy+'.ensemble_model.best.pth')
                        torch.save(save_list, args.save_dir+args.subdir\
                                                        +'/'+xy+'.checkpoint.best.pth')
                        n_patience = torch.tensor(0).cuda()
                    else:
                        n_patience += 1
            valid_model.cpu()

        dist.broadcast(n_patience, 0)

        if n_patience >= args.patience:
            early_stop = True
        dist.barrier()
    print("Final running time : {}".format(timeSince(start)))

