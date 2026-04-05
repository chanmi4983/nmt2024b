# -*- coding: utf-8 -*-

# H. Choi, hchoi@handong.edu
from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import numpy as np
import re
from subprocess import Popen, PIPE

import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#k...
#import statistics as S
#k

from text_data import TextPairIterator
from libs.utils import timeSince, ids2words, unbpe
from libs.layers import CudaVariable
from libs.opts import ScheduledOptim

import nmt_const as Const
#from nmt_model_tm import Transformer
from nmt_model_scale import Transformer
#from nmt_model_tmqv import Transformer
#from nmt_model_tm import Transformer
#from nmt_mod_tm_init import Transformer
from nmt_trans import translate_file
#import neptune
import wandb

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def train(model, optimizer, x_data, x_mask, y_data, y_mask, y_lens, args):
    model.train()
    loss = model(x_data, x_mask, y_data, y_mask)
    optimizer.zero_grad()
    loss.backward()
    if args.grad_clip > 0.0:
        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
    optimizer.step()

    return loss.item()#, y_hat

###d###v
#def validation(model, x_data, x_mask, y_data, y_mask, y_lens, args):
 #   model.eval()
  #  loss = model(x_data, x_mask, y_data, y_mask)
   # return loss.item()#, y_hat
###d###^
def validation(model, x_d, x_m, y_d, y_m, y_l, args):
    model.eval()
    loss1 = model(x_d, x_m, y_d, y_m)

    return loss1.item()#, y_hat


def train_main(args):

    start = time.time()
    loss_total = 0  # Reset every args.print_every

#..............
    loss_total1 =0
#...............

    # model 
    model = Transformer(args=args).to(device) # Transformer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt_scheduled:
        optim_adam = optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)
        optimizer = ScheduledOptim(optim_adam, 2.0, args.dim_model, args.opt_start)
    else:
        optimizer = optim.Adam(params, lr=args.lr) 

    if args.resume:  # resume training
        file_name = args.save_dir + '/' + args.resume_file

        if os.path.exists(file_name):
            print('resume training from', file_name)
            chk_point = torch.load(file_name)
            model.load_state_dict(chk_point['state_dict'], strict=False)
            if args.opt_scheduled:
                optimizer.load_state_dict(chk_point['scheduler'])
                optim_adam.load_state_dict(chk_point['optimizer'])

    train_iter = TextPairIterator(args.train_src_file, args.train_trg_file,
            args.src_dict, args.trg_dict,
            batch_size=args.batch_size, maxlen=args.max_length,
            ahead=args.ahead, resume_num=args.resume_line, const_id=Const)

    print('training iteration starts...')
    loss_avg, prev_loss, patience = 0, 4000000, 0
    iter_count, len_x, len_y, px,py, tt = 0, [], [], [], [], []
    for x_data, x_mask, y_data, y_mask, y_lens, cur_line, iloop in train_iter:
        ##if iter_count < 600:
        ##    cx = x_data.shape[0]*x_data.shape[1] - np.count_nonzero(x_data)
        ##    cy = y_data.shape[0]*y_data.shape[1] - np.count_nonzero(y_data)
        ##    px.append((cx * 100)/(x_data.shape[0]*x_data.shape[1]))
        ##    py.append((cy * 100)/(y_data.shape[0]*y_data.shape[1]))
        ##    len_x.append(x_data.shape[0])
        ##    len_y.append(y_data.shape[0])
        ##    iter_count +=1
        ##elif iter_count == 600:
        ##    len_x.append(x_data.shape[0])
        ##    len_y.append(y_data.shape[0])
        ##    px.append((cx * 100)/(x_data.shape[0]*x_data.shape[1]))
        ##    py.append((cy * 100)/(y_data.shape[0]*y_data.shape[1]))
        ##    iter_count +=1
        ##    if args.wandb == 1:
        ##        mx = sum(len_x)/len(len_x)
        ##        my = sum(len_y)/len(len_y)
        ##        x_std = sum(px)/len(px)
        ##        y_std = sum(py)/len(py)
        ##        wandb.log({'avg_len_en': mx})
        ##        wandb.log({'avg_len_lu': my})
        ##        wandb.log({'perc_pad_en': x_std})
        ##        wandb.log({'perc_pad_lu': y_std})
        bb = time.time()
        loss = train(model, optimizer, x_data, x_mask, y_data, y_mask, y_lens, args)
        tt.append(time.time()-bb)
        iter_count += x_data.shape[1]
        loss_total += loss
        file_pre = args.save_dir + '/' + args.model_file +'_'+str(args.ahead)
        iter_count +=1
        if iter_count % 39428 == 0:
            tt_mean = sum(tt)/len(tt)
            wandb.log({'time_all': tt_mean})
        ##if iter_count % 650 == 0:
        ##    now = time.time()
        ##    s = now - start
        ##    ttime_it = s/iloop

        ##    wandb.log({'time_all': ttime_it})
        if iloop % args.print_every == 0:
            loss_avg = loss_total/args.print_every
            loss_total = 0
            print('%s: %d iters - %.4f %s' % (args.model_file, iloop, loss_avg, timeSince(start)))
            if args.wandb == 1:
                wandb.log({'loss': loss_avg})
   
        if iloop % args.valid_every == 0:
            if args.opt_scheduled:
                save_list = {'iloop':iloop, 'model':model, ':state_dict':model.state_dict(),
                        'scheduler':optimizer.state_dict(), 'optimizer':optim_adam.state_dict()} 
            else:
                save_list = {'iloop':iloop, 'model':model, 'state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict()} 
            torch.save(save_list, file_pre+'.pth')
            with open(file_pre+'.line', 'w') as lfp:
                lfp.write('cur_line: ' + str(cur_line)+'\n')

            if iloop >= args.valid_start:
                ###d###v
                v_loss_sum = []
                v_loss_avg=0
                
                #v_sumloss = 0
                #n1=0
                valid_iter = TextPairIterator(args.valid_src_file, args.valid_trg_file,args.src_dict, \
                        args.trg_dict, batch_size=args.batch_size, maxlen=args.max_length,ahead=1, resume_num=0, \
                        just_one_epoch=1, const_id=Const)
                
                #for x_data, x_mask, y_data, y_mask, y_lens, cur_l, il in valid_iter:
                for x_d, x_m, y_d, y_m, y_l, cur_l, il in valid_iter:
                   #vloss = validation(model, x_d, x_m, y_d, y_m,y_l,args)
                   vloss = validation(model, x_data, x_mask, y_data, y_mask, y_lens, args)
                   #v_sumloss += vloss
                   #n1+=1
                   v_loss_sum.append(vloss)
                    #count=len(vloss)
                v_loss_avg = sum(v_loss_sum)/len(v_loss_sum)
                #v_loss_avg = v_sumloss/n1
                #v_loss_avg = v_sumloss/len(v_sumloss)
                
                err = abs(v_loss_avg - prev_loss)
                if prev_loss < v_loss_avg or err < args.thresh:
                    patience += 1
                    if patience == args.patience:
                        print('Well done! we are done with training !!!')
                        break
                else:
                    prev_loss = v_loss_avg

                if args.wandb == 1:
                    wandb.log({'v_loss_avg': v_loss_avg})
                v_loss_avg=0
                ###d###^
                

                bleu_score = translate_file(model, args, valid=True)
                if os.path.exists(file_pre+'.bleu'):
                    with open(file_pre+'.bleu', 'r') as bfp:
                        lines = bfp.readlines()
                        prev_bleus = [float(bs.split()[1]) for bs in lines]
                else:
                    prev_bleus = [0] 
               
                if args.wandb == 1:
                    wandb.log({'bleu': bleu_score})

                if bleu_score >= np.max(prev_bleus):
                    torch.save(save_list, file_pre+'.best.pth')
                    with open(file_pre+'.line', 'w') as lfp:
                        lfp.write('cur_line: ' + str(cur_line)+'\n')

                mode = 'w' if iloop == args.valid_start else 'a'
                with open(file_pre+'.bleu', mode) as bfp:
                    bfp.write(str(iloop) + '\t' + str(bleu_score) + '\n') 
