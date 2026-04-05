# -*- coding: utf-8 -*-

# H. Choi, hchoi@handong.edu
from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import numpy as np

import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from text_data import TextPairIterator, read_dict
from libs.utils import timeSince 
from libs.layers import CudaVariable
from libs.opts import ScheduledOptim, InverseRootSquareScheduler

import nmt_const as Const
from nmt_model_tm import Transformer, myEmbedding
from nmt_trans_main import translate_file
import wandb

import pickle as pkl
from collections import OrderedDict

import pdb

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")
  
def train(model, optimizer, x_data, x_mask, y_data, y_mask, grad_clip):
    model.train()
    loss = model(x_data, x_mask, y_data, y_mask)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    if grad_clip > 0.0:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item() #, y_hat

def train_main(args):
    if args.logging == 1:
        wandb.init(project="nmt", config=args, settings=wandb.Settings(code_dir="."))
        # resume wandb
        # resume wandb 하는 곳
        #wandb.init(project="nmt", config=args, settings=wandb.Settings(code_dir="."),resume="allow", id="3zgh46o0")
    # model 
    src_dict = read_dict(args.src_dict)
    trg_dict = read_dict(args.trg_dict)
    src_words_n = len(src_dict) 
    trg_words_n = len(trg_dict) 
    model = Transformer(src_words_n, trg_words_n, args=args).to(device) # Transformer
        
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt_scheduled:
        optim_adam = optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)
        optimizer = ScheduledOptim(optim_adam, 2.0, args.dim_wemb, args.opt_start)
        #optimizer = InverseRootSquareScheduler(optim_adam, args.lr, args.opt_start)
    else:
        optimizer = optim.Adam(params, lr=args.lr) 

    file_pre = args.save_dir + '/' + args.model_file

    if args.resume:  # resume training
        file_name = args.save_dir + '/' + args.resume_file
        file_pre = args.save_dir + '/' + args.resume_file
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
                         ahead=1000, resume_num=args.resume_line, const_id=Const) #cur_line 가져와서 resume line에 넣기

     
    start = time.time()
    loss_total = 0  # Reset every args.log_interval

    print('training iteration starts...')
    if args.logging == 1: wandb.watch(model, log_freq=args.log_interval)
    
    for x_data, x_mask, y_data, y_mask, cur_line, iloop in train_iter:

        loss = train(model, optimizer, x_data, x_mask, y_data, y_mask, args.grad_clip)
        loss_total += loss
    
        if iloop % args.valid_interval == 0 and iloop >= args.valid_start: #디버깅때만 낮춰서 
            if args.opt_scheduled:
                save_list = {'iloop':iloop, 'model':model, 'state_dict':model.state_dict(),
                        'scheduler':optimizer.state_dict(), 'optimizer':optim_adam.state_dict()} #cur_line 넣기
    
            else:
                save_list = {'iloop':iloop, 'model':model, 'state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict()} #cur_line 넣기
            torch.save(save_list, file_pre+'.pth')
            with open(file_pre+'.line', 'w') as lfp:
                lfp.write('cur_line: ' + str(cur_line)+'\n')

            bleu_score = translate_file(model, args.valid_src_file+args.valid_post, args.src_dict, args.valid_trg_file, args.trg_dict, '', args, valid=True)

            prev_bleus = [0] 
            if os.path.exists(file_pre+'.bleu'):
                with open(file_pre+'.bleu', 'r') as bfp:
                    lines = bfp.readlines()
                    prev_bleus = [float(bs.split()[1]) for bs in lines]
            if bleu_score >= np.max(prev_bleus): # save the best model
                torch.save(save_list, file_pre+'.best.pth')
                with open(file_pre+'.line', 'w') as lfp:
                    lfp.write('cur_line: ' + str(cur_line)+'\n')

            mode = 'w' if iloop == args.valid_start else 'a'
            with open(file_pre+'.bleu', mode) as bfp: # keep the bleu scores
                bfp.write(str(iloop) + '\t' + str(bleu_score) + '\n') 

            if args.logging == 1: wandb.log({"bleu": bleu_score})

        if iloop % args.log_interval == 0:
            loss_avg = loss_total/args.log_interval
            loss_total = 0
            print('%s: %d iters - %.4f %s' % (args.model_file, iloop, loss_avg, timeSince(start)))

            if args.logging == 1: wandb.log({"train_loss": loss_avg})

# End of File

