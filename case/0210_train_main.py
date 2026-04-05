# -*- coding: utf-8 -*-

# H. Choi, hchoi@handong.edu
from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import numpy as np
import random

import torch  # type: ignore
import torch.nn as nn # type: ignore
from torch import optim # type: ignore
import torch.nn.functional as F # type: ignore

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

def validation(model, x_d, x_m, y_d, y_m, args):
   model.eval()
   loss1 = model(x_d, x_m, y_d, y_m)
   return loss1.item()

def train(model, optimizer, x_data, x_mask, y_data, y_mask, grad_clip):
   model.train()
   loss = model(x_data, x_mask, y_data, y_mask)
   
   optimizer.zero_grad()
   loss.backward()
   
   if grad_clip > 0.0:
       nn.utils.clip_grad_value_(model.parameters(), grad_clip)
       
   optimizer.step() # update parameters
   
   return loss.item()

def train_main(args):
   if args.logging == 1:
       wandb.init(project="nmt", config=args, settings=wandb.Settings(code_dir="."))

   # model initialization
   src_dict = read_dict(args.src_dict)
   trg_dict = read_dict(args.trg_dict)
   src_words_n = len(src_dict) 
   trg_words_n = len(trg_dict) 
   model = Transformer(src_words_n, trg_words_n, args=args).to(device)
       
   # optimizer setup
   params = filter(lambda p: p.requires_grad, model.parameters())
   if args.opt_scheduled:
       optim_adam = optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)
       optimizer = ScheduledOptim(optim_adam, 2.0, args.dim_wemb, args.opt_start)
   else:
       optimizer = optim.Adam(params, lr=args.lr) 

   file_pre = args.save_dir + '/' + args.model_file

   # resume training if needed
   if args.resume:
       file_name = args.save_dir + '/' + args.resume_file
       file_pre = args.save_dir + '/' + args.resume_file
       if os.path.exists(file_name):
           print('resume training from', file_name)
           chk_point = torch.load(file_name)
           model.load_state_dict(chk_point['state_dict'], strict=False)
           if args.opt_scheduled:
               optimizer.load_state_dict(chk_point['scheduler'])
               optim_adam.load_state_dict(chk_point['optimizer'])

   # Training data 로드
   with open(args.train_src_file, 'r') as f:
       train_src_lines = f.readlines()
   with open(args.train_trg_file, 'r') as f:
       train_trg_lines = f.readlines()
   
   # validation용 데이터 랜덤 추출 (10%)
   valid_size = int(len(train_src_lines) * 0.1)
   valid_indices = random.sample(range(len(train_src_lines)), valid_size)
   
   # validation 데이터 추출
   valid_src = [train_src_lines[i] for i in valid_indices]
   valid_trg = [train_trg_lines[i] for i in valid_indices]
   
   # 임시 파일로 저장
   temp_valid_src_file = args.save_dir + '/temp_valid_src'
   temp_valid_trg_file = args.save_dir + '/temp_valid_trg'
   with open(temp_valid_src_file, 'w') as f:
        f.writelines(valid_src)
   with open(temp_valid_trg_file, 'w') as f:
        f.writelines(valid_trg)
   # Training iterator (전체 데이터 사용)
   train_iter = TextPairIterator(args.train_src_file, args.train_trg_file, 
                       args.src_dict, args.trg_dict,
                       batch_size=args.batch_size, maxlen=args.max_length,
                       ahead=1000, resume_num=args.resume_line, const_id=Const)
   
   # Validation iterator (추출된 10% 데이터 사용)
   valid_iter = TextPairIterator(temp_valid_src_file, temp_valid_trg_file,
                              args.src_dict, args.trg_dict,
                              batch_size=args.batch_size, maxlen=args.max_length,
                              ahead=1, resume_num=0,
                              just_one_epoch=1, const_id=Const)

   start = time.time()
   loss_total = 0
   v_loss_avg = 0
   
   print('training iteration starts...')
   if args.logging == 1: 
       wandb.watch(model, log_freq=args.log_interval)

   for x_data, x_mask, y_data, y_mask, cur_line, iloop in train_iter:
       # Train step
       loss = train(model, optimizer, x_data, x_mask, y_data, y_mask, args.grad_clip)
       loss_total += loss

       # Validation step
       if iloop % args.valid_interval == 0 and iloop >= args.valid_start:
           print(f"Starting validation at iteration {iloop}")
           v_loss_sum = []
           
           with torch.no_grad():
               for x_d, x_m, y_d, y_m, cur_l, il in valid_iter:
                   vloss = validation(model, x_d, x_m, y_d, y_m, args)
                   v_loss_sum.append(vloss)
                   #print(f"Validation batch loss: {vloss}")

           model.train()
           if len(v_loss_sum) > 0:
               v_loss_avg = sum(v_loss_sum)/len(v_loss_sum)
               print(f"Validation average loss: {v_loss_avg}")

           os.remove(temp_valid_src_file)
           os.remove(temp_valid_trg_file)
           
           # Save model
           if args.opt_scheduled:
               save_list = {'iloop':iloop, 'model':model, 'state_dict':model.state_dict(),
                       'scheduler':optimizer.state_dict(), 'optimizer':optim_adam.state_dict()}
           else:
               save_list = {'iloop':iloop, 'model':model, 'state_dict':model.state_dict(),
                       'optimizer':optimizer.state_dict()}
                       
           torch.save(save_list, file_pre+'.pth')
           with open(file_pre+'.line', 'w') as lfp:
               lfp.write('cur_line: ' + str(cur_line)+'\n')

           # Calculate BLEU score
           bleu_score = translate_file(model, args.train_src_file, 
                                     args.src_dict, args.train_trg_file, 
                                     args.trg_dict, '', args, valid=True)

           # Save best model based on BLEU score
           prev_bleus = [0] 
           if os.path.exists(file_pre+'.bleu'):
               with open(file_pre+'.bleu', 'r') as bfp:
                   lines = bfp.readlines()
                   prev_bleus = [float(bs.split()[1]) for bs in lines]
                   
           if bleu_score >= np.max(prev_bleus):
               torch.save(save_list, file_pre+'.best.pth')
               with open(file_pre+'.line', 'w') as lfp:
                   lfp.write('cur_line: ' + str(cur_line)+'\n')

           mode = 'w' if iloop == args.valid_start else 'a'
           with open(file_pre+'.bleu', mode) as bfp:
               bfp.write(str(iloop) + '\t' + str(bleu_score) + '\n') 

           if args.logging == 1: 
               wandb.log({"0_bleu": bleu_score})

       # Logging
       if iloop % args.log_interval == 0:
           loss_avg = loss_total/args.log_interval
           print('%s: %d iters - %.4f %s' % (args.model_file, iloop, loss_avg, timeSince(start)))

           if args.logging == 1:
               log_dict = {
                   "train_loss": loss_avg,
                   "valid_loss": v_loss_avg
               }
               wandb.log(log_dict)
               
           loss_total = 0

# End of File