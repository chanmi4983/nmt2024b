# -*- coding: utf-8 -*-

# H. Choi, hchoi@handong.edu
from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import numpy as np

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
  
def train(model, optimizer, x_data, x_mask, y_data, y_mask, grad_clip):
    model.train()
    loss,enc_entropies, dec_self_entropies, dec_cross_entropies = model(x_data, x_mask, y_data, y_mask)
    optimizer.zero_grad()
    loss.backward()
    #loss.backward(retain_graph=True) post11에서 사용했음
    if grad_clip > 0.0:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item(),enc_entropies, dec_self_entropies, dec_cross_entropies #, y_hat

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
    
    enc_entropy_total = [0.0] * 6  # Encoder 레이어 6개의 엔트로피 누적합
    dec_self_entropy_total = [0.0] * 6  # Decoder Self-Attention 레이어 6개의 엔트로피 누적합
    dec_cross_entropy_total = [0.0] * 6  # Decoder Cross-Attention 레이어 6개의 엔트로피 누적합

    for x_data, x_mask, y_data, y_mask, cur_line, iloop in train_iter:

        loss, enc_entropies, dec_self_entropies, dec_cross_entropies = train(model, optimizer, x_data, x_mask, y_data, y_mask, args.grad_clip)
        loss_total += loss
        for i in range(6):  # 레이어 수만큼 반복
            enc_entropy_total[i] += enc_entropies[i].mean().item()
            dec_self_entropy_total[i] += dec_self_entropies[i].mean().item()
            dec_cross_entropy_total[i] += dec_cross_entropies[i].mean().item()

        
    
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
            
            enc_entropy_avg = [total / args.log_interval for total in enc_entropy_total]
            dec_self_entropy_avg = [total / args.log_interval for total in dec_self_entropy_total]
            dec_cross_entropy_avg = [total / args.log_interval for total in dec_cross_entropy_total]
            
            loss_total = 0
            enc_entropy_total = [0.0] * 6
            dec_self_entropy_total = [0.0] * 6
            dec_cross_entropy_total = [0.0] * 6
            print('%s: %d iters - %.4f %s' % (args.model_file, iloop, loss_avg, timeSince(start)))

            # if args.logging == 1: 
            #     wandb.log({"train_loss": loss_avg})
            #     param = model.state_dict()
            #     wandb.log({
            #         "encoder.pos_enc.scale": param['encoder.pos_enc.scale'].item(),
            #         "decoder.pos_enc.scale": param['decoder.pos_enc.scale'].item(),
            #         **{f"encoder.attn_temp{i}": param[f'encoder.layer_stack.{i}.self_attn.sdp_attn.temper'].item() for i in range(6)},
            #         **{f"decoder.attn_temp{i}": param[f'decoder.layer_stack.{i}.self_attn.sdp_attn.temper'].item() for i in range(6)},
            #         **{f"ende.attn_temp{i}": param[f'decoder.layer_stack.{i}.enc_attn.sdp_attn.temper'].item() for i in range(6)},
            # })
            if args.logging == 1:
                #wandb.log({"train_loss": loss_avg})
                param = model.state_dict()
                wandb.log({
                    "train_loss": loss_avg,
                    **{f"encoder_entropy_layer_{i+1}_avg": enc_entropy_avg[i] for i in range(6)},
                    **{f"decoder_self_entropy_layer_{i+1}_avg": dec_self_entropy_avg[i] for i in range(6)},
                    **{f"decoder_cross_entropy_layer_{i+1}_avg": dec_cross_entropy_avg[i] for i in range(6)},
                    "encoder.pos_enc.scale": param['encoder.pos_enc.scale'].item(),
                    "decoder.pos_enc.scale": param['decoder.pos_enc.scale'].item(),
                    "encoder.word_embedding.scale": param['encoder.src_emb.weight'].norm().item(),
                    "decoder.word_embedding.scale": param['decoder.trg_emb.weight'].norm().item(),
                    #**{f"encoder.attn_temp{i}": param[f'encoder.layer_stack.{i}.self_attn.sdp_attn.temper'].item() for i in range(6)},
                    #**{f"decoder.attn_temp{i}": param[f'decoder.layer_stack.{i}.self_attn.sdp_attn.temper'].item() for i in range(6)},
                    #**{f"ende.attn_temp{i}": param[f'decoder.layer_stack.{i}.enc_attn.sdp_attn.temper'].item() for i in range(6)},
                    **{f"encoder.ff_layer.norm{i}": param[f'encoder.layer_stack.{i}.ff_layer.layer_norm.weight'].norm().item() for i in range(6)},
                    **{f"decoder.ff_layer.norm{i}": param[f'decoder.layer_stack.{i}.ff_layer.layer_norm.weight'].norm().item() for i in range(6)},
})


# End of File

