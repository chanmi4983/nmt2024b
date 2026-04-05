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
    loss,enc_entropy_list, self_entropy_list, cross_entropy_list= model(x_data, x_mask, y_data, y_mask)
    #loss,enc_entropy, dec_self_entropy, dec_cross_entropy, enc_layer_entropies, dec_self_layer_entropies, dec_cross_layer_entropies = model(x_data, x_mask, y_data, y_mask)
    
    #print("Encoder entropies shape:", enc_entropy.shape)
    #print("Decoder self entropies shape:", dec_self_entropy.shape)
    #print("Decoder cross entropies shape:", dec_cross_entropy.shape)
    
    optimizer.zero_grad()
    loss.backward()
    #loss.backward(retain_graph=True) post11에서 사용했음
    
    if grad_clip > 0.0:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        
    optimizer.step() # update parameters
    
    #return loss.item(), enc_entropy, dec_self_entropy, dec_cross_entropy ,enc_layer_entropies, dec_self_layer_entropies, dec_cross_layer_entropies#, y_hat
    return loss.item(),enc_entropy_list, self_entropy_list, cross_entropy_list #, y_hat

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
    
    enc_layer_entropies_total = [0] * 6
    dec_self_layer_entropies_total = [0] * 6
    dec_cross_layer_entropies_total = [0] * 6
    
    enc_entropy_total=0
    dec_self_entropy_total=0
    dec_cross_entropy_total=0
    
    print('training iteration starts...')
    if args.logging == 1: wandb.watch(model, log_freq=args.log_interval)

    for x_data, x_mask, y_data, y_mask, cur_line, iloop in train_iter:

        loss, enc_entropy_list, self_entropy_list, cross_entropy_list= train(model, optimizer, x_data, x_mask, y_data, y_mask, args.grad_clip)
        #loss, enc_entropy, dec_self_entropy, dec_cross_entropy, enc_layer_entropies, dec_self_layer_entropies, dec_cross_layer_entropies= train(model, optimizer, x_data, x_mask, y_data, y_mask, args.grad_clip)
        loss_total += loss
        
        enc_entropy_total += sum(enc_entropy_list)  # 전체 엔트로피 합산
        dec_self_entropy_total += sum(self_entropy_list)  # 전체 셀프 엔트로피 합산
        dec_cross_entropy_total += sum(cross_entropy_list)  # 전체 크로스 엔트로피 합산
        
        for i in range(6):
            enc_layer_entropies_total[i] += enc_entropy_list[i]
            dec_self_layer_entropies_total[i] += self_entropy_list[i]
            dec_cross_layer_entropies_total[i] += cross_entropy_list[i]
        
        #print(f"enc_layer_entropies length: {len(enc_layer_entropies)}")  # 리스트 길이 확인
        #print(f"dec_self_layer_entropies length: {len(dec_self_layer_entropies)}")  # 리스트 길이 확인
        #print(f"dec_cross_layer_entropies length: {len(dec_cross_layer_entropies)}")  # 리스트 길이 확인
        
        # for i in range(6):
        #     enc_layer_entropies_total[i] += enc_layer_entropies[i].detach()
        #     dec_self_layer_entropies_total[i] += dec_self_layer_entropies[i].detach()
        #     dec_cross_layer_entropies_total[i] += dec_cross_layer_entropies[i].detach()
            

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

            if args.logging == 1: wandb.log({"0_bleu": bleu_score})

        if iloop % args.log_interval == 0:
            loss_avg = loss_total/args.log_interval
            
            total_avg_enc_entropy = enc_entropy_total / args.log_interval
            total_avg_dec_self_entropy = dec_self_entropy_total / args.log_interval
            total_avg_dec_cross_entropy = dec_cross_entropy_total / args.log_interval
            
            avg_enc_entropy = [enc_layer_entropies_total[i] / args.log_interval for i in range(6)]
            avg_dec_self_entropy = [dec_self_layer_entropies_total[i] / args.log_interval for i in range(6)]
            avg_dec_cross_entropy = [dec_cross_layer_entropies_total[i] / args.log_interval for i in range(6)]
            
            # avg_enc_layer_entropies = {f"encoder.layer_{i+1}_entropy": enc_layer_entropies_total[i] / args.log_interval for i in range(6)}
            # avg_dec_self_layer_entropies = {f"decoder.self_layer_{i+1}_entropy": dec_self_layer_entropies_total[i] / args.log_interval for i in range(6)}
            # avg_dec_cross_layer_entropies = {f"decoder.cross_layer_{i+1}_entropy": dec_cross_layer_entropies_total[i] / args.log_interval for i in range(6)}

 
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
                    "0_train_loss": loss_avg,
                    
                    "0_Total_avg_Encoder Entropy": total_avg_enc_entropy, #5000번마다 평균값을 보여줌
                    "0_Total_avg_Decoder Self Entropy": total_avg_dec_self_entropy,
                    "0_Total_avg_Decoder Cross Entropy": total_avg_dec_cross_entropy,
                    
                    **{f"encoder_layer_{i}_entropy": avg_enc_entropy[i] for i in range(6)},
                    **{f"dec_self_layer_{i}_entropy": avg_dec_self_entropy[i] for i in range(6)},
                    **{f"dec_cross_layer_{i}_entropy": avg_dec_cross_entropy[i] for i in range(6)},
                    
                    #**avg_enc_layer_entropies,
                    #**avg_dec_self_layer_entropies,
                    #**avg_dec_cross_layer_entropies,
                    # "encoder.pos_enc.scale": param['encoder.pos_enc.scale'].item(),
                    # "decoder.pos_enc.scale": param['decoder.pos_enc.scale'].item(),
                    # "encoder.word_embedding.scale": param['encoder.src_emb.weight'].norm().item(),
                    # "decoder.word_embedding.scale": param['decoder.trg_emb.weight'].norm().item(),
                    
                    #**{f"encoder.entropy_param{i}": param[f'encoder.layer_stack.{i}.entropy'].item() for i in range(6)},#각 레이어의 entropy_param
                    #**{f"decoder.self_entropy_param{i}": param[f'decoder.layer_stack.{i}.self_entropy'].item() for i in range(6)},
                    #**{f"decoder.cross_entropy_param{i}": param[f'decoder.layer_stack.{i}.cross_entropy'].item() for i in range(6)},
                    
                    **{f"encoder.attn_temp{i}": param[f'encoder.layer_stack.{i}.self_attn.sdp_attn.temper'].item() for i in range(6)},
                    **{f"decoder.attn_temp{i}": param[f'decoder.layer_stack.{i}.self_attn.sdp_attn.temper'].item() for i in range(6)},
                    
                    # **{f"ende.attn_temp{i}": param[f'decoder.layer_stack.{i}.enc_attn.sdp_attn.temper'].item() for i in range(6)},
                    #**{f"encoder.ff_layer.norm{i}": param[f'encoder.layer_stack.{i}.ff_layer.layer_norm.weight'].norm().item() for i in range(6)},
                    #**{f"decoder.ff_layer.norm{i}": param[f'decoder.layer_stack.{i}.ff_layer.layer_norm.weight'].norm().item() for i in range(6)},
                })
                
            loss_total = 0

            enc_layer_entropies_total = [0] * 6
            dec_self_layer_entropies_total = [0] * 6
            dec_cross_layer_entropies_total = [0] * 6
            
            enc_entropy_total = 0
            dec_self_entropy_total = 0
            dec_cross_entropy_total = 0
            

# End of File

