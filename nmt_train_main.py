# -*- coding: utf-8 -*-

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
# [수정] MultiHeadAttention, FeedForward 클래스를 임포트합니다.
from nmt_model_tm import Transformer, MultiHeadAttention, FeedForward, myEmbedding
from nmt_trans_main import translate_file
import wandb

import pickle as pkl
from collections import OrderedDict

import pdb

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(model, optimizer, x_data, x_mask, y_data, y_mask, grad_clip, update_flag, update_step):
    model.train()
    loss = model(x_data, x_mask, y_data, y_mask)
    loss /= update_step
    
    loss.backward()
    
    if update_flag:
        if grad_clip > 0.0:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()


def train_main(args):
    if args.logging == 1:
        wandb.init(project="nmt", config=args, settings=wandb.Settings(code_dir="."))
        log = wandb.log
    else:
        log = lambda *a, **k: None

    # model 
    src_dict = read_dict(args.src_dict)
    trg_dict = read_dict(args.trg_dict)
    src_words_n = len(src_dict)
    trg_words_n = len(trg_dict)
    model = Transformer(src_words_n, trg_words_n, args=args).to(device)

    # [수정] Cross-Attention ('enc_attn')을 올바르게 인식하도록 수정
    def _pretty_name(name: str):
        try:
            parts = name.split('.')
            if len(parts) < 4: return None
            
            comp, layer_idx_str, sub = parts[0], parts[2], parts[3]
            if comp not in ('encoder', 'decoder'): return None
            
            if 'self_attn' in sub: kind = 'SelfAttn'
            elif 'enc_attn' in sub or 'enc_dec' in sub or 'cross' in sub: kind = 'CrossAttn'
            elif sub.startswith('ff'): kind = 'FFN'
            else: return None
            
            tag = 'Enc' if comp == 'encoder' else 'Dec'
            return f'{tag}_{kind}_L{layer_idx_str}'
        except IndexError:
            return None

    log_name_map = {}
    grad_stats = {}
    last_known = {}
    init_grad = {}

    for name, module in model.named_modules():
        pretty = _pretty_name(name)
        if pretty is None:
            continue
        log_name_map[name] = pretty
        grad_stats[name] = []

    def _make_mid_hook(layer_name):
        def _fwd(mod, inp, out):
            outs = out if isinstance(out, (tuple, list)) else (out,)
            for o in outs:
                if torch.is_tensor(o) and getattr(o, 'requires_grad', False):
                    o.register_hook(lambda g: grad_stats[layer_name].append(g.norm().item()))
            return None
        return _fwd

    for name, module in model.named_modules():
        if name in log_name_map:
            module.register_forward_hook(_make_mid_hook(name))

    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt_scheduled:
        optim_adam = optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)
        optimizer = ScheduledOptim(optim_adam, 2.0, args.dim_wemb, args.opt_start)
    else:
        optimizer = optim.Adam(params, lr=args.lr)

    file_pre = args.save_dir + '/' + args.model_file

    if args.resume:
        file_name = args.save_dir + '/' + args.resume_file
        if os.path.exists(file_name):
            print('resume training from', file_name)
            chk_point = torch.load(file_name)
            model.load_state_dict(chk_point['state_dict'], strict=False)
            if args.opt_scheduled and 'scheduler' in chk_point and 'optimizer' in chk_point:
                optimizer.load_state_dict(chk_point['scheduler'])
                optim_adam.load_state_dict(chk_point['optimizer'])

    train_iter = TextPairIterator(
        args.train_src_file, args.train_trg_file,
        args.src_dict, args.trg_dict,
        batch_size=args.batch_size, maxlen=args.max_length,
        ahead=1000, resume_num=args.resume_line, const_id=Const
    )

    start = time.time()
    loss_total = 0
    if args.logging == 1:
        wandb.watch(model, log_freq=args.log_interval)

    update_step = args.update_step
    log_interval = args.log_interval * update_step
    valid_interval = args.valid_interval * update_step
    valid_start = args.valid_start * update_step
    got_init = False
    n_update = 0

    print('training iteration starts...')

    try:
        for x_data, x_mask, y_data, y_mask, cur_line, iloop in train_iter:
            if not got_init and args.logging == 1:
                handles = []
                def _make_init_hook(layer_name):
                    def _hook(g): init_grad[layer_name] = g.norm().item()
                    return _hook

                for name, module in model.named_modules():
                    if name in log_name_map:
                        def _fwd(mod, inp, out, ln=name):
                            outs = out if isinstance(out, (tuple, list)) else (out,)
                            for o in outs:
                                if torch.is_tensor(o) and getattr(o, 'requires_grad', False):
                                    o.register_hook(_make_init_hook(ln))
                            return None
                        handles.append(module.register_forward_hook(_fwd))

                optimizer.zero_grad()
                _loss0 = model(x_data, x_mask, y_data, y_mask)
                _loss0.backward()
                for h in handles: h.remove()
                optimizer.zero_grad()

                init_values = list(init_grad.values())
                if init_values:
                    log({"grad_mean/init": float(np.mean(init_values)), 
                         "grad_var/init": float(np.var(init_values))})
                
                for name, norm in init_grad.items():
                    last_known[name] = norm
                
                got_init = True
            
            if iloop % update_step == 0: update_flag = True; n_update += 1
            else: update_flag = False
            loss = train(model, optimizer, x_data, x_mask, y_data, y_mask, args.grad_clip, update_flag, update_step)
            loss_total += loss
            
            if iloop % valid_interval == 0 and iloop >= valid_start:
                save_list = {'n_update': n_update,'state_dict': model.state_dict()}
                if args.opt_scheduled:
                    save_list['scheduler'] = optimizer.state_dict()
                    save_list['optimizer'] = optim_adam.state_dict()
                else:
                    save_list['optimizer'] = optimizer.state_dict()
                torch.save(save_list, file_pre + '.pth')
                with open(file_pre + '.line', 'w') as lfp:
                    lfp.write('cur_line: ' + str(cur_line) + '\n')
                
                bleu_score = translate_file(model, args.valid_src_file + args.valid_post,
                                            args.src_dict, args.valid_trg_file, args.trg_dict, '', args, valid=True)
                prev_bleus = [0] 
                if os.path.exists(file_pre+'.bleu'):
                    with open(file_pre+'.bleu', 'r') as bfp:
                        lines = bfp.readlines()
                        prev_bleus = [float(bs.split()[1]) for bs in lines]
                if bleu_score >= np.max(prev_bleus): # save the best model
                    torch.save(save_list, file_pre+'.best.pth')
                    with open(file_pre+'.line', 'w') as lfp:
                        lfp.write('cur_line: ' + str(cur_line)+'\n')

                mode = 'w' if iloop == valid_start else 'a'
                with open(file_pre+'.bleu', mode) as bfp: # keep the bleu scores
                    bfp.write(str(n_update) + '\t' + str(bleu_score) + '\n')                 
                    
                if args.logging == 1: log({"0_bleu": bleu_score})


            if iloop % log_interval == 0:
                loss_avg = loss_total / log_interval
                loss_total = 0
                print('%s: %d iters - %.4f %s' % (args.model_file, n_update, loss_avg, timeSince(start)))
                if args.logging == 1:
                    grad_log = {}
                    for raw, pretty in log_name_map.items():
                        vals = grad_stats.get(raw, [])
                        avg = float(sum(vals) / len(vals)) if vals else 0.0
                        # running 로그는 초기값이 0으로 찍히도록 그대로 둠
                        grad_log[f"grad_norm_running/{pretty}"] = avg
                        if vals:
                            last_known[raw] = avg # 값이 있을 때만 last_known 업데이트
                            grad_stats[raw].clear()
                    log({"0_train_loss": loss_avg, **grad_log})

    except KeyboardInterrupt:
        print("\n사용자에 의해 학습이 중단되었습니다. 최종 지표를 기록합니다.")
        if args.logging == 1 and last_known:
            final_values = list(last_known.values())
            if final_values:
                log({"grad_mean/final": float(np.mean(final_values)), 
                     "grad_var/final": float(np.var(final_values))})
            print("최종 그래디언트 요약 지표를 기록했습니다.")

    finally:
        if args.logging == 1:
            time.sleep(5) 
            wandb.finish()

    print("학습 스크립트를 종료합니다.")