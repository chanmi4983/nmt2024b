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

from text_data import TextPairIterator, read_dict
from libs.utils import timeSince
from libs.opts import ScheduledOptim, InverseRootSquareScheduler

import nmt_const as Const
from nmt_model_tm import Transformer, MultiHeadAttention, FeedForward
from nmt_trans_main import translate_file
import wandb

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(model, optimizer, x_data, x_mask, y_data, y_mask, grad_clip, update_flag, update_step):
    model.train()
    # forward returns loss, since Transformer.forward computes and returns loss
    loss = model(x_data, x_mask, y_data, y_mask)
    loss = loss / update_step
    loss.backward()
    if update_flag:
        if grad_clip > 0.0:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def train_main(args):
    # Load data and model
    src_dict = read_dict(args.src_dict)
    trg_dict = read_dict(args.trg_dict)
    model = Transformer(len(src_dict), len(trg_dict), args=args).to(device)

    # W&B setup
    if args.logging == 1:
        wandb.init(project="nmt", config=args, settings=wandb.Settings(code_dir="."))
        wandb.watch(model, log_freq=args.log_interval)
        log = wandb.log
    else:
        log = lambda *a, **k: None

    # Identify modules for gradient monitoring
    layer_names = [name for name, module in model.named_modules()
                   if isinstance(module, (MultiHeadAttention, FeedForward))]
    grad_stats = {name: [] for name in layer_names}
    grad_norm_init = {}

    # Mid-training hooks: record grad norm on module outputs
    for name, module in model.named_modules():
        if name in layer_names:
            def make_mid_hook(layer_name):
                def fwd_hook(mod, inp, output):
                    def grad_hook(grad):
                        grad_stats[layer_name].append(grad.norm().item())
                    output.register_hook(grad_hook)
                    return None
                return fwd_hook
            module.register_forward_hook(make_mid_hook(name))

    # Optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt_scheduled:
        optim_adam = optim.Adam(params, betas=(0.9,0.98), eps=1e-9)
        optimizer = ScheduledOptim(optim_adam, 2.0, args.dim_wemb, args.opt_start)
    else:
        optimizer = optim.Adam(params, lr=args.lr)

    # Resume checkpoint
    file_pre = f"{args.save_dir}/{args.model_file}"
    if args.resume:
        ckpt = torch.load(os.path.join(args.save_dir, args.resume_file))
        model.load_state_dict(ckpt['state_dict'], strict=False)
        if args.opt_scheduled:
            optimizer.load_state_dict(ckpt['scheduler'])
            optim_adam.load_state_dict(ckpt['optimizer'])

    # Data iterator
    train_iter = TextPairIterator(
        args.train_src_file, args.train_trg_file,
        args.src_dict, args.trg_dict,
        batch_size=args.batch_size, maxlen=args.max_length,
        ahead=1000, resume_num=args.resume_line, const_id=Const)

    # Training loop variables
    start_time = time.time()
    loss_total = 0
    print('training iteration starts...')  # 초기 학습 시작 메시지 추가
    update_step = args.update_step
    log_interval = args.log_interval * update_step
    valid_interval = args.valid_interval * update_step
    valid_start = args.valid_start * update_step
    n_update = 0
    got_init = False

    for x_data, x_mask, y_data, y_mask, cur_line, iloop in train_iter:
        # Init gradient norm recording at first batch
        if not got_init:
            # Register hooks to capture init grad norms
            handles = []
            def make_init_hook(layer_name):
                def hook(grad):
                    grad_norm_init[layer_name] = grad.norm().item()
                return hook
            for name, module in model.named_modules():
                if name in layer_names:
                    def fwd_hook(mod, inp, output, layer=name):
                        output.register_hook(make_init_hook(layer))
                        return None
                    handles.append(module.register_forward_hook(fwd_hook))
            # Zero gradients
            optimizer.zero_grad()
            # Forward/backward to capture init
            # Use model.forward for loss then backward
            loss0 = model(x_data, x_mask, y_data, y_mask)
            loss0.backward()
            # Remove init hooks
            for h in handles: h.remove()
            optimizer.zero_grad()
            got_init = True

        # Standard training step
        is_update = (iloop % update_step == 0)
        if is_update: n_update += 1
        loss = train(model, optimizer, x_data, x_mask, y_data, y_mask,
                     args.grad_clip, is_update, update_step)
        loss_total += loss

        # Validation and checkpointing
        if iloop >= valid_start and iloop % valid_interval == 0:
            # Save checkpoint
            save_dict = {'n_update':n_update,
                         'state_dict':model.state_dict(),
                         'optimizer':optimizer.state_dict()}
            if args.opt_scheduled:
                save_dict['scheduler'] = optimizer.state_dict()
            torch.save(save_dict, file_pre+'.pth')
            with open(file_pre+'.line','w') as f:
                f.write(f'cur_line: {cur_line}\n')
            # Compute BLEU
            bleu_score = translate_file(model,
                                        args.valid_src_file+args.valid_post,
                                        args.src_dict, args.valid_trg_file,
                                        args.trg_dict, '', args, valid=True)
            log({'0_bleu': bleu_score})

        # Logging train loss and grad norms
        if iloop % log_interval == 0:
            avg_loss = loss_total / log_interval
            loss_total = 0
            elapsed = timeSince(start_time)
            print(f"{args.model_file}: {n_update} iters - {avg_loss:.4f} {elapsed}")
            # grad norms
            grad_log = {}
            for layer in layer_names:
                norms = grad_stats[layer]
                avg_norm = sum(norms)/len(norms) if norms else 0.0
                print(f"{layer:30s}: avg grad norm = {avg_norm:.4f}")
                grad_log[f"grad_norm/{layer}"] = avg_norm
                grad_stats[layer].clear()
            log({'0_train_loss': avg_loss, **grad_log})

    # After training: final metrics
    grad_norm_final = {l:(grad_stats[l][-1] if grad_stats[l] else 0.0) for l in layer_names}
    mean_init = np.mean(list(grad_norm_init.values()))
    mean_final = np.mean(list(grad_norm_final.values()))
    var_init = np.var(list(grad_norm_init.values()))
    var_final = np.var(list(grad_norm_final.values()))
    log({'grad_norm/init': mean_init,
         'grad_norm/final':mean_final,
         'grad_var/init':var_init,
         'grad_var/final':var_final})

# End of File
