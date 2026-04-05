# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import math
import numpy as np
import random
import copy
import os
from io import open
import time
import re
from subprocess import Popen, PIPE

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from text_data import TextIterator, read_dict
from libs.utils import timeSince, ids2words, unbpe, unbpe_fix
from libs.layers import CudaVariable

import nmt_const as Const
from Beam import Beam
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장 가능하게 설정
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm

font_path = os.path.expanduser("~/.fonts/D2Coding/D2Coding-Ver1.3.2-20180524.ttf")
if os.path.exists(font_path):
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()
    print(f"D2Coding 폰트 적용 완료: {fontprop.get_name()}")
else:
    fontprop = None
    print("D2Coding 폰트를 찾을 수 없습니다.")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def save_attention_values_and_sums(attn, tokens_x, tokens_y, save_dir, filename_prefix):
    np.save(os.path.join(save_dir, f"{filename_prefix}_values.npy"), attn)
    np.savetxt(os.path.join(save_dir, f"{filename_prefix}_values.csv"), attn, delimiter=",")
    x_sums = attn.sum(axis=0)
    with open(os.path.join(save_dir, f"{filename_prefix}_x_token_sums.txt"), "w", encoding="utf-8") as f:
        for token, score in zip(tokens_x, x_sums):
            f.write(f"{token}\t{score:.6f}\n")
    y_sums = attn.sum(axis=1)
    with open(os.path.join(save_dir, f"{filename_prefix}_y_token_sums.txt"), "w", encoding="utf-8") as f:
        for token, score in zip(tokens_y, y_sums):
            f.write(f"{token}\t{score:.6f}\n")
def translate_tm(model, x_data, x_mask, max_length, remove_bos=False):
    x_data = CudaVariable(torch.LongTensor(x_data)).transpose(0, 1)
    x_mask = CudaVariable(torch.LongTensor(x_mask)).transpose(0, 1)

    if remove_bos:
        x_data = x_data[:, 1:]
        x_mask = x_mask[:, 1:]

    Bn, Tx = x_data.size()
    enc_out, enc_entropy_list, enc_attn_list = model.encoder(x_data, x_mask)

    y_hat0 = CudaVariable(torch.ones((Bn, 1)) * Const.BOS).type(torch.cuda.LongTensor)
    y_hat = y_hat0
    EOSs = torch.zeros((Bn, 1)).to(device)

    last_self_attn_list = None
    last_cross_attn_list = None

    for yi in range(max_length):
        dec_seq = y_hat.view(Bn, -1)
        y_mask = torch.ones((Bn, dec_seq.size(1)), device=x_mask.device)

        dec_out, self_entropy_list, cross_entropy_list, self_attn_list, cross_attn_list = model.decoder(
            dec_seq, y_mask, enc_out, x_mask
        )

        last_self_attn_list = self_attn_list
        last_cross_attn_list = cross_attn_list

        topv, yt = dec_out.topk(1, dim=2)
        yt = yt.view(Bn, yt.size(1))
        y_hat = torch.cat((y_hat0, yt), dim=1)
        EOS1 = torch.eq(yt[:, -1], Const.EOS).view(Bn, 1)
        EOSs = EOSs + EOS1.type(torch.cuda.FloatTensor)

        if yi > 0 and torch.sum(torch.gt(EOSs, 0)) >= Bn:
            break

    y_hat = y_hat.cpu().detach().numpy()[:, 1:]
    return y_hat, enc_attn_list, last_self_attn_list, last_cross_attn_list

def process_tokens_for_display(tokens):
    processed_tokens = []
    for i, token in enumerate(tokens):
        if "@@" in token:
            token = token.replace("@@", "⋄")
        if i > 0 and token == tokens[i - 1]:
            count = 1
            j = i - 1
            while j >= 0 and tokens[j] == token:
                count += 1
                j -= 1
            token = f"{token}({count})"
        processed_tokens.append(token)
    return processed_tokens

def merge_repeated_tokens(tokens, attention_matrix):
    merged_tokens = []
    merged_indices = []
    token_groups = {}

    for i, token in enumerate(tokens):
        if token not in token_groups:
            token_groups[token] = []
        token_groups[token].append(i)

    for token, indices in token_groups.items():
        merged_tokens.append(token)
        merged_indices.append(indices)

    rows, cols = len(merged_indices), len(merged_indices)
    merged_attention = np.zeros((rows, cols))

    for i, row_indices in enumerate(merged_indices):
        for j, col_indices in enumerate(merged_indices):
            total = 0
            count = 0
            for ri in row_indices:
                for ci in col_indices:
                    if ri < attention_matrix.shape[0] and ci < attention_matrix.shape[1]:
                        total += attention_matrix[ri, ci]
                        count += 1
            if count > 0:
                merged_attention[i, j] = total / count

    return merged_tokens, merged_attention

def plot_avg_attention_heatmap(layer_attn, tokens_src, tokens_tgt, layer_idx, save_dir, filename_prefix, is_self_attn=True):
    if isinstance(layer_attn, torch.Tensor):
        avg_attn = layer_attn[0].mean(dim=0).cpu().detach().numpy()
        xticklabels = tokens_src if is_self_attn else tokens_src
        yticklabels = tokens_src if is_self_attn else tokens_tgt

        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_attn, xticklabels=xticklabels, yticklabels=yticklabels,
                    cmap='Blues', square=True, cbar_kws={'shrink': .8})
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.title(f'{filename_prefix} Layer {layer_idx+1} - Average Attention')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{filename_prefix}_layer{layer_idx+1}_avg_heads.png")
        plt.savefig(save_path, dpi=300)
        save_attention_values_and_sums(avg_attn, xticklabels, yticklabels, save_dir, f"{filename_prefix}_layer{layer_idx+1}_avg_heads")
        plt.close()

def plot_all_layers_avg_attention(attn_list, tokens_src, tokens_tgt, save_dir, filename_prefix, is_self_attn=True):
    if all(isinstance(layer_attn, torch.Tensor) for layer_attn in attn_list):
        all_layers_attn = torch.stack([layer_attn[0].mean(dim=0) for layer_attn in attn_list])
        avg_attn = all_layers_attn.mean(dim=0).cpu().detach().numpy()

        xticklabels = tokens_src if is_self_attn else tokens_src
        yticklabels = tokens_src if is_self_attn else tokens_tgt

        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_attn, xticklabels=xticklabels, yticklabels=yticklabels,
                    cmap='Blues', square=True, cbar_kws={'shrink': .8})
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.title(f'{filename_prefix} - All Layers Average Attention')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{filename_prefix}_all_layers_avg.png")
        plt.savefig(save_path, dpi=300)
        save_attention_values_and_sums(avg_attn, xticklabels, yticklabels, save_dir, f"{filename_prefix}_all_layers_avg")
        plt.close()

def translate_file(model, src_file, s_dict_file, trg_file, t_dict_file, trans_file, args, valid=None, save_attention=False, remove_bos=False):
    model.eval()
    print('data loader')
    valid_iter = TextIterator(src_file, s_dict_file, batch_size=1, maxlen=1000, ahead=1, resume_num=0, just_one_epoch=1, const_id=Const)

    src_dict = read_dict(s_dict_file, const_id=Const)
    trg_dict = read_dict(t_dict_file, const_id=Const)
    src_inv_dict = {v: k for k, v in src_dict.items()}
    trg_inv_dict = {v: k for k, v in trg_dict.items()}

    if save_attention:
        attn_mode = "no_bos" if remove_bos else "with_bos"
        save_base_dir = os.path.join(os.path.dirname(trans_file), f"attention_maps_{attn_mode}")
        os.makedirs(save_base_dir, exist_ok=True)

    if valid:
        multibleu_cmd = ["perl", args.bleu_script, trg_file, "<"]
        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True, encoding='utf-8')
    else:
        fp = open(trans_file, 'w')

    start = time.time()
    with torch.no_grad():
        print('translate')
        for i, (x_data, x_mask, cur_line, iloop) in enumerate(valid_iter):
            if save_attention and i >= 5:
                break

            if args.model == 'tm':
                save_dir = os.path.join(save_base_dir, f"sentence_{i+1}")
                os.makedirs(save_dir, exist_ok=True)

                samples, enc_attn_list, self_attn_list, cross_attn_list = translate_tm(
                    model, x_data, x_mask, args.max_length, remove_bos=remove_bos
                )

                original_x_data = torch.LongTensor(x_data).transpose(0,1)
                src_tokens = [src_inv_dict.get(id.item(), "<unk>") for id in original_x_data[0] if id.item() != Const.PAD]
                tgt_tokens = []
                for k in range(samples.shape[0]):
                    tokens = [trg_inv_dict.get(id, "<unk>") for id in samples[k] if id != Const.PAD and id != Const.EOS]
                    tgt_tokens.append(tokens)

                display_src = src_tokens[1:] if remove_bos and len(src_tokens) > 1 else src_tokens
                processed_src = process_tokens_for_display(display_src)
                processed_tgt = process_tokens_for_display(["<s>"] + tgt_tokens[0])

                for layer_idx, layer_attn in enumerate(enc_attn_list):
                    for head_idx in range(layer_attn.size(1)):
                        attn = layer_attn[0, head_idx].cpu().detach().numpy()
                        filename = f"encoder_layer{layer_idx+1}_head{head_idx+1}.png"
                        if attn.shape[0] > 0 and attn.shape[1] > 0:
                            plt.figure(figsize=(12, 10))
                            sns.heatmap(attn, xticklabels=processed_src, yticklabels=processed_src, cmap='Blues', square=True, cbar_kws={'shrink': .8})
                            plt.xticks(rotation=45, ha='right', fontsize=8)
                            plt.yticks(rotation=0, fontsize=8)
                            plt.tight_layout()
                            plt.savefig(os.path.join(save_dir, filename), dpi=300)
                            save_attention_values_and_sums(attn, processed_src, processed_src, save_dir, filename[:-4])
                            plt.close()

                    plot_avg_attention_heatmap(layer_attn, processed_src, processed_src, layer_idx, save_dir, "encoder", is_self_attn=True)

                plot_all_layers_avg_attention(enc_attn_list, processed_src, processed_src, save_dir, "encoder", is_self_attn=True)

                # 디코더 셀프 어텐션
                if isinstance(self_attn_list, list):
                    for layer_idx, self_attn in enumerate(self_attn_list):
                        if isinstance(self_attn, torch.Tensor):
                            for head_idx in range(self_attn.size(1)):
                                attn = self_attn[0, head_idx].cpu().detach().numpy()
                                filename = f"decoder_self_layer{layer_idx+1}_head{head_idx+1}.png"
                                if attn.shape[0] > 0 and attn.shape[1] > 0:
                                    plt.figure(figsize=(12, 10))
                                    sns.heatmap(attn, xticklabels=processed_tgt, yticklabels=processed_tgt, cmap='Blues', square=True, cbar_kws={'shrink': .8})
                                    plt.xticks(rotation=45, ha='right', fontsize=8)
                                    plt.yticks(rotation=0, fontsize=8)
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(save_dir, filename), dpi=300)
                                    save_attention_values_and_sums(attn, processed_tgt, processed_tgt, save_dir, filename[:-4])
                                    plt.close()

                            plot_avg_attention_heatmap(self_attn, processed_tgt, processed_tgt, layer_idx, save_dir, "decoder_self", is_self_attn=True)

                    plot_all_layers_avg_attention(self_attn_list, processed_tgt, processed_tgt, save_dir, "decoder_self", is_self_attn=True)
                if isinstance(cross_attn_list, list):
                    for layer_idx, cross_attn in enumerate(cross_attn_list):
                        if isinstance(cross_attn, torch.Tensor):
                            for head_idx in range(cross_attn.size(1)):
                                attn = cross_attn[0, head_idx].cpu().detach().numpy()
                                filename = f"cross_attn_layer{layer_idx+1}_head{head_idx+1}.png"
                                if attn.shape[0] > 0 and attn.shape[1] > 0:
                                    plt.figure(figsize=(12, 10))
                                    sns.heatmap(attn, xticklabels=processed_src, yticklabels=processed_tgt, cmap='Blues', square=True, cbar_kws={'shrink': .8})
                                    plt.xticks(rotation=45, ha='right', fontsize=8)
                                    plt.yticks(rotation=0, fontsize=8)
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(save_dir, filename), dpi=300)
                                    save_attention_values_and_sums(attn, processed_src, processed_tgt, save_dir, filename[:-4])
                                    plt.close()

                            plot_avg_attention_heatmap(cross_attn, processed_src, processed_tgt, layer_idx, save_dir, "cross_attn", is_self_attn=False)

                    plot_all_layers_avg_attention(cross_attn_list, processed_src, processed_tgt, save_dir, "cross_attn", is_self_attn=False)

            else:
                samples, _, _, _ = translate_tm(model, x_data, x_mask, args.max_length)

            for k in range(samples.shape[0]):
                sentence = ids2words(trg_inv_dict, samples[k, :], eos_id=Const.EOS)
                sentence = unbpe(sentence)
                if valid:
                    mb_subprocess.stdin.write(sentence + '\n')
                    mb_subprocess.stdin.flush()
                else:
                    fp.write(sentence + '\n')

    ret = -1
    if valid:
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        mb_subprocess.terminate()
        if out_parse:
            ret = float(out_parse.group()[6:])
        print('BLEU: %f, time / sentences: %s / %d' % (ret, timeSince(start), iloop))
    else:
        fp.close()
        print('time / sentences: %s / %d' % (timeSince(start), iloop))

    if save_attention:
        print(f'어텐션 맵 저장 위치: {save_base_dir}')

    return ret
