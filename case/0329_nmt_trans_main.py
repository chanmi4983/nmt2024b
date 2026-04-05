# -*- coding: utf-8 -*-

# H. Choi, hchoi@handong.edu
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
import numpy as np
import matplotlib.font_manager as fm

font_path = os.path.expanduser("~/.fonts/D2Coding/D2Coding-Ver1.3.2-20180524.ttf")

# 폰트가 존재하는지 확인 후 적용
if os.path.exists(font_path):
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()
    print(f"D2Coding 폰트 적용 완료: {fontprop.get_name()}")
else:
    fontprop = None  # 기본 폰트 사용
    print("D2Coding 폰트를 찾을 수 없습니다.")
use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")
def translate_tm(model, x_data, x_mask, max_length, remove_bos=False):
    """
    번역 함수
    
    Parameters:
        model: 번역 모델
        x_data: 소스 데이터
        x_mask: 소스 마스크
        max_length: 최대 번역 길이
        remove_bos: 인코더에서 BOS 제거 여부
    """
    x_data = CudaVariable(torch.LongTensor(x_data)).transpose(0,1) # B T
    x_mask = CudaVariable(torch.LongTensor(x_mask)).transpose(0,1) # B T

    # BOS 제거 여부에 따라 처리
    if remove_bos:
        # print("인코더에서 BOS 제거 후 계산")
        x_data = x_data[:,1:] # remove BOS
        x_mask = x_mask[:,1:] # remove BOS
    # else:
        # print("BOS/EOS를 포함한 상태로 계산")

    Bn, Tx = x_data.size()
    enc_out, enc_entropy_list, enc_attn_list = model.encoder(x_data, x_mask)
    
    y_hat0 = CudaVariable(torch.ones((Bn,1))*Const.BOS).type(torch.cuda.LongTensor)
    y_hat = y_hat0
    EOSs = torch.zeros((Bn, 1)).to(device)
    
    # 마지막 단계의 어텐션 저장용
    last_self_attn_list = None
    last_cross_attn_list = None
    
    for yi in range(max_length):
        len_dec_seq = yi + 1
        dec_seq = y_hat.view(Bn, -1)
        y_mask = torch.ones((Bn,len_dec_seq), device=x_mask.device)

        dec_out, self_entropy_list, cross_entropy_list, self_attn_list, cross_attn_list = model.decoder(dec_seq, y_mask, enc_out, x_mask)

        # 현재 단계의 어텐션 저장
        last_self_attn_list = self_attn_list
        last_cross_attn_list = cross_attn_list
        
        topv, yt = dec_out.topk(1, dim=2)
        yt = yt.view(Bn, yt.size(1)) # B Ty
        y_hat = torch.cat((y_hat0, yt), dim=1) # B Ty+1
        EOS1 = torch.eq(yt[:,-1], Const.EOS).view(Bn, 1) # B 1 
        EOSs = EOSs + EOS1.type(torch.cuda.FloatTensor)
        if yi > 0 and torch.sum(torch.gt(EOSs,0)) >= Bn:
            break

    y_hat = y_hat.cpu().detach().numpy()[:,1:] # ignore BOS

    return y_hat, enc_attn_list, last_self_attn_list, last_cross_attn_list

def translate_file(model, src_file, s_dict_file, trg_file, t_dict_file, trans_file, args, valid=None, save_attention=False, remove_bos=False):
    model.eval()
    
    # 시각화에 필요한 라이브러리 임포트
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    print('data loader')
    valid_iter = TextIterator(src_file, s_dict_file, batch_size=1, 
                    maxlen=1000, ahead=1, resume_num=0, just_one_epoch=1, const_id=Const)

    src_dict = read_dict(s_dict_file, const_id=Const)
    trg_dict = read_dict(t_dict_file, const_id=Const)

    src_inv_dict = dict()
    for kk, vv in src_dict.items():
        src_inv_dict[vv] = kk
        
    trg_inv_dict = dict()
    for kk, vv in trg_dict.items():
        trg_inv_dict[vv] = kk

    # 어텐션 맵 저장을 위한 디렉토리 설정
    if save_attention:
        attn_mode = "no_bos" if remove_bos else "with_bos"
        save_base_dir = os.path.join(os.path.dirname(trans_file), f"attention_maps_{attn_mode}")
        os.makedirs(save_base_dir, exist_ok=True)

    # translate
    if valid:
        multibleu_cmd = ["perl", args.bleu_script, trg_file, "<"]
        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE, 
                                universal_newlines=True, encoding='utf-8')
    else:
        fp = open(trans_file, 'w')

    # 반복 토큰 처리 함수 정의
    def process_tokens_for_display(tokens):
        # 서브워드 토큰 마커 식별 및 처리
        processed_tokens = []
        for i, token in enumerate(tokens):
            # 서브워드 토큰인 경우 특별한 마커 추가
            if "@@" in token:
                token = token.replace("@@", "⋄")
            
            # 반복 토큰 처리 - 이전에 이미 토큰이 있는지 확인
            if i > 0 and token == tokens[i-1]:
                # 반복 토큰인 경우 인덱스 추가
                count = 1
                j = i - 1
                while j >= 0 and tokens[j] == token:
                    count += 1
                    j -= 1
                token = f"{token}({count})"
            
            processed_tokens.append(token)
        return processed_tokens
    
    # 반복 토큰 병합 및 어텐션 행렬 조정 함수
    def merge_repeated_tokens(tokens, attention_matrix):
        merged_tokens = []
        merged_indices = []
        token_groups = {}
        
        # 반복 토큰 그룹화
        for i, token in enumerate(tokens):
            if token not in token_groups:
                token_groups[token] = []
            token_groups[token].append(i)
        
        # 새로운 토큰 목록 및 인덱스 맵 생성
        for token, indices in token_groups.items():
            merged_tokens.append(token)
            merged_indices.append(indices)
        
        # 어텐션 행렬 조정
        rows, cols = len(merged_indices), len(merged_indices)
        merged_attention = np.zeros((rows, cols))
        
        for i, row_indices in enumerate(merged_indices):
            for j, col_indices in enumerate(merged_indices):
                # 해당 그룹 내 모든 어텐션 값의 평균 계산
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
    
    # 레이어별 헤드 평균 어텐션 맵 생성 함수
    def plot_avg_attention_heatmap(layer_attn, tokens_src, tokens_tgt, layer_idx, save_dir, filename_prefix, is_self_attn=True):
        """
        레이어의 모든 헤드에 대한 평균 어텐션 맵을 생성합니다.
        
        Parameters:
            layer_attn: 어텐션 텐서 (shape: [batch, heads, seq_len_tgt, seq_len_src])
            tokens_src: 소스 토큰 리스트
            tokens_tgt: 타겟 토큰 리스트 (self-attention인 경우 tokens_src와 동일)
            layer_idx: 레이어 인덱스
            save_dir: 저장할 디렉토리
            filename_prefix: 파일명 접두사 ('encoder', 'decoder_self', 'cross_attn')
            is_self_attn: self-attention 여부
        """
        if isinstance(layer_attn, torch.Tensor):
            # 모든 헤드의 어텐션 값 평균 계산
            avg_attn = layer_attn[0].mean(dim=0).cpu().detach().numpy()
            
            # 적절한 토큰 라벨 결정
            if is_self_attn:
                xticklabels = tokens_src
                yticklabels = tokens_src
            else:
                xticklabels = tokens_src
                yticklabels = tokens_tgt
                
            # 평균 어텐션 맵 시각화
            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(avg_attn, xticklabels=xticklabels, yticklabels=yticklabels,
                            cmap='Blues', square=True, cbar_kws={'shrink': .8})
            
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.title(f'{filename_prefix} Layer {layer_idx+1} - Average Attention')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{filename_prefix}_layer{layer_idx+1}_avg_heads.png"), dpi=300)
            plt.close()
    
    # 모든 레이어의 평균 어텐션 맵 생성 함수
    def plot_all_layers_avg_attention(attn_list, tokens_src, tokens_tgt, save_dir, filename_prefix, is_self_attn=True):
        """
        모든 레이어의 평균 어텐션 맵을 생성합니다.
        """
        # 모든 레이어의 어텐션 텐서를 스택
        if all(isinstance(layer_attn, torch.Tensor) for layer_attn in attn_list):
            all_layers_attn = torch.stack([layer_attn[0].mean(dim=0) for layer_attn in attn_list])
            # 레이어 간 평균 계산
            avg_attn = all_layers_attn.mean(dim=0).cpu().detach().numpy()
            
            # 적절한 토큰 라벨 결정
            if is_self_attn:
                xticklabels = tokens_src
                yticklabels = tokens_src
            else:
                xticklabels = tokens_src
                yticklabels = tokens_tgt
                
            # 평균 어텐션 맵 시각화
            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(avg_attn, xticklabels=xticklabels, yticklabels=yticklabels,
                            cmap='Blues', square=True, cbar_kws={'shrink': .8})
            
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.title(f'{filename_prefix} - All Layers Average Attention')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{filename_prefix}_all_layers_avg.png"), dpi=300)
            plt.close()

    start = time.time()
    with torch.no_grad():
        print('translate')
        for i, (x_data, x_mask, cur_line, iloop) in enumerate(valid_iter):
            # 시각화는 최대 5개 문장에 대해서만 수행
            if save_attention and i >= 5:
                break
                
            if args.model == 'tm':
                if save_attention:
                    # 어텐션 저장이 필요한 경우
                    save_dir = os.path.join(save_base_dir, f"sentence_{i+1}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 번역 및 어텐션 정보 얻기
                    samples, enc_attn_list, self_attn_list, cross_attn_list = translate_tm(
                        model, x_data, x_mask, args.max_length, remove_bos=remove_bos
                    )
                    
                    # 소스 토큰 추출
                    original_x_data = torch.LongTensor(x_data).transpose(0,1)
                    src_tokens = [src_inv_dict.get(id.item(), "<unk>") for id in original_x_data[0] if id.item() != Const.PAD]
                    
                    # 타겟 토큰 추출 (번역 결과)
                    tgt_tokens = []
                    for k in range(samples.shape[0]):
                        tokens = [trg_inv_dict.get(id, "<unk>") for id in samples[k] if id != Const.PAD and id != Const.EOS]
                        tgt_tokens.append(tokens)
                    
                    # BOS 제거 옵션에 따라 표시할 토큰 조정
                    display_src = src_tokens[1:] if remove_bos and len(src_tokens) > 1 else src_tokens
                    
                    # 토큰 처리
                    processed_src = process_tokens_for_display(display_src)
                    processed_tgt = process_tokens_for_display(["<s>"] + tgt_tokens[0])  # BOS + 생성된 토큰
                    
                    # 인코더 어텐션 맵 시각화 (모든 레이어)
                    for layer_idx, layer_attn in enumerate(enc_attn_list):
                        # 각 헤드별 어텐션 맵 
                        for head_idx in range(layer_attn.size(1)):
                            attn = layer_attn[0, head_idx].cpu().detach().numpy()
                            
                            # 반복 토큰 병합이 필요한 경우
                            if len(set(display_src)) < len(display_src):
                                merged_tokens, merged_attn = merge_repeated_tokens(display_src, attn)
                                
                                # 병합된 토큰으로 시각화
                                filename = f"encoder_layer{layer_idx+1}_head{head_idx+1}_merged.png"
                                
                                if merged_attn.shape[0] > 0 and merged_attn.shape[1] > 0:
                                    plt.figure(figsize=(12, 10))
                                    ax = sns.heatmap(merged_attn, xticklabels=merged_tokens, yticklabels=merged_tokens, 
                                                    cmap='Blues', square=True, cbar_kws={'shrink': .8})
                                    
                                    plt.xticks(rotation=45, ha='right', fontsize=8)
                                    plt.yticks(rotation=0, fontsize=8)
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(save_dir, filename), dpi=300)
                                    plt.close()
                            
                            # 원본 토큰으로도 시각화 (반복 토큰 표시 개선)
                            filename = f"encoder_layer{layer_idx+1}_head{head_idx+1}.png"
                            
                            if attn.shape[0] > 0 and attn.shape[1] > 0:
                                plt.figure(figsize=(12, 10))
                                ax = sns.heatmap(attn, xticklabels=processed_src, yticklabels=processed_src, 
                                                cmap='Blues', square=True, cbar_kws={'shrink': .8})
                                
                                plt.xticks(rotation=45, ha='right', fontsize=8)
                                plt.yticks(rotation=0, fontsize=8)
                                plt.tight_layout()
                                plt.savefig(os.path.join(save_dir, filename), dpi=300)
                                plt.close()
                        
                        # 레이어별 헤드 평균 어텐션 맵 추가
                        plot_avg_attention_heatmap(
                            layer_attn, 
                            processed_src, 
                            processed_src, 
                            layer_idx, 
                            save_dir,
                            "encoder",
                            is_self_attn=True
                        )
                    
                    # 모든 레이어의 평균 어텐션 맵 추가 (인코더)
                    plot_all_layers_avg_attention(
                        enc_attn_list,
                        processed_src,
                        processed_src,
                        save_dir,
                        "encoder",
                        is_self_attn=True
                    )
                    
                    # 디코더 셀프 어텐션 맵 (모든 레이어)
                    if isinstance(self_attn_list, list):
                        for layer_idx, self_attn in enumerate(self_attn_list):
                            if isinstance(self_attn, torch.Tensor):
                                # 각 헤드별 어텐션 맵
                                num_heads = self_attn.size(1)
                                for head_idx in range(num_heads):
                                    attn = self_attn[0, head_idx].cpu().detach().numpy()
                                    
                                    # 반복 토큰 병합이 필요한 경우
                                    if len(set(processed_tgt)) < len(processed_tgt):
                                        merged_tokens, merged_attn = merge_repeated_tokens(processed_tgt, attn)
                                        
                                        # 병합된 토큰으로 시각화
                                        filename = f"decoder_self_layer{layer_idx+1}_head{head_idx+1}_merged.png"
                                        
                                        if merged_attn.shape[0] > 0 and merged_attn.shape[1] > 0:
                                            plt.figure(figsize=(12, 10))
                                            ax = sns.heatmap(merged_attn, xticklabels=merged_tokens, yticklabels=merged_tokens, 
                                                            cmap='Blues', square=True, cbar_kws={'shrink': .8})
                                            
                                            plt.xticks(rotation=45, ha='right', fontsize=8)
                                            plt.yticks(rotation=0, fontsize=8)
                                            plt.tight_layout()
                                            plt.savefig(os.path.join(save_dir, filename), dpi=300)
                                            plt.close()
                                    
                                    # 원본 토큰으로도 시각화
                                    filename = f"decoder_self_layer{layer_idx+1}_head{head_idx+1}.png"
                                    
                                    if attn.shape[0] > 0 and attn.shape[1] > 0:
                                        plt.figure(figsize=(12, 10))
                                        ax = sns.heatmap(attn, xticklabels=processed_tgt, yticklabels=processed_tgt, 
                                                        cmap='Blues', square=True, cbar_kws={'shrink': .8})
                                        
                                        plt.xticks(rotation=45, ha='right', fontsize=8)
                                        plt.yticks(rotation=0, fontsize=8)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(save_dir, filename), dpi=300)
                                        plt.close()
                                
                                # 레이어별 헤드 평균 어텐션 맵 추가
                                plot_avg_attention_heatmap(
                                    self_attn, 
                                    processed_tgt, 
                                    processed_tgt, 
                                    layer_idx, 
                                    save_dir,
                                    "decoder_self",
                                    is_self_attn=True
                                )
                        
                        # 모든 레이어의 평균 어텐션 맵 추가 (디코더 셀프)
                        plot_all_layers_avg_attention(
                            self_attn_list,
                            processed_tgt,
                            processed_tgt,
                            save_dir,
                            "decoder_self",
                            is_self_attn=True
                        )
                    
                    # 크로스 어텐션 맵 (모든 레이어)
                    if isinstance(cross_attn_list, list):
                        for layer_idx, cross_attn in enumerate(cross_attn_list):
                            if isinstance(cross_attn, torch.Tensor):
                                # 각 헤드별 어텐션 맵
                                num_heads = cross_attn.size(1)
                                for head_idx in range(num_heads):
                                    attn = cross_attn[0, head_idx].cpu().detach().numpy()
                                    
                                    # 원본 토큰으로 시각화
                                    filename = f"cross_attn_layer{layer_idx+1}_head{head_idx+1}.png"
                                    
                                    if attn.shape[0] > 0 and attn.shape[1] > 0:
                                        plt.figure(figsize=(12, 10))
                                        ax = sns.heatmap(attn, xticklabels=processed_src, yticklabels=processed_tgt, 
                                                        cmap='Blues', square=True, cbar_kws={'shrink': .8})
                                        
                                        plt.xticks(rotation=45, ha='right', fontsize=8)
                                        plt.yticks(rotation=0, fontsize=8)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(save_dir, filename), dpi=300)
                                        plt.close()
                                
                                # 레이어별 헤드 평균 어텐션 맵 추가
                                plot_avg_attention_heatmap(
                                    cross_attn, 
                                    processed_src, 
                                    processed_tgt, 
                                    layer_idx, 
                                    save_dir,
                                    "cross_attn",
                                    is_self_attn=False
                                )
                        
                        # 모든 레이어의 평균 어텐션 맵 추가 (크로스 어텐션)
                        plot_all_layers_avg_attention(
                            cross_attn_list,
                            processed_src,
                            processed_tgt,
                            save_dir,
                            "cross_attn",
                            is_self_attn=False
                        )
                else:
                    # 일반 번역
                    samples, _, _, _ = translate_tm(model, x_data, x_mask, args.max_length)

            for k in range(samples.shape[0]): # over the batch
                sentence = ids2words(trg_inv_dict, samples[k,:], eos_id=Const.EOS)
                sentence = unbpe(sentence)
                if valid: 
                    mb_subprocess.stdin.write(sentence + '\n')
                    mb_subprocess.stdin.flush()
                else:
                    fp.write(sentence+'\n')

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