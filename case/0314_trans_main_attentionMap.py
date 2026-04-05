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

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def translate_tm(model, x_data, x_mask, max_length):
    x_data = CudaVariable(torch.LongTensor(x_data)).transpose(0,1) # B T
    x_mask = CudaVariable(torch.LongTensor(x_mask)).transpose(0,1) # B T

    Bn, Tx = x_data.size() #배치와 시퀀스 길이 가져오기
    enc_out, enc_entropy_list, enc_attn_list = model.encoder(x_data, x_mask) #0203 수정 후
    
    y_hat0 = CudaVariable(torch.ones((Bn,1))*Const.BOS).type(torch.cuda.LongTensor) #시작 토큰(BOS) 설정
    y_hat = y_hat0 #모델을 reset하기 위해 y_hat0을 y_hat에 저장
    EOSs = torch.zeros((Bn, 1)).to(device)
    
    # 각 디코딩 단계별 어텐션 저장
    all_self_attentions = []
    all_cross_attentions = []
    for yi in range(max_length):
        len_dec_seq = yi + 1
        dec_seq = y_hat.view(Bn, -1)
        y_mask = torch.ones((Bn,len_dec_seq), device=x_mask.device)

        dec_out, self_entropy_list, cross_entropy_list, self_attn_list, cross_attn_list= model.decoder(dec_seq, y_mask, enc_out, x_mask) # 수정 후
        #dec_out, _, _= model.decoder(dec_seq, y_mask, enc_out, x_mask)
        
        # 현재 단계의 어텐션 저장
        all_self_attentions.append(self_attn_list)
        all_cross_attentions.append(cross_attn_list)
        
        topv, yt = dec_out.topk(1, dim=2)
        yt = yt.view(Bn, yt.size(1)) # B Ty
        y_hat = torch.cat((y_hat0, yt), dim=1) # B Ty+1
        EOS1 = torch.eq(yt[:,-1], Const.EOS).view(Bn, 1) # B 1 
        EOSs = EOSs + EOS1.type(torch.cuda.FloatTensor)
        if yi > 0 and torch.sum(torch.gt(EOSs,0)) >= Bn:
            #_, self_entropy_list, cross_entropy_list= model.decoder(dec_seq, y_mask, enc_out, x_mask)
            break

    y_hat = y_hat.cpu().detach().numpy()[:,1:] # ignore BOS
    
    # 최종 어텐션 결과 반환
    attention_results = {
        'encoder_attention': enc_attn_list,
        'decoder_self_attention': all_self_attentions,
        'decoder_cross_attention': all_cross_attentions
    }
    
    return y_hat,attention_results

def translate_file(model, src_file, s_dict_file, trg_file, t_dict_file, trans_file, args, valid=None):
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 이미지 저장 가능하게 설정
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    model.eval()
    
    print('data loader')
    # 배치 크기를 1로 설정
    valid_iter = TextIterator(src_file, s_dict_file, batch_size=1, 
                    maxlen=1000, ahead=1, resume_num=0, just_one_epoch=1, const_id=Const)
    
    # 사전 로드
    trg_dict = read_dict(t_dict_file, const_id=Const)
    trg_inv_dict = dict()
    for kk, vv in trg_dict.items():
        trg_inv_dict[vv] = kk
    
    # 소스 사전도 로드 (원본 토큰 표시용)
    src_dict = read_dict(s_dict_file, const_id=Const)
    src_inv_dict = dict()
    for kk, vv in src_dict.items():
        src_inv_dict[vv] = kk
    
    # 어텐션 분석 결과를 저장할 폴더 생성
    attention_dir = f"{trans_file}_attention_analysis"
    os.makedirs(attention_dir, exist_ok=True)
    
    # 히트맵을 저장할 폴더 생성
    heatmap_dir = f"{attention_dir}/heatmaps"
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # 진행 상황 추적용
    processed_sentences = set()
    
    # translate
    if valid:
        multibleu_cmd = ["perl", args.bleu_script, trg_file, "<"]
        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE, 
                                universal_newlines=True, encoding='utf-8')
    else:
        fp = open(trans_file, 'w')

    start = time.time()
    with torch.no_grad():
        print('translate')
        for x_data, x_mask, cur_line, iloop in valid_iter:
            # 현재 라인이 이미 처리되었으면 건너뛰기
            line_id = cur_line if isinstance(cur_line, int) else iloop
            if line_id in processed_sentences:
                print(f"Already processed sentence: {line_id}")
                continue
            
            # 처리된 문장 추적
            processed_sentences.add(line_id)
            
            if args.model == 'tm':
                # 번역 및 어텐션 분석
                samples, attention_results = translate_tm(model, x_data, x_mask, args.max_length)
                
                # 번역된 문장
                translation = ids2words(trg_inv_dict, samples[0,:], eos_id=Const.EOS)
                translation = unbpe(translation)
                
                # 원본 문장 복원 (토큰 ID -> 텍스트)
                source_tokens_ids = [idx for idx in x_data[0] if idx != Const.PAD]
                source_tokens = [src_inv_dict.get(idx, '<unk>') for idx in source_tokens_ids]
                source = ' '.join(source_tokens)
                
                # 번역 과정에서 생성된 토큰들 (EOS 제외)
                generated_tokens = []
                for i in range(samples.shape[1]):
                    if samples[0, i] == Const.EOS:
                        break
                    token = trg_inv_dict.get(samples[0, i], '<unk>')
                    generated_tokens.append(token)
                
                # 각 문장별 어텐션 시각화 파일 생성
                sent_file = open(f"{attention_dir}/sentence_{line_id}.txt", 'w')
                sent_file.write(f"Source: {source}\n")
                sent_file.write(f"Translation: {translation}\n\n")
                
                # 디코딩 과정의 어텐션 변화 분석
                sent_file.write("=== Decoder Attention Evolution Analysis ===\n")
                
                # 각 디코딩 단계별 어텐션 분석
                for step, (self_att, cross_att) in enumerate(zip(
                    attention_results['decoder_self_attention'], 
                    attention_results['decoder_cross_attention'])):
                    
                    # 현재 단계의 생성 토큰 (인덱스가 범위를 벗어나지 않도록 확인)
                    current_token = generated_tokens[step] if step < len(generated_tokens) else "<eos>"
                    sent_file.write(f"\n[Step {step+1}] Generated token: {current_token}\n")
                    
                    # 각 레이어의 어텐션 패턴 요약
                    for layer in range(len(self_att)):
                        # 셀프 어텐션 정보
                        if isinstance(self_att[layer], torch.Tensor):
                            # 어텐션 텐서 차원 출력 (디버깅용)
                            sent_file.write(f"  Layer {layer} self-attention dimensions: {tuple(self_att[layer].shape)}\n")
                            
                            # 평균, 최대값, 표준편차 계산
                            self_att_tensor = self_att[layer].cpu().detach()
                            self_att_mean = self_att_tensor.mean().item()
                            self_att_max = self_att_tensor.max().item()
                            self_att_std = self_att_tensor.std().item()
                            sent_file.write(f"  Layer {layer} self-attention: mean={self_att_mean:.4f}, max={self_att_max:.4f}, std={self_att_std:.4f}\n")
                            
                            # 마지막 레이어의 셀프 어텐션 히트맵 생성
                            if layer == len(self_att) - 1:
                                try:
                                    # 차원 변환
                                    self_att_matrix = process_attention_tensor(self_att[layer])
                                    
                                    # 현재까지 생성된 토큰에 대한 셀프 어텐션만 표시
                                    # 히트맵 크기에 맞게 조정 (step+1 x step+1)
                                    att_size = min(step+1, self_att_matrix.shape[0], self_att_matrix.shape[1])
                                    if att_size > 0:
                                        plt.figure(figsize=(8, 6))
                                        # 커스텀 컬러맵: 흰색에서 파란색으로
                                        cmap = LinearSegmentedColormap.from_list('blue_cmap', ['white', 'navy'], N=100)
                                        
                                        # 표시할 토큰 목록 (현재까지 생성된 토큰들)
                                        tokens_to_show = generated_tokens[:att_size]
                                        
                                        # 히트맵 그리기
                                        sns.heatmap(self_att_matrix[:att_size, :att_size], 
                                                    cmap=cmap,
                                                    vmin=0.0, 
                                                    vmax=np.max(self_att_matrix[:att_size, :att_size]),
                                                    xticklabels=tokens_to_show,
                                                    yticklabels=tokens_to_show,
                                                    square=True)
                                        
                                        # 영어 라벨 사용
                                        plt.title(f"Decoder Self-Attention - Step {step+1}", fontsize=14)
                                        plt.ylabel('Query Tokens', fontsize=12)
                                        plt.xlabel('Key Tokens', fontsize=12)
                                        plt.xticks(rotation=45, ha='right', fontsize=10)
                                        plt.yticks(rotation=0, fontsize=10)
                                        plt.tight_layout()
                                        
                                        # 파일로 저장
                                        plt.savefig(f"{heatmap_dir}/self_attention_step_{step+1}.png", dpi=200, bbox_inches='tight')
                                        plt.close()
                                except Exception as e:
                                    sent_file.write(f"  Self-attention heatmap error: {str(e)}\n")
                        
                        # 크로스 어텐션 정보
                        if isinstance(cross_att[layer], torch.Tensor):
                            # 어텐션 텐서 차원 출력 (디버깅용)
                            sent_file.write(f"  Layer {layer} cross-attention dimensions: {tuple(cross_att[layer].shape)}\n")
                            
                            # 평균, 최대값, 표준편차 계산
                            cross_att_tensor = cross_att[layer].cpu().detach()
                            cross_att_mean = cross_att_tensor.mean().item()
                            cross_att_max = cross_att_tensor.max().item()
                            cross_att_std = cross_att_tensor.std().item()
                            sent_file.write(f"  Layer {layer} cross-attention: mean={cross_att_mean:.4f}, max={cross_att_max:.4f}, std={cross_att_std:.4f}\n")
                            
                            # 마지막 레이어의 크로스 어텐션 히트맵 생성
                            if layer == len(cross_att) - 1:
                                try:
                                    # 차원 변환
                                    cross_att_matrix = process_attention_tensor(cross_att[layer])
                                    
                                    # 히트맵의 행 수 확인
                                    num_rows = min(1, cross_att_matrix.shape[0])
                                    
                                    # 현재 생성된 토큰과 소스 문장 사이의 크로스 어텐션
                                    plt.figure(figsize=(10, 6))
                                    cmap = LinearSegmentedColormap.from_list('red_cmap', ['white', 'darkred'], N=100)
                                    
                                    # 히트맵 그리기 (현재 토큰 x 소스 토큰)
                                    sns.heatmap(cross_att_matrix[:num_rows, :], 
                                                cmap=cmap,
                                                vmin=0.0, 
                                                vmax=np.max(cross_att_matrix[:num_rows, :]),
                                                xticklabels=source_tokens,
                                                yticklabels=[current_token],
                                                square=False)
                                    
                                    # 영어 라벨 사용
                                    plt.title(f"Decoder Cross-Attention - Step {step+1} ({current_token})", fontsize=14)
                                    plt.ylabel('Query Token', fontsize=12)
                                    plt.xlabel('Source Tokens', fontsize=12)
                                    plt.xticks(rotation=45, ha='right', fontsize=10)
                                    plt.yticks(rotation=0, fontsize=10)
                                    plt.tight_layout()
                                    
                                    # 파일로 저장
                                    plt.savefig(f"{heatmap_dir}/cross_attention_step_{step+1}.png", dpi=200, bbox_inches='tight')
                                    plt.close()
                                except Exception as e:
                                    sent_file.write(f"  Cross-attention heatmap error: {str(e)}\n")
                
                # 인코더 어텐션 분석
                sent_file.write("\n=== Encoder Attention Analysis ===\n")
                for i, att in enumerate(attention_results['encoder_attention']):
                    if isinstance(att, torch.Tensor):
                        # 어텐션 텐서 차원 출력 (디버깅용)
                        sent_file.write(f"Layer {i} encoder attention dimensions: {tuple(att.shape)}\n")
                        
                        # 평균, 최대값, 표준편차 계산
                        att_tensor = att.cpu().detach()
                        att_mean = att_tensor.mean().item()
                        att_max = att_tensor.max().item()
                        att_std = att_tensor.std().item()
                        sent_file.write(f"Layer {i}: mean={att_mean:.4f}, max={att_max:.4f}, std={att_std:.4f}\n")
                        
                        # 마지막 레이어의 인코더 어텐션 히트맵 생성
                        if i == len(attention_results['encoder_attention']) - 1:
                            try:
                                # 차원 변환
                                enc_att_matrix = process_attention_tensor(att)
                                
                                plt.figure(figsize=(10, 8))
                                cmap = LinearSegmentedColormap.from_list('green_cmap', ['white', 'darkgreen'], N=100)
                                
                                # 히트맵 그리기
                                sns.heatmap(enc_att_matrix, 
                                            cmap=cmap,
                                            vmin=0.0, 
                                            vmax=np.max(enc_att_matrix),
                                            xticklabels=source_tokens,
                                            yticklabels=source_tokens,
                                            square=True)
                                
                                # 영어 라벨 사용
                                plt.title("Encoder Self-Attention", fontsize=14)
                                plt.ylabel('Query Tokens', fontsize=12)
                                plt.xlabel('Key Tokens', fontsize=12)
                                plt.xticks(rotation=45, ha='right', fontsize=10)
                                plt.yticks(rotation=0, fontsize=10)
                                plt.tight_layout()
                                
                                # 파일로 저장
                                plt.savefig(f"{heatmap_dir}/encoder_attention.png", dpi=200, bbox_inches='tight')
                                plt.close()
                            except Exception as e:
                                sent_file.write(f"Encoder attention heatmap error: {str(e)}\n")
                
                # 모든 어텐션 값 분석 및 요약 추가
                sent_file.write("\n=== Total Attention Analysis Summary ===\n")
                sent_file.write("| Layer | Enc Self Att | Dec Self Att | Dec Cross Att |\n")
                sent_file.write("|-------|-------------|-------------|-------------|\n")

                # 각 레이어별 최종 어텐션 값 요약
                for layer in range(6):  # 6개 레이어에 대해 반복
                    # 인코더 셀프 어텐션 (마지막 레이어의 값 사용)
                    if layer < len(attention_results['encoder_attention']):
                        enc_att = attention_results['encoder_attention'][layer]
                        if isinstance(enc_att, torch.Tensor):
                            enc_att_tensor = enc_att.cpu().detach()
                            enc_att_mean = enc_att_tensor.mean().item()
                            enc_att_std = enc_att_tensor.std().item()
                            enc_att_info = f"{enc_att_mean:.4f}±{enc_att_std:.4f}"
                        else:
                            enc_att_info = "N/A"
                    else:
                        enc_att_info = "N/A"
                    
                    # 디코더 셀프 어텐션 (마지막 생성 단계의 값 사용)
                    last_step = len(attention_results['decoder_self_attention']) - 1
                    if last_step >= 0 and layer < len(attention_results['decoder_self_attention'][last_step]):
                        dec_self_att = attention_results['decoder_self_attention'][last_step][layer]
                        if isinstance(dec_self_att, torch.Tensor):
                            dec_self_tensor = dec_self_att.cpu().detach()
                            dec_self_mean = dec_self_tensor.mean().item()
                            dec_self_std = dec_self_tensor.std().item()
                            dec_self_info = f"{dec_self_mean:.4f}±{dec_self_std:.4f}"
                        else:
                            dec_self_info = "N/A"
                    else:
                        dec_self_info = "N/A"
                    
                    # 디코더 크로스 어텐션 (마지막 생성 단계의 값 사용)
                    if last_step >= 0 and layer < len(attention_results['decoder_cross_attention'][last_step]):
                        dec_cross_att = attention_results['decoder_cross_attention'][last_step][layer]
                        if isinstance(dec_cross_att, torch.Tensor):
                            dec_cross_tensor = dec_cross_att.cpu().detach()
                            dec_cross_mean = dec_cross_tensor.mean().item()
                            dec_cross_std = dec_cross_tensor.std().item()
                            dec_cross_info = f"{dec_cross_mean:.4f}±{dec_cross_std:.4f}"
                        else:
                            dec_cross_info = "N/A"
                    else:
                        dec_cross_info = "N/A"
                    
                    # 테이블 형식으로 출력
                    sent_file.write(f"| {layer}     | {enc_att_info} | {dec_self_info} | {dec_cross_info} |\n")

                # 전체 어텐션 타입별 평균
                sent_file.write("\n=== Overall Attention Statistics ===\n")

                # 인코더 어텐션 평균
                enc_values = []
                for att in attention_results['encoder_attention']:
                    if isinstance(att, torch.Tensor):
                        enc_values.append(att.cpu().detach().mean().item())
                if enc_values:
                    enc_avg = sum(enc_values) / len(enc_values)
                    sent_file.write(f"Encoder Self-Attention Average: {enc_avg:.6f}\n")

                # 디코더 셀프 어텐션 평균 (마지막 단계)
                if last_step >= 0:
                    dec_self_values = []
                    for att in attention_results['decoder_self_attention'][last_step]:
                        if isinstance(att, torch.Tensor):
                            dec_self_values.append(att.cpu().detach().mean().item())
                    if dec_self_values:
                        dec_self_avg = sum(dec_self_values) / len(dec_self_values)
                        sent_file.write(f"Decoder Self-Attention Average: {dec_self_avg:.6f}\n")

                # 디코더 크로스 어텐션 평균 (마지막 단계)
                if last_step >= 0:
                    dec_cross_values = []
                    for att in attention_results['decoder_cross_attention'][last_step]:
                        if isinstance(att, torch.Tensor):
                            dec_cross_values.append(att.cpu().detach().mean().item())
                    if dec_cross_values:
                        dec_cross_avg = sum(dec_cross_values) / len(dec_cross_values)
                        sent_file.write(f"Decoder Cross-Attention Average: {dec_cross_avg:.6f}\n")

                # 어텐션 히트맵 추가 생성 (전체 18개 히트맵)
                sent_file.write("\n=== All 18 Attention Heatmaps ===\n")

                # 인코더 6개 레이어의 셀프 어텐션 히트맵
                for i, enc_att in enumerate(attention_results['encoder_attention']):
                    if isinstance(enc_att, torch.Tensor):
                        try:
                            # 차원 변환
                            enc_att_matrix = process_attention_tensor(enc_att)
                            
                            plt.figure(figsize=(8, 6))
                            cmap = LinearSegmentedColormap.from_list('green_cmap', ['white', 'darkgreen'], N=100)
                            
                            # 히트맵 그리기
                            sns.heatmap(enc_att_matrix, 
                                        cmap=cmap,
                                        vmin=0.0, 
                                        vmax=np.max(enc_att_matrix),
                                        xticklabels=source_tokens,
                                        yticklabels=source_tokens,
                                        square=True)
                            
                            plt.title(f"Encoder Self-Attention Layer {i}", fontsize=14)
                            plt.ylabel('Query Tokens', fontsize=12)
                            plt.xlabel('Key Tokens', fontsize=12)
                            plt.xticks(rotation=45, ha='right', fontsize=10)
                            plt.yticks(rotation=0, fontsize=10)
                            plt.tight_layout()
                            
                            # 파일로 저장
                            plt.savefig(f"{heatmap_dir}/encoder_attention_layer_{i}.png", dpi=200, bbox_inches='tight')
                            plt.close()
                            
                            sent_file.write(f"{i+1}. Encoder Self-Attention Layer {i}: {heatmap_dir}/encoder_attention_layer_{i}.png\n")
                        except Exception as e:
                            sent_file.write(f"{i+1}. Encoder Self-Attention Layer {i}: Error - {str(e)}\n")

                # 디코더 6개 레이어의 셀프 어텐션 히트맵 (마지막 생성 단계 기준)
                if last_step >= 0:
                    for i, dec_self_att in enumerate(attention_results['decoder_self_attention'][last_step]):
                        if isinstance(dec_self_att, torch.Tensor):
                            try:
                                # 차원 변환
                                dec_self_matrix = process_attention_tensor(dec_self_att)
                                
                                plt.figure(figsize=(8, 6))
                                cmap = LinearSegmentedColormap.from_list('blue_cmap', ['white', 'navy'], N=100)
                                
                                # 히트맵 그리기
                                sns.heatmap(dec_self_matrix, 
                                            cmap=cmap,
                                            vmin=0.0, 
                                            vmax=np.max(dec_self_matrix),
                                            xticklabels=generated_tokens,
                                            yticklabels=generated_tokens,
                                            square=True)
                                
                                plt.title(f"Decoder Self-Attention Layer {i} (Final Step)", fontsize=14)
                                plt.ylabel('Query Tokens', fontsize=12)
                                plt.xlabel('Key Tokens', fontsize=12)
                                plt.xticks(rotation=45, ha='right', fontsize=10)
                                plt.yticks(rotation=0, fontsize=10)
                                plt.tight_layout()
                                
                                # 파일로 저장
                                plt.savefig(f"{heatmap_dir}/decoder_self_attention_layer_{i}.png", dpi=200, bbox_inches='tight')
                                plt.close()
                                
                                sent_file.write(f"{i+7}. Decoder Self-Attention Layer {i}: {heatmap_dir}/decoder_self_attention_layer_{i}.png\n")
                            except Exception as e:
                                sent_file.write(f"{i+7}. Decoder Self-Attention Layer {i}: Error - {str(e)}\n")

                # 디코더 6개 레이어의 크로스 어텐션 히트맵 (마지막 생성 단계 기준)
                if last_step >= 0:
                    for i, dec_cross_att in enumerate(attention_results['decoder_cross_attention'][last_step]):
                        if isinstance(dec_cross_att, torch.Tensor):
                            try:
                                # 차원 변환
                                dec_cross_matrix = process_attention_tensor(dec_cross_att)
                                
                                plt.figure(figsize=(10, 6))
                                cmap = LinearSegmentedColormap.from_list('red_cmap', ['white', 'darkred'], N=100)
                                
                                # 히트맵 그리기
                                sns.heatmap(dec_cross_matrix, 
                                            cmap=cmap,
                                            vmin=0.0, 
                                            vmax=np.max(dec_cross_matrix),
                                            xticklabels=source_tokens,
                                            yticklabels=generated_tokens,
                                            square=False)
                                
                                plt.title(f"Decoder Cross-Attention Layer {i} (Final Step)", fontsize=14)
                                plt.ylabel('Query Tokens (Generated)', fontsize=12)
                                plt.xlabel('Key Tokens (Source)', fontsize=12)
                                plt.xticks(rotation=45, ha='right', fontsize=10)
                                plt.yticks(rotation=0, fontsize=10)
                                plt.tight_layout()
                                
                                # 파일로 저장
                                plt.savefig(f"{heatmap_dir}/decoder_cross_attention_layer_{i}.png", dpi=200, bbox_inches='tight')
                                plt.close()
                                
                                sent_file.write(f"{i+13}. Decoder Cross-Attention Layer {i}: {heatmap_dir}/decoder_cross_attention_layer_{i}.png\n")
                            except Exception as e:
                                sent_file.write(f"{i+13}. Decoder Cross-Attention Layer {i}: Error - {str(e)}\n")
                
                # 히트맵 목록을 텍스트 파일에 추가
                sent_file.write("\n=== Generated Heatmaps ===\n")
                sent_file.write(f"1. Encoder Self-Attention: {heatmap_dir}/encoder_attention.png\n")
                for step in range(len(attention_results['decoder_self_attention'])):
                    sent_file.write(f"{step+2}. Decoder Self-Attention (Step {step+1}): {heatmap_dir}/self_attention_step_{step+1}.png\n")
                    sent_file.write(f"{step+2+len(attention_results['decoder_self_attention'])}. Decoder Cross-Attention (Step {step+1}): {heatmap_dir}/cross_attention_step_{step+1}.png\n")
                
                sent_file.close()
                
                # 번역 결과 저장
                if valid: 
                    mb_subprocess.stdin.write(translation + '\n')
                    mb_subprocess.stdin.flush()
                else:
                    fp.write(translation+'\n')
                
                print(f"Sentence {line_id} processed - Heatmaps saved to {heatmap_dir}")

    
    # 분석 요약 파일 생성
    summary_file = open(f"{attention_dir}/analysis_summary.txt", 'w')
    summary_file.write(f"Number of processed sentences: {len(processed_sentences)}\n")
    summary_file.write("="*50 + "\n\n")
    summary_file.write(f"Attention analysis results are saved in: {attention_dir}\n")
    summary_file.write(f"Heatmap images are saved in: {heatmap_dir}\n\n")
    summary_file.write("Each sentence file contains the following information:\n")
    summary_file.write("1. Source sentence and translation\n")
    summary_file.write("2. Decoder attention evolution analysis (for each generation step)\n")
    summary_file.write("3. Encoder attention analysis\n")
    summary_file.write("4. Total attention analysis summary (18 attention heads)\n")
    summary_file.write("5. List of all 18 attention heatmaps\n")
    summary_file.write("6. List of generated step-by-step heatmaps\n")
    summary_file.close()
    
    ret = -1
    if valid: 
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        mb_subprocess.terminate()
        if out_parse:
            ret = float(out_parse.group()[6:])
        print(f'BLEU: {ret:.4f}, Time / Number of sentences: {timeSince(start)} / {len(processed_sentences)}')
    else:
        fp.close()
        print(f'Time / Number of sentences: {timeSince(start)} / {len(processed_sentences)}')
        print(f"Attention analysis results are saved in: {attention_dir}")
    
    return ret


def process_attention_tensor(attention_tensor):
    """다양한 차원의 어텐션 텐서를 2D 행렬로 변환"""
    # CPU로 이동하고 detach
    att = attention_tensor.cpu().detach()
    
    # 차원에 따라 적절히 처리
    if att.dim() == 4:  # [batch_size, num_heads, seq_len_q, seq_len_k]
        # 배치와 헤드에 대해 평균 계산
        att = att.mean(dim=(0, 1)).numpy()
    elif att.dim() == 3:  # [num_heads, seq_len_q, seq_len_k] 또는 [batch_size, seq_len_q, seq_len_k]
        # 첫 번째 차원에 대해 평균 계산
        att = att.mean(dim=0).numpy()
    else:  # 이미 2D인 경우
        att = att.numpy()
    
    return att