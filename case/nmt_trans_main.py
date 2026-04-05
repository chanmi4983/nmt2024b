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

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def translate_tm(model, x_data, x_mask, max_length):
    x_data = CudaVariable(torch.LongTensor(x_data)).transpose(0,1) # B T
    x_mask = CudaVariable(torch.LongTensor(x_mask)).transpose(0,1) # B T

    Bn, Tx = x_data.size() #배치와 시퀀스 길이 가져오기
    enc_out, enc_entropy_list = model.encoder(x_data, x_mask) #0203 수정 후
    
    y_hat0 = CudaVariable(torch.ones((Bn,1))*Const.BOS).type(torch.cuda.LongTensor) #시작 토큰(BOS) 설정
    y_hat = y_hat0 #모델을 reset하기 위해 y_hat0을 y_hat에 저장
    EOSs = torch.zeros((Bn, 1)).to(device)
    
    for yi in range(max_length):
        len_dec_seq = yi + 1
        dec_seq = y_hat.view(Bn, -1)
        y_mask = torch.ones((Bn,len_dec_seq), device=x_mask.device)

        dec_out, _, _= model.decoder(dec_seq, y_mask, enc_out, x_mask) # 수정 후
        
        topv, yt = dec_out.topk(1, dim=2)
        yt = yt.view(Bn, yt.size(1)) # B Ty
        y_hat = torch.cat((y_hat0, yt), dim=1) # B Ty+1
        EOS1 = torch.eq(yt[:,-1], Const.EOS).view(Bn, 1) # B 1 
        EOSs = EOSs + EOS1.type(torch.cuda.FloatTensor)
        if yi > 0 and torch.sum(torch.gt(EOSs,0)) >= Bn:
            _, self_entropy_list, cross_entropy_list= model.decoder(dec_seq, y_mask, enc_out, x_mask)
            break

    y_hat = y_hat.cpu().detach().numpy()[:,1:] # ignore BOS
    
    return y_hat, enc_entropy_list, self_entropy_list, cross_entropy_list

def translate_file(model, src_file, s_dict_file, trg_file, t_dict_file, trans_file, args, valid=None):
    model.eval()
    
    print('data loader')
    valid_iter = TextIterator(src_file, s_dict_file, batch_size=1, 
                    maxlen=1000, ahead=1, resume_num=0, just_one_epoch=1, const_id=Const)
    trg_dict = read_dict(t_dict_file, const_id=Const)
    trg_inv_dict = dict()
    for kk, vv in trg_dict.items():
        trg_inv_dict[vv] = kk
    
    # 각 레이어별 엔트로피 누적값을 저장할 딕셔너리 초기화
    enc_entropy_sums = {}
    self_entropy_sums = {}
    cross_entropy_sums = {}
    batch_count = 0
    
    # translate
    if valid:
        multibleu_cmd = ["perl", args.bleu_script, trg_file, "<"]
        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE, 
                                universal_newlines=True, encoding='utf-8')
    else:
        fp = open(trans_file, 'w')
    
    entropy_file = open(f"{trans_file}_entropy.txt", 'w')

    start = time.time()
    with torch.no_grad():
        print('translate')
        for x_data, x_mask, cur_line, iloop in valid_iter:
            if args.model == 'tm':
                samples, enc_entropy_list, self_entropy_list, cross_entropy_list = translate_tm(model, x_data, x_mask, args.max_length)
                
                # 배치별 엔트로피 값을 로그 파일에 기록
                entropy_file.write(f"Batch {batch_count}, Loop {iloop}:\n")
                entropy_file.write(f"Encoder entropy: {enc_entropy_list}\n")
                entropy_file.write(f"Decoder self entropy: {self_entropy_list}\n")
                entropy_file.write(f"Decoder cross entropy: {cross_entropy_list}\n\n")
                
                # 각 레이어별 누적 엔트로피 계산
                batch_count += 1
                
                # 인코더 엔트로피 누적
                for i, ent in enumerate(enc_entropy_list):
                    if isinstance(ent, torch.Tensor):
                        ent = ent.cpu().detach().item() if ent.numel() == 1 else ent.cpu().detach().mean().item()
                    if i not in enc_entropy_sums:
                        enc_entropy_sums[i] = 0.0
                    enc_entropy_sums[i] += ent
                
                # 디코더 셀프 어텐션 엔트로피 누적
                for i, ent in enumerate(self_entropy_list):
                    if isinstance(ent, torch.Tensor):
                        ent = ent.cpu().detach().item() if ent.numel() == 1 else ent.cpu().detach().mean().item()
                    if i not in self_entropy_sums:
                        self_entropy_sums[i] = 0.0
                    self_entropy_sums[i] += ent
                
                # 디코더 크로스 어텐션 엔트로피 누적
                for i, ent in enumerate(cross_entropy_list):
                    if isinstance(ent, torch.Tensor):
                        ent = ent.cpu().detach().item() if ent.numel() == 1 else ent.cpu().detach().mean().item()
                    if i not in cross_entropy_sums:
                        cross_entropy_sums[i] = 0.0
                    cross_entropy_sums[i] += ent
           
            for k in range(samples.shape[0]): # over the batch
                sentence = ids2words(trg_inv_dict, samples[k,:], eos_id=Const.EOS)
                sentence = unbpe(sentence)
                if valid: 
                    mb_subprocess.stdin.write(sentence + '\n')
                    mb_subprocess.stdin.flush()
                else:
                    fp.write(sentence+'\n')
    
    # 평균 엔트로피 계산 및 출력
    if not valid and batch_count > 0:
        entropy_file.write("\n" + "="*50 + "\n")
        entropy_file.write("AVERAGE ENTROPY VALUES PER LAYER\n")
        entropy_file.write("="*50 + "\n\n")
        
        # 인코더 레이어별 평균 엔트로피
        entropy_file.write("Encoder Average Entropy per Layer:\n")
        for layer, total in sorted(enc_entropy_sums.items()):
            avg = total / batch_count
            entropy_file.write(f"  Layer {layer}: {avg:.6f}\n")
        
        # 디코더 셀프 어텐션 레이어별 평균 엔트로피
        entropy_file.write("\nDecoder Self-Attention Average Entropy per Layer:\n")
        for layer, total in sorted(self_entropy_sums.items()):
            avg = total / batch_count
            entropy_file.write(f"  Layer {layer}: {avg:.6f}\n")
        
        # 디코더 크로스 어텐션 레이어별 평균 엔트로피
        entropy_file.write("\nDecoder Cross-Attention Average Entropy per Layer:\n")
        for layer, total in sorted(cross_entropy_sums.items()):
            avg = total / batch_count
            entropy_file.write(f"  Layer {layer}: {avg:.6f}\n")
        
        # 콘솔에도 평균 엔트로피 출력
        print("batchcount",batch_count)
        print("\nAVERAGE ENTROPY VALUES PER LAYER:")
        print("Encoder Average Entropy per Layer:")
        for layer, total in sorted(enc_entropy_sums.items()):
            avg = total / batch_count
            print(f"  Layer {layer}: {avg:.6f}")
        
        print("\nDecoder Self-Attention Average Entropy per Layer:")
        for layer, total in sorted(self_entropy_sums.items()):
            avg = total / batch_count
            print(f"  Layer {layer}: {avg:.6f}")
        
        print("\nDecoder Cross-Attention Average Entropy per Layer:")
        for layer, total in sorted(cross_entropy_sums.items()):
            avg = total / batch_count
            print(f"  Layer {layer}: {avg:.6f}")
    
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
        entropy_file.close()
        print('time / sentences: %s / %d' % (timeSince(start), iloop))
    
    return ret