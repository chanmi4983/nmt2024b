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
from collections import defaultdict

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

import numpy as np
import csv

import csv

def batch_layer_meanratio_row(resratio_logs, only_prefix=None):
    """
    한 배치(=한 문장)에서 layer_id별 mean_ratio를 dict로 반환
    반환: {layer_id: mean_ratio}
    """
    acc = defaultdict(list)

    for item in resratio_logs:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue

        layer_id = str(item[0])
        ratio = item[-1]  # 마지막 원소를 ratio로 사용(현재 코드와 동일한 가정)

        if only_prefix is not None and not layer_id.startswith(only_prefix):
            continue

        if ratio is None:
            continue

        acc[layer_id].append(float(ratio))

    # 배치 내에서 동일 layer가 여러번 기록되면 평균
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) > 0}

def batch_decoder_global_laststep_meanratio_row_from_step(step_logs, only_prefix="Dec_"):
    # 1) 전역 마지막 step_T
    max_step = None
    for item in step_logs:
        if not isinstance(item, (list, tuple)) or len(item) != 6:
            continue
        layer_id, step_T, *_ = item
        layer_id = str(layer_id)
        if only_prefix is not None and not layer_id.startswith(only_prefix):
            continue
        step_T = int(step_T)
        if (max_step is None) or (step_T > max_step):
            max_step = step_T

    if max_step is None:
        return {}

    # 2) 그 step_T의 sent_ratio(=그 step의 문장 mean ratio)만 layer별로 저장
    out = {}
    for item in step_logs:
        if not isinstance(item, (list, tuple)) or len(item) != 6:
            continue
        layer_id, step_T, res_n, br_n, ratio_tok, sent_ratio = item
        layer_id = str(layer_id)
        if only_prefix is not None and not layer_id.startswith(only_prefix):
            continue
        if int(step_T) == max_step:
            out[layer_id] = float(sent_ratio)

    return out



def write_wide_csv(batch_rows, all_layers, csv_path):
    """
    batch_rows: [{"_batch": "batch 0", "Dec_CrossAttn_L0":0.1, ...}, ...]
    all_layers: 컬럼으로 쓸 layer_id 리스트
    """
    header = ["batch"] + all_layers

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in batch_rows:
            bname = row.get("_batch", "")
            line = [bname]
            for layer in all_layers:
                val = row.get(layer, "")
                line.append(f"{val:.6f}" if isinstance(val, (float, int)) else val)
            writer.writerow(line)


def gather_step_detail_logs(model):
    """
    모든 모듈에서 step_detail_log를 모아서 반환
    item: (layer_id, step_T, res_norm[T], br_norm[T], ratio_tok[T], sent_ratio)
    """
    logs = []
    for m in model.modules():
        if hasattr(m, "step_detail_log"):
            logs.extend(m.step_detail_log)
    return logs


def write_step_detail_summary(step_logs, fp_log, max_steps=None, max_print_tokens=8, only_layers=None):
    """
    step_logs: gather_step_detail_logs(model) 결과
    fp_log: file handle

    max_steps: step 몇 개까지만 출력할지 (None이면 전체)
    max_print_tokens: token별 ratio를 앞에서 몇 개까지 보여줄지
    only_layers: 특정 layer_id만 보고 싶으면 set/list로 전달 (None이면 전체)

    출력 형식:
      layer_id | step_T | sent_ratio | ratio_head | ratio_last
    """
    fp_log.write("=== Step-wise detail (debug) ===\n")
    fp_log.write("format: layer_id | step_T | sent_ratio | ratio_head[0:k] | ratio_last\n")

    by_layer = {}
    for item in step_logs:
        if not isinstance(item, (list, tuple)) or len(item) != 6:
            continue
        layer_id, step_T, res_n, br_n, ratio_tok, sent_ratio = item

        if only_layers is not None and layer_id not in only_layers:
            continue

        by_layer.setdefault(layer_id, []).append((int(step_T), ratio_tok, float(sent_ratio)))

    for layer_id in sorted(by_layer.keys()):
        rows = by_layer[layer_id]
        rows.sort(key=lambda x: x[0])  # step_T 오름차순

        if max_steps is not None:
            rows = rows[:max_steps]

        for step_T, ratio_tok, sent_ratio in rows:
            rt = ratio_tok.numpy()  # (T,)
            k = min(max_print_tokens, len(rt))
            head = rt[:k]
            tail = float(rt[-1]) if len(rt) > 0 else float("nan")

            fp_log.write(
                f"{layer_id}\t"
                f"step_T={step_T}\t"
                f"sent_ratio={sent_ratio:.6f}\t"
                f"ratio_head={head}\t"
                f"ratio_last={tail:.6f}\n"
            )

    fp_log.write("\n")
    fp_log.flush()

def write_layerwise_resratio_mean(logs, fp_log):
    """
    logs: list of tuples
      case A (old format): (layer_id, res_n, br_n, out_n, ratio)
      case B (new format): (layer_id, out_n, ratio)  # out_n = used token count
    fp_log: file handle (open(..., 'w') or 'a')
    """

    # layer_id별로 모으기
    by_layer = {}
    for item in logs:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue

        if len(item) == 5:
            layer_id, res_n, br_n, out_n, ratio = item
        elif len(item) == 3:
            layer_id, out_n, ratio = item
            res_n, br_n = None, None
        else:
            # 길이가 애매하면 앞 3개만이라도 맞춰서 처리
            layer_id = item[0]
            out_n = item[1]
            ratio = item[-1]
            res_n, br_n = None, None

        by_layer.setdefault(layer_id, []).append((res_n, br_n, out_n, ratio))

    # 출력
    fp_log.write("=== Layerwise residual ratio summary ===\n")
    fp_log.write("format: layer_id | n_sent | mean_ratio | std_ratio | mean_n_tokens"
                 " | (optional) mean_res_norm | mean_br_norm\n")

    for layer_id in sorted(by_layer.keys()):
        rows = by_layer[layer_id]

        ratios = [float(r[3]) for r in rows if r[3] is not None]
        ns = [float(r[2]) for r in rows if r[2] is not None]

        mean_ratio = float(np.mean(ratios)) if len(ratios) > 0 else float("nan")
        std_ratio = float(np.std(ratios)) if len(ratios) > 0 else float("nan")
        mean_n = float(np.mean(ns)) if len(ns) > 0 else float("nan")

        # res_n, br_n이 있는 경우에만 평균 내서 출력
        res_list = [float(r[0]) for r in rows if r[0] is not None]
        br_list = [float(r[1]) for r in rows if r[1] is not None]
        mean_res = float(np.mean(res_list)) if len(res_list) > 0 else None
        mean_br = float(np.mean(br_list)) if len(br_list) > 0 else None

        if mean_res is None or mean_br is None:
            fp_log.write(
                f"{layer_id}\t"
                f"n_sent={len(rows)}\t"
                f"mean_ratio={mean_ratio:.6f}\t"
                f"std_ratio={std_ratio:.6f}\t"
                f"mean_n_tokens={mean_n:.2f}\n"
            )
        else:
            fp_log.write(
                f"{layer_id}\t"
                f"n_sent={len(rows)}\t"
                f"mean_ratio={mean_ratio:.6f}\t"
                f"std_ratio={std_ratio:.6f}\t"
                f"mean_n_tokens={mean_n:.2f}\t"
                f"mean_res_norm={mean_res:.6f}\t"
                f"mean_br_norm={mean_br:.6f}\n"
            )

    fp_log.write("\n")
    fp_log.flush()

def enable_step_detail(model, enabled=True):
    for layer in model.encoder.layer_stack:
        layer.self_attn.collect_step_detail = enabled
        layer.ff_layer.collect_step_detail = enabled
    for layer in model.decoder.layer_stack:
        layer.self_attn.collect_step_detail = enabled
        layer.enc_attn.collect_step_detail = enabled
        layer.ff_layer.collect_step_detail = enabled


def clear_step_detail(model):
    for layer in model.encoder.layer_stack:
        layer.self_attn.step_detail_log.clear()
        layer.ff_layer.step_detail_log.clear()
    for layer in model.decoder.layer_stack:
        layer.self_attn.step_detail_log.clear()
        layer.enc_attn.step_detail_log.clear()
        layer.ff_layer.step_detail_log.clear()


def print_step_detail_for_one_layer(model, which="Dec_SelfAttn_L0", max_steps=None):
    # 디코더 self-attn L0 예시: model.decoder.layer_stack[0].self_attn
    # 원하는 layer_id로 찾아서 출력
    modules = []
    for layer in model.encoder.layer_stack:
        modules += [layer.self_attn, layer.ff_layer]
    for layer in model.decoder.layer_stack:
        modules += [layer.self_attn, layer.enc_attn, layer.ff_layer]

    target = None
    for m in modules:
        if getattr(m, "layer_id", None) == which:
            target = m
            break
    if target is None:
        print(f"[print_step_detail] layer_id not found: {which}")
        return

    logs = target.step_detail_log
    if max_steps is not None:
        logs = logs[:max_steps]

    for (layer_id, step_T, res_n, br_n, ratio_tok, sent_ratio) in logs:
        print(f"[{layer_id}] step_T={step_T}  sent_ratio={sent_ratio:.6f}")
        print("  res_norm :", res_n.numpy())
        print("  br_norm  :", br_n.numpy())
        print("  ratio_tok:", ratio_tok.numpy())



def enable_resratio_collect(model, flag=True, clear=True):
    for m in model.modules():
        if hasattr(m, "collect_res_ratio"):
            m.collect_res_ratio = flag
        if clear and hasattr(m, "res_ratio_log"):
            m.res_ratio_log.clear()

def gather_resratio_logs(model):
    logs = []
    for m in model.modules():
        if hasattr(m, "res_ratio_log"):
            logs.extend(m.res_ratio_log)
    return logs

def clear_resratio_logs(model):
    for m in model.modules():
        if hasattr(m, "res_ratio_log"):
            m.res_ratio_log.clear()


def print_layerwise_resratio_mean(logs):
    acc = defaultdict(list)
    # logs: (layer_id, res_n, br_n, out_n, ratio)
    for layer_id, res_n, br_n, out_n, ratio in logs:
        acc[layer_id].append((res_n, br_n, out_n, ratio))

    print("\n=== Residual Add Ratio (mean over dataset) ===")
    for layer_id in sorted(acc.keys()):
        vals = acc[layer_id]
        res_m = sum(v[0] for v in vals) / len(vals)
        br_m  = sum(v[1] for v in vals) / len(vals)
        out_m = sum(v[2] for v in vals) / len(vals)
        rt_m  = sum(v[3] for v in vals) / len(vals)
        print(f"{layer_id}\tres={res_m:.6f}\tbranch={br_m:.6f}\tout={out_m:.6f}\tratio={rt_m:.6f}\tn={len(vals)}")
    print("=== End ===\n")


def translate_tm(model, x_data, x_mask, max_length):
    x_data = CudaVariable(torch.LongTensor(x_data)).transpose(0,1) # B T
    x_mask = CudaVariable(torch.FloatTensor(x_mask)).transpose(0,1) # B T

    #x_data = x_data[:,1:] # remove BOS
    #x_mask = x_mask[:,1:] # remove BOS

    Bn, Tx = x_data.size()
    enc_out = model.encoder(x_data, x_mask)
    
    y_hat0 = CudaVariable(torch.ones((Bn,1))*Const.BOS).type(torch.cuda.LongTensor)
    y_hat = y_hat0
    EOSs = torch.zeros((Bn, 1)).to(device)
    for yi in range(max_length):
        len_dec_seq = yi + 1
        dec_seq = y_hat.view(Bn, -1)
        y_mask = torch.ones((Bn,len_dec_seq), device=x_mask.device)

        dec_out = model.decoder(dec_seq, y_mask, enc_out, x_mask) # Bn T Word

        topv, yt = dec_out.topk(1, dim=2)
        yt = yt.view(Bn, yt.size(1)) # B Ty
        y_hat = torch.cat((y_hat0, yt), dim=1) # B Ty+1
        EOS1 = torch.eq(yt[:,-1], Const.EOS).view(Bn, 1) # B 1 
        EOSs = EOSs + EOS1.type(torch.cuda.FloatTensor)
        if yi > 0 and torch.sum(torch.gt(EOSs,0)) >= Bn:
            break

    y_hat = y_hat.cpu().detach().numpy()[:,1:] # ignore BOS

    return y_hat


def translate_file(model, src_file, s_dict_file, trg_file, t_dict_file, trans_file, args, valid=None):
    model.eval()

    # 요약 ratio 로그 ON
    enable_resratio_collect(model, True, clear=True)

    # ✅ (중요) step detail 로그 ON  ← 이게 빠져있어서 step 로그가 비어있던 것
    enable_step_detail(model, True)

    log_path = args.trans_file + ".midEnKrFullText_ver2.txt"
    fp_log = open(log_path, "w", encoding="utf-8")

    print('data loader')
    valid_iter = TextIterator(
        src_file, s_dict_file,
        batch_size=1,
        maxlen=1000,
        ahead=1,
        resume_num=0,
        just_one_epoch=1,
        const_id=Const
    )

    trg_dict = read_dict(t_dict_file, const_id=Const)
    trg_inv_dict = {vv: kk for kk, vv in trg_dict.items()}

    # translate
    if valid:
        multibleu_cmd = ["perl", args.bleu_script, trg_file, "<"]
        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE,
                              universal_newlines=True, encoding='utf-8')
    else:
        fp = open(trans_file, 'w')

    start = time.time()

    # ===== wide CSV 누적용 =====
    wide_rows = []      # 배치별 row(dict)
    layer_set = set()   # 전체 layer_id 컬럼 수집

    with torch.no_grad():
        print('translate')
        batch_idx = 0

        for x_data, x_mask, cur_line, iloop in valid_iter:

            # 이번 문장 시작 전: 이전 문장 로그 제거
            clear_resratio_logs(model)
            clear_step_detail(model)

            # 번역 수행
            if args.model == 'tm':
                samples = translate_tm(model, x_data, x_mask, args.max_length)
            else:
                samples = translate_tm(model, x_data, x_mask, args.max_length)

            # 번역 결과 저장
            for k in range(samples.shape[0]):
                sentence = ids2words(trg_inv_dict, samples[k, :], eos_id=Const.EOS)
                sentence = unbpe(sentence)

                if valid:
                    mb_subprocess.stdin.write(sentence + '\n')
                    mb_subprocess.stdin.flush()
                else:
                    fp.write(sentence + '\n')

            # # ====== 요약 로그 출력 ======
            # logs = gather_resratio_logs(model)
            # fp_log.write(f"Batch {batch_idx}, Loop {iloop}:\n")
            # write_layerwise_resratio_mean(logs, fp_log)

            # ====== step detail 로그 출력 ======
            step_logs = gather_step_detail_logs(model)

            # ===== (추가) batch x layer wide row 누적 =====
            row_dict = batch_decoder_global_laststep_meanratio_row_from_step(step_logs, only_prefix="Dec_")
            row_dict["_batch"] = f"batch {batch_idx}"
            wide_rows.append(row_dict)

            for k in row_dict.keys():
                if k != "_batch":
                    layer_set.add(k)




            # 디버그용: step 로그가 실제로 쌓였는지 확인
            fp_log.write(f"(debug) step_logs_len={len(step_logs)}\n")

            # 마지막 step 기준 mean_ratio dict (Dec_만)
            last_dict = batch_decoder_global_laststep_meanratio_row_from_step(step_logs, only_prefix="Dec_")

            fp_log.write(f"Batch {batch_idx}, Loop {iloop}:\n")
            fp_log.write("=== Last-step residual ratio summary (Dec_) ===\n")
            fp_log.write("format: layer_id | last_step_mean_ratio\n")
            for layer_id in sorted(last_dict.keys()):
                fp_log.write(f"{layer_id}\tmean_ratio={last_dict[layer_id]:.6f}\n")
            fp_log.write("\n")
            fp_log.flush()

            only_layers = None  # 예: {"Dec_SelfAttn_L0", "Dec_CrossAttn_L0", "Dec_FFN_L0"} 로 제한 가능

            # write_step_detail_summary(
            #     step_logs,
            #     fp_log,
            #     max_steps=None,
            #     max_print_tokens=1000,
            #     only_layers=only_layers
            # )

            fp_log.write("\n")
            fp_log.flush()

            print(f"배치 {batch_idx + 1} 완료")
            batch_idx += 1

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

    fp_log.close()

    # ===== (추가) wide CSV 저장 =====
    csv_path = args.trans_file + ".midbest_ver2.csv"
    all_layers = sorted(layer_set)
    write_wide_csv(wide_rows, all_layers, csv_path)


    # 다음 호출에 영향 없게 OFF
    enable_resratio_collect(model, False, clear=False)
    enable_step_detail(model, False)

    return ret
