# -*- coding: utf-8 -*-

# H. Choi, hchoi@handong.edu
from __future__ import unicode_literals, print_function, division

import math
import numpy as np
import pandas as pd
import random
import copy
import os
from io import open
import time
import re
from subprocess import Popen, PIPE

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn.functional as F

from libs.utils import timeSince, ids2words, unbpe, PostProcess
from libs.layers import CudaVariable

import nmt_const as Const
from Beam import Beam, Beam_Sampling, LengthPenaltyBeam

from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer, MosesDetokenizer

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def translate_tm(model, x_data, x_mask, args):
    x_data = CudaVariable(torch.LongTensor(x_data)) # B T
    x_mask = CudaVariable(torch.LongTensor(x_mask)) # B T

    Bn, Tx = x_data.size()
    enc_out = model.encoder(x_data, x_mask)

    y_hat0 = CudaVariable(torch.ones((Bn,1))*Const.BOS).type(torch.cuda.LongTensor)
    y_hat = y_hat0
    EOSs = torch.zeros((Bn, 1)).to(device)
    for yi in range(args.max_length):
        len_dec_seq = yi + 1
        dec_seq = y_hat.view(Bn, -1)
        y_mask = torch.ones((Bn,len_dec_seq), device=x_mask.device)

        dec_out = model.decoder(dec_seq, y_mask, enc_out, x_mask) # Bn T Word
        dec_out = model.logit_layer(dec_out)

        topv, yt = dec_out.topk(1, dim=2)
        yt = yt.view(Bn, yt.size(1)) # B Ty
        if args.strict_generation == 1:
            y_hat = torch.cat((y_hat, yt[:,-1].unsqueeze(1)), dim=1) # B Ty+1
        else:
            y_hat = torch.cat((y_hat0, yt), dim=1) # B Ty+1
        EOS1 = torch.eq(yt[:,-1], Const.EOS).view(Bn, 1) # B 1 
        EOSs = EOSs + EOS1.type(torch.cuda.FloatTensor)
        if yi > 0 and torch.sum(torch.gt(EOSs,0)) >= Bn:
            break

    y_hat = y_hat.cpu().detach().numpy()[:,1:] # ignore BOS

    return y_hat


def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, Bs):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * Bs, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor

def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list, Bs):
    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.cuda.LongTensor(active_inst_idx, device=device)

    active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, Bs)
    active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, Bs)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return active_src_seq, active_src_enc, active_inst_idx_to_position_map

def beam_decode_step(model, inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, Bs, sampling):
    ''' Decode and update beam status, and then return active beam idx '''

    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def prepare_beam_dec_pos(len_dec_seq, n_active_inst, Bs):
        dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=device)
        dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * Bs, 1)
        return dec_partial_pos

    def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, Bs, sampling):
        # !!! Here src_seq should be src_mask, and enc_output should be in 3rd
        src_seq = (src_seq != Const.PAD)
        dec_out = model.decoder(dec_seq, dec_pos, enc_output, src_seq)
        dec_out = model.logit_layer(dec_out)
        dec_out = dec_out[:, -1, :]  # Pick the last step: (bh * bm) * d_h
        #word_prob = F.log_softmax(model.trg_word_proj(dec_out), dim=1)
        if sampling == 1:
            word_prob = F.softmax(dec_out, dim=1)
        else:
            word_prob = F.log_softmax(dec_out, dim=1)
        word_prob = word_prob.view(n_active_inst, Bs, -1)

        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)

    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, Bs)
    word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, Bs, sampling)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(
        inst_dec_beams, word_prob, inst_idx_to_position_map)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        #all_hyp += [hyps]
        all_hyp += hyps
    return all_hyp, all_scores

def translate_tm_beam(model, x_data, x_mask, args): # not working
    start_time = time.time()

    x_data = CudaVariable(torch.LongTensor(x_data)) # T B
    x_mask = CudaVariable(torch.LongTensor(x_mask)) # T B

    #x_data = x_data.transpose(0, 1) # B T
    #x_mask = x_mask.transpose(0, 1) # B T
    xm = (x_data.data.ne(Const.PAD)).type(torch.cuda.FloatTensor)

    Bs = args.beam_width
    Bn, Tx = x_data.size()
    enc_out = model.encoder(x_data, x_mask) * xm.unsqueeze(2)
    
    #-- Repeat data for beam search
    n_inst, Ts, d_h = enc_out.size()
    x_data = x_data.repeat(1, Bs).view(n_inst * Bs, Ts)
    enc_out = enc_out.repeat(1, Bs, 1).view(n_inst * Bs, Ts, d_h)

    #-- Prepare beams
    if args.sampling == 0:
        inst_dec_beams = [Beam(Bs, device=device) for _ in range(n_inst)]
    else:
        inst_dec_beams = [Beam_Sampling(Bs, args.sampling_topk, device=device)\
                         for _ in range(n_inst)]

    #-- Bookkeeping for active or not
    active_inst_idx_list = list(range(n_inst))
    inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    #-- Decode
    times = torch.ones(Bn)*(time.time() - start_time)
    for len_dec_seq in range(1, args.max_length + 1):
        active_inst_idx_list = beam_decode_step(model, inst_dec_beams, len_dec_seq, x_data, enc_out, inst_idx_to_position_map, Bs, args.sampling)


        times[:len(active_inst_idx_list)] = time.time() - start_time

        if not active_inst_idx_list:
            break  # all instances have finished their path to <EOS>

        x_data, enc_out, inst_idx_to_position_map = collate_active_info(
            x_data, enc_out, inst_idx_to_position_map, active_inst_idx_list, Bs)

    n_best = 1
    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)

    #y_hat = batch_hyp[0]#cpu().numpy().flatten().tolist()
    #return y_hat
    return batch_hyp, times

def beam_decode_step_lenpen(model, inst_dec_beams, len_dec_seq, src_seq, enc_output,\
                             inst_idx_to_position_map, Bs, sampling):
    ''' Decode and update beam status, and then return active beam idx '''

    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def prepare_beam_dec_pos(len_dec_seq, n_active_inst, Bs):
        dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=device)
        dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * Bs, 1)
        return dec_partial_pos

    def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, Bs, sampling):
        # !!! Here src_seq should be src_mask, and enc_output should be in 3rd
        src_seq = (src_seq != Const.PAD)
        dec_out = model.decoder(dec_seq, dec_pos, enc_output, src_seq)
        dec_out = model.logit_layer(dec_out)
        dec_out = dec_out[:, -1, :]  # Pick the last step: (bh * bm) * d_h
        #word_prob = F.log_softmax(model.trg_word_proj(dec_out), dim=1)
        if sampling == 1:
            word_prob = F.softmax(dec_out, dim=1)
        else:
            word_prob = F.log_softmax(dec_out, dim=1)
        word_prob = word_prob.view(n_active_inst, Bs, -1)

        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, len_dec_seq):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], len_dec_seq)
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)

    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, Bs)
    word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, Bs, sampling)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(
        inst_dec_beams, word_prob, inst_idx_to_position_map, len_dec_seq)

    return active_inst_idx_list

def translate_tm_beam_lenpen(model, x_data, x_mask, args): # not working
    start_time = time.time()

    x_data = CudaVariable(torch.LongTensor(x_data)) # T B
    x_mask = CudaVariable(torch.LongTensor(x_mask)) # T B

    #x_data = x_data.transpose(0, 1) # B T
    #x_mask = x_mask.transpose(0, 1) # B T
    xm = (x_data.data.ne(Const.PAD)).type(torch.cuda.FloatTensor)

    Bs = args.beam_width
    Bn, Tx = x_data.size()
    enc_out = model.encoder(x_data, x_mask) * xm.unsqueeze(2)
    
    #-- Repeat data for beam search
    n_inst, Ts, d_h = enc_out.size()
    x_data = x_data.repeat(1, Bs).view(n_inst * Bs, Ts)
    enc_out = enc_out.repeat(1, Bs, 1).view(n_inst * Bs, Ts, d_h)

    #-- Prepare beams
    #if args.sampling == 0:
    #    inst_dec_beams = [Beam(Bs, device=device) for _ in range(n_inst)]
    #else:
    #    inst_dec_beams = [Beam_Sampling(Bs, args.sampling_topk, device=device)\
    #                     for _ in range(n_inst)]
    inst_dec_beams = [LengthPenaltyBeam(Bs, device=device) for _ in range(n_inst)]

    #-- Bookkeeping for active or not
    active_inst_idx_list = list(range(n_inst))
    inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    #-- Decode
    times = torch.ones(Bn)*(time.time() - start_time)
    for len_dec_seq in range(1, args.max_length + 1):
        active_inst_idx_list = beam_decode_step_lenpen(model, inst_dec_beams, len_dec_seq, x_data, enc_out, inst_idx_to_position_map, Bs, sampling=0)


        times[:len(active_inst_idx_list)] = time.time() - start_time

        if not active_inst_idx_list:
            break  # all instances have finished their path to <EOS>

        x_data, enc_out, inst_idx_to_position_map = collate_active_info(
            x_data, enc_out, inst_idx_to_position_map, active_inst_idx_list, Bs)

    n_best = 1
    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)

    #y_hat = batch_hyp[0]#cpu().numpy().flatten().tolist()
    #return y_hat
    return batch_hyp, times


def bleu_score(ref_file, trans_file):
    multibleu_cmd = ["perl", "./tools/multi-bleu.perl", ref_file, "<"]
    mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True)
    with open(trans_file, 'r') as fin:
        mb_subprocess.stdin.write(''.join(line for line in fin))
    mb_subprocess.stdin.flush()
    mb_subprocess.stdin.close()
    output = mb_subprocess.stdout.readline()
    mb_subprocess.terminate()

    try:
        bleu = float(output[output.find('=')+2:output.find(',')])
    except ValueError:
        bleu = -1.0
    return bleu



def multi_validation(model_ddp, valid_iter, args, rank):
    # The BLEU score computed by this function is not 100% correct. Run Test_SacreBLEU for the final BLEU score. This is for fast validation.

    def delete_eos_sentences(sample):
        B, T = sample.size()
        eos_index = list(range(B))
        for b in range(B):
            eos_test = torch.eq(sample[b], Const.EOS).long()
            if torch.sum(eos_test) == T:
                eos_index.remove(b)
        return torch.index_select(sample, dim=0, index=torch.tensor(eos_index, device=sample.device))


    model_ddp.eval()

    src_dict2 = valid_iter.vocab_dict['src']
    trg_dict2 = valid_iter.vocab_dict['trg']

    trg_inv_dict = dict()
    for kk, vv in trg_dict2.items():
        trg_inv_dict[vv] = kk

    if rank == 0:
        y_references = []
        y_generation = []

    print_iloop = random.randint(0,valid_iter.num_iloop)

    loss_total = 0
    latent_norms_total = 0
    latent_stds_total = 0
    avg_time_total = 0

    max_len = model_ddp.module.test_max_len
    B_max = valid_iter.max_rank_batch_size

    start = time.time()
    with torch.no_grad():
        valid_iter.initialize()
        for iloop, [x_data, x_mask, y_data, y_mask, idxes, epoch_run] in enumerate(valid_iter):
            print("{} Iterations..".format(iloop), end="\r")
            x_data, x_mask, y_data, y_mask = x_data[0], x_mask[0], y_data[0], y_mask[0]

            if args.beam_width > 1:
                raise SyntaxError("Multi-GPU Validation is not implemented for beam search")
            else:
                (samples, tmp_loss, tmp_avg_time, y_data) \
                             = model_ddp(x_data, x_mask, y_data, y_mask, B_max, 'multi_validation')

                loss_total += tmp_loss.item()
                avg_time_total += tmp_avg_time.item()

                sample_list = [torch.zeros((B_max, max_len), dtype=torch.long,\
                                 device=samples.device) for _ in range(args.world_size)]
                y_data_list = [torch.zeros((B_max, max_len), dtype=torch.long,\
                                 device=y_data.device) for _ in range(args.world_size)]

                if args.world_size >= 2:
                    if rank == 0:
                        dist.gather(samples.contiguous(), sample_list)
                        dist.gather(y_data.contiguous(), y_data_list)
                    else:
                        dist.gather(samples.contiguous(), dst=0)
                        dist.gather(y_data.contiguous(), dst=0)
                    dist.barrier()

                else:
                    sample_list = [samples]
                    y_data_list = [y_data]

                if rank == 0:
                    sample_total = delete_eos_sentences(sample_list[0])
                    y_data_total = delete_eos_sentences(y_data_list[0])

                    for i in range(1, args.world_size):
                        sample_total = torch.cat((sample_total,\
                                            delete_eos_sentences(sample_list[i])),dim=0)
                        y_data_total = torch.cat((y_data_total,\
                                            delete_eos_sentences(y_data_list[i])),dim=0)

                    sample_total = sample_total.cpu().tolist()
                    y_data_total = y_data_total.cpu().tolist()

            if iloop == print_iloop and rank == 0:
                print_idx = random.randint(0, len(sample_total)-1)
                gen_sentence = ids2words(trg_inv_dict, sample_total[print_idx],\
                                             eos_id=Const.EOS)
                gen_sentence = PostProcess(gen_sentence, Const.PAD_WORD)
                gen_sentence = unbpe(gen_sentence)

                ref_sentence = ids2words(trg_inv_dict, y_data_total[print_idx],\
                                             eos_id=Const.EOS)
                ref_sentence = PostProcess(ref_sentence, Const.PAD_WORD)
                ref_sentence = unbpe(ref_sentence)

                print("ILOOP : {}".format(iloop))
                print("GEN : ", gen_sentence)
                print("REF : ", ref_sentence)

            if rank == 0:
                for k in range(len(sample_total)): # over the batch
                    gen_sentence = ids2words(trg_inv_dict, sample_total[k], eos_id=Const.EOS)
                    gen_sentence = PostProcess(gen_sentence, Const.PAD_WORD)
                    gen_sentence = unbpe(gen_sentence)
                    y_generation.append(gen_sentence)

                    fwd_ref_sentence = ids2words(trg_inv_dict, y_data_total[k], eos_id=Const.EOS)
                    fwd_ref_sentence = PostProcess(fwd_ref_sentence, Const.PAD_WORD)
                    fwd_ref_sentence = unbpe(fwd_ref_sentence)
                    y_references.append(fwd_ref_sentence)

            dist.barrier()

            if epoch_run == False:
                break

    if rank == 0:
        if args.trg_lang == 'kr':
            y_bleu = BLEU(force=True, tokenize='ko-mecab')
        else:
            y_bleu = BLEU(force=True)

        y_references = [y_references]

        print("----Validation Results-----")
        detok_bleu = y_bleu.corpus_score(y_generation, y_references).score
        print('Detok BLEU: %f time / sentences: %s / %d' % (detok_bleu, timeSince(start), iloop))

        iloop += 1
        loss = loss_total / iloop
        avg_time = avg_time_total / iloop
        print("Valid Loss : {:.4f}".format(loss))
        print("Avg. Time  : {:.8f} (Sec)".format(avg_time))
        print("--------------------------")
    else:
        detok_bleu = 0
        loss = 0
        avg_time = 0

    return detok_bleu, loss, avg_time


def Test_SacreBLEU(model, test_iter, args):
    print("===================================")
    print("SacreBLEU Tokenizer : ", args.sacrebleu_tokenizer)
    print("Lowercase Insensitivity : ", bool(args.sacrebleu_lowercase))
    print("Beam Width : ", args.beam_width)
    model.eval()

    src_dict2 = test_iter.dataset.src_dict2
    trg_dict2 = test_iter.dataset.trg_dict2

    src_inv_dict = dict()
    for kk, vv in src_dict2.items():
        src_inv_dict[vv] = kk

    trg_inv_dict = dict()
    for kk, vv in trg_dict2.items():
        trg_inv_dict[vv] = kk

    print_iloop = random.randint(0,20)

    avg_time_total = 0

    y_references = test_iter.dataset.target_references
    y_token_references = test_iter.dataset.target_token_references
    y_generation = []
    y_token_generation = []

    target_md = MosesDetokenizer(lang=args.trg_lang)

    start = time.time()
    with torch.no_grad():
        for iloop, [x_data, x_mask, y_data, y_mask] in enumerate(test_iter):
            print("{} Iterations..".format(iloop), end="\r")
            if args.beam_width > 1:
                (B, _) = x_data.shape
                fwd_samples, fwd_times = translate_tm_beam_lenpen(model.model, x_data, x_mask, args)
                #fwd_samples, fwd_times = translate_tm_beam(model.model, x_data, x_mask, args)
                avg_time_total += fwd_times.mean()
            else:
                (B, _) = x_data.shape
                fwd_samples, _, tmp_avg_time\
                             = model(x_data, x_mask, y_data, y_mask, 'validation')

                avg_time_total += tmp_avg_time.item()

            if iloop == print_iloop:
                fwd_gen_sentence = ids2words(trg_inv_dict, fwd_samples[0], eos_id=Const.EOS)
                fwd_gen_sentence = PostProcess(fwd_gen_sentence, Const.PAD_WORD)
                fwd_gen_sentence = unbpe(fwd_gen_sentence)
                fwd_gen_sentence = target_md.detokenize(fwd_gen_sentence.strip().split())

                fwd_ref_sentence = y_references[0][B*iloop+0]

                print("ILOOP : {}".format(iloop))
                print("FWD_GEN : ", fwd_gen_sentence)
                print("FWD_REF : ", fwd_ref_sentence)

            for k in range(len(fwd_samples)): # over the batch
                fwd_gen_sentence = ids2words(trg_inv_dict, fwd_samples[k], eos_id=Const.EOS)
                fwd_gen_sentence = PostProcess(fwd_gen_sentence, Const.PAD_WORD)
                fwd_gen_sentence = unbpe(fwd_gen_sentence)
                y_token_generation.append(fwd_gen_sentence)
                fwd_gen_sentence = target_md.detokenize(fwd_gen_sentence.strip().split())
                y_generation.append(fwd_gen_sentence)

    avg_time = avg_time_total / iloop


    if args.trg_lang == 'kr':
        y_bleu = BLEU(force=True, tokenize='ko-mecab')
    else:
        y_bleu = BLEU(force=True, tokenize=args.sacrebleu_tokenizer,\
                     lowercase=bool(args.sacrebleu_lowercase))
    print("----Test Results-----")
    print("Fwd SacreBLEU score : {}".format(y_bleu.corpus_score(y_generation, y_references)))
    print("Fwd TokenBLEU score : {}".format(y_bleu.corpus_score(y_token_generation,\
                                                                 y_token_references)))
    print("Avg. Time    : {:.8f} (Sec)".format(avg_time))
    print("--------------------------")

