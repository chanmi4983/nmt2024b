# H. Choi, hchoi@handong.edu

import numpy as np
import random
import time
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import torch.distributed as dist
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import copy

from libs.layers import CudaVariable, GaussianNoise
import nmt_const as Const

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

class LabelSmoothingCrossEntropy(nn.Module):
    # Referenced from 
    # https://github.com/seominseok0429/label-smoothing-visualization-pytorch

    def __init__(self, epsilon):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, target):
        smoothing = self.epsilon
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss

def get_scale(nin, nout):
    return math.sqrt(6)/math.sqrt(nin+nout) # Xavier

def one_hot(input, class_tensor, num_classes):
    Bn, Tx = input.size()
    input = input.reshape(-1).unsqueeze(1)
    return (input == class_tensor.reshape(1, num_classes)).float().view(Bn, Tx, -1)

'''
class myEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(myEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input):
        

    def reset_parameters(self):
        scale = get_scale(1, self.embedding_dim)
        self.weight.data.uniform_(-scale, scale)
'''

class myEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(myEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.class_tensor = torch.arange(num_embeddings).cuda()

    def forward(self, input):
        # input : Bn, Tx
        if len(input.size()) == 2:
            Bn, Tx = input.size()
            input = one_hot(input, self.class_tensor, self.num_embeddings) # Bn*Tx, C
        else:
            Bn, Tx, _ = input.size()
        input = input.reshape(Bn*Tx, -1)
        out = torch.mm(input, self.weight) # Bn*Tx, Emb
        return out.view(Bn, Tx, -1)

    def reset_parameters(self):
        scale = get_scale(1, self.embedding_dim)
        self.weight.data.uniform_(-scale, scale)

class myLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(myLinear, self).__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        if self.in_features == self.out_features: # Identity
            self.weight.data.copy_(torch.eye(self.in_features))
        else:
            scale = get_scale(self.in_features, self.out_features)
            self.weight.data.uniform_(-scale, scale)

        if self.bias is not None:
            self.bias.data.zero_()

class ScaledDotProductAttention(nn.Module): 
    def __init__(self, dk, drop_p=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = float(dk)# ** 0.5 # this is better without 0.5
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, q, k, v, mask=None): # B H T E
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temper # B H Tq Tk
        if mask is not None:
            attn = attn.masked_fill(mask<0.1, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn) # B H Tq Tk 
        output = torch.matmul(attn, v) # B H Tv E 
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, dim_model, dk, dv, drop_p=0.):
        super(MultiHeadAttention, self).__init__()

        self.n_head, self.dk, self.dv = n_head, dk, dv

        self.mat_qs = myLinear(dim_model, n_head*dk, bias=False) 
        self.mat_ks = myLinear(dim_model, n_head*dk, bias=False)
        self.mat_vs = myLinear(dim_model, n_head*dv, bias=False)

        self.sdp_attn = ScaledDotProductAttention(dk, drop_p=drop_p)

        self.out_proj = myLinear(n_head*dv, dim_model)
        self.dropout = nn.Dropout(p=drop_p)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, q, k, v, attn_mask=None): # (QK')V # q = B T E
        n_head, dk, dv = self.n_head, self.dk, self.dv
        Bn, Tq, Tk, Tv = q.size(0), q.size(1), k.size(1), v.size(1)

        # TODO: merge the matrices and project q,k,v at the same time and split
        qnew = self.mat_qs(q).view(Bn, Tq, n_head, dk).transpose(1,2)
        k = self.mat_ks(k).view(Bn, Tk, n_head, dk).transpose(1,2)
        v = self.mat_vs(v).view(Bn, Tv, n_head, dv).transpose(1,2)

        if attn_mask is not None: #  B ? T -> B ? ? T 
            attn_mask = attn_mask.unsqueeze(1)
        
        output = self.sdp_attn(qnew, k, v, mask=attn_mask) # Bn H Ty E
        output = output.transpose(1, 2).contiguous().view(Bn, Tq, -1) # Bn Ty H*E
        output = self.dropout(self.out_proj(output))

        return self.layer_norm(output) + q

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, drop_p=0.):
        super().__init__()
        self.layer1 = myLinear(dim_model, dim_ff)
        self.layer2 = myLinear(dim_ff, dim_model)
        self.dropout = nn.Dropout(p=drop_p)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        output = self.layer2(F.relu(self.layer1(x)))
        output = self.dropout(output)
        return self.layer_norm(output) + x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len + 2 #BOS, EOS tokens
        self.get_pe(self.max_len)

    def forward(self, x, coeff=1.0):
        #self.max_len = 200 # don't need! but necessary for compatability with the previous model. 
        if self.training or x.size(1) <= self.max_len:
            pass
        else:
            #self.d_model = 512 # don't need! but 
            self.get_pe(x.size(1))
        x = x + coeff*self.pe[:, :x.size(1)]
        return self.dropout(x)

    def get_pe(self, max_len):
        scale = get_scale(1, self.d_model)
        pe = torch.zeros(max_len, self.d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position*div_term) * scale
        pe[:, 1::2] = torch.cos(position*div_term) * scale
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

class TM_EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ff, n_head, dk, dv, drop_p=0.):
        super(TM_EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p)
        self.ff_layer = FeedForward(dim_model, dim_ff, drop_p=drop_p)

    def forward(self, enc_in, x_mask=None):
        enc_out = self.self_attn(enc_in, enc_in, enc_in, attn_mask=x_mask)
        enc_out = self.ff_layer(enc_out)
        return enc_out

class TM_Encoder(nn.Module):
    def __init__(self, src_words_n, n_layers=6, n_head=8, dk=64, dv=64,
                    dim_wemb=512, dim_model=512, dim_ff=1024, drop_p=0., emb_noise=0., max_len=250):
        super(TM_Encoder, self).__init__()

        # first in : 
        self.src_emb = myEmbedding(src_words_n, dim_wemb)#, padding_idx=Const.PAD)
        self.pos_enc = PositionalEncoding(dim_wemb, max_len, drop_p)
        self.layer_norm = nn.LayerNorm(dim_wemb)
        # repeated layer
        self.layer_stack = nn.ModuleList([
            TM_EncoderLayer(dim_model, dim_ff, n_head, dk, dv, drop_p=drop_p)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_mask):
        src_mask = src_mask.unsqueeze(1) # B 1 T
        enc_out = self.src_emb(src_seq) # Word embedding look up # Bn Tx Emb
        enc_out = self.pos_enc(enc_out) # Position Encoding
        enc_out = self.layer_norm(enc_out) 

        for enc_layer in self.layer_stack:
            enc_out = enc_layer(enc_out, x_mask=src_mask)
        return enc_out

    def forward2(self, src_seq, src_mask, emb_scale):
        src_mask = src_mask.unsqueeze(1) # B 1 T
        enc_out = self.src_emb(src_seq) # Word embedding look up # Bn Tx Emb
        Bn, Tx, E = enc_out.size()        

        emb_scale = emb_scale.repeat(Bn,1).reshape(-1)
        emb_scale = emb_scale.unsqueeze(1).repeat(1,E)
        enc_out = enc_out.reshape(Bn*Tx, -1)
        enc_out = emb_scale * enc_out
        enc_out = enc_out.reshape(Bn, Tx, -1)

        enc_out = self.pos_enc(enc_out) # Position Encoding
        enc_out = self.layer_norm(enc_out) 

        for enc_layer in self.layer_stack:
            enc_out = enc_layer(enc_out, x_mask=src_mask)

        return enc_out

class TM_DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ff, n_head, dk, dv, drop_p=0.):
        super(TM_DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p)
        self.enc_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p)
        self.ff_layer = FeedForward(dim_model, dim_ff, drop_p=drop_p)

    def forward(self, dec_in, enc_out, y_mask=None, enc_mask=None):
        dec_out = self.self_attn(dec_in, dec_in, dec_in, attn_mask=y_mask)
        dec_out = self.enc_attn(dec_out, enc_out, enc_out, attn_mask=enc_mask)
        dec_out = self.ff_layer(dec_out)
        return dec_out

class TM_Decoder(nn.Module):
    def __init__(self, trg_words_n, n_layers=6, n_head=8, dk=64, dv=64,
            dim_wemb=512, dim_model=512, dim_ff=1024, drop_p=0., emb_noise=0., max_len=250):
        super(TM_Decoder, self).__init__()

        # first in
        self.dec_emb = myEmbedding(trg_words_n, dim_wemb)#, padding_idx=Const.PAD)
        self.pos_enc = PositionalEncoding(dim_wemb, max_len, drop_p)
        self.layer_norm = nn.LayerNorm(dim_wemb)
        # repeated layer
        self.layer_stack = nn.ModuleList([
            TM_DecoderLayer(dim_model, dim_ff, n_head, dk, dv, drop_p=drop_p)
            for _ in range(n_layers)])
        #self.trg_word_proj.weight = self.dec_emb.weight # Share the weight 

    def get_subsequent_mask(self, seq):
        if len(seq.size()) == 2: # seq is scalar values
            s0, s1 = seq.size()
        else:
            s0, s1, _ = seq.size() # seq is one-hot processed values
        mask = torch.tril(torch.ones((s1, s1), device=seq.device), diagonal=0)
        return mask.type(torch.cuda.FloatTensor).unsqueeze(0)   

    def forward(self, trg_seq, trg_mask, enc_out, src_mask):
        src_mask = src_mask.unsqueeze(1) # B 1 T
        trg_mask = trg_mask.unsqueeze(1) # B 1 T
        dec_out = self.dec_emb(trg_seq) # Word embedding look up
        dec_out = self.pos_enc(dec_out) # Posision Encoding
        dec_out = self.layer_norm(dec_out) 


        mh_attn_sub_mask = self.get_subsequent_mask(trg_seq) # lower traingle matrix 
        y_mask = trg_mask * mh_attn_sub_mask

        for dec_layer in self.layer_stack:
            dec_out = dec_layer(dec_out, enc_out, y_mask=y_mask, enc_mask=src_mask)

        return dec_out # B Ty E


class Transformer(nn.Module):
    def __init__(self, args=None):
        super(Transformer, self).__init__()
        self.class_tensor = torch.arange(args.trg_words_n).cuda()
        self.trg_words_n = args.trg_words_n
        self.max_len = 150 # 121 is fairseq's maximum sentence length considers during test in IWSLT2014 (there are longer sentence but filtered out)

        src_words_n, trg_words_n = args.src_words_n, args.trg_words_n
        dim_wemb, dim_model = args.dim_wemb, args.dim_model
        drop_p = args.dropout_p
        dim_ff, n_layers = args.tm_dim_ff, args.tm_n_layers
        n_head, dk, dv = args.tm_n_head, args.tm_dk, args.tm_dv
        assert dim_model == dim_wemb, 'dim_model == dim_wemb for residual connections'

        self.encoder=TM_Encoder(src_words_n, n_layers=n_layers, n_head=n_head,
                dim_wemb=dim_wemb, dim_model=dim_model, dim_ff=dim_ff, drop_p=drop_p, 
                emb_noise=args.emb_noise, max_len=self.max_len)
        self.decoder=TM_Decoder(trg_words_n, n_layers=n_layers, n_head=n_head,
                dim_wemb=dim_wemb, dim_model=dim_model, dim_ff=dim_ff, drop_p=drop_p, 
                emb_noise=args.emb_noise, max_len=self.max_len)
        self.logit_layer = nn.Linear(dim_model, trg_words_n)

        # Logit layer can share weight with encoder's embedding : It is better not to share
        #self.encoder.src_emb.weight = self.logit_layer.weight

        if args.joined_dictionary == 1:
            self.decoder.dec_emb.weight = self.encoder.src_emb.weight

        if args.label_smoothing > 0.0:
            self.criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.nll = nn.NLLLoss(reduction='none')

    def forward(self, x_data, x_mask, y_data, y_mask):
        if torch.is_tensor(x_data) == False:
            x_data = CudaVariable(torch.LongTensor(x_data)) # B T
            x_mask = CudaVariable(torch.FloatTensor(x_mask)) # B T
        if torch.is_tensor(y_data) == False:
            y_data = CudaVariable(torch.LongTensor(y_data)) # B T
            y_mask = CudaVariable(torch.FloatTensor(y_mask)) # B T

        y_target = y_data[:,1:] # label
        y_mask = y_mask[:,1:]
        y_in = y_data[:,:-1] # input as teacher forcing
        Bn, Ty = y_in.size()

        # encode and decode
        enc_out = self.encoder(x_data, x_mask) # B Tx E
        dec_out = self.decoder(y_in, y_mask, enc_out, x_mask) # B Ty E(num of words)
        out = self.logit_layer(dec_out)

        # loss
        loss = self.criterion(out.view(-1, out.size(2)), y_target.contiguous().view(-1)) 
        loss = loss * y_mask.contiguous().view(-1)

        return loss

    def decode_forward(self, enc_out, x_mask, y_data, y_mask):
        # Instead of x_data, it receives processed enc_out
        if torch.is_tensor(y_data) == False:
            y_data = CudaVariable(torch.LongTensor(y_data)) # B T
            y_mask = CudaVariable(torch.FloatTensor(y_mask)) # B T

        y_target = y_data[:,1:] # label
        y_mask = y_mask[:,1:]
        y_in = y_data[:,:-1] # input as teacher forcing
        Bn, Ty = y_in.size()

        # encode and decode
        dec_out = self.decoder(y_in, y_mask, enc_out, x_mask) # B Ty E(num of words)
        out = self.logit_layer(dec_out)

        # loss
        loss = self.criterion(out.view(-1, out.size(2)), y_target.contiguous().view(-1)) 
        loss = loss * y_mask.contiguous().view(-1)

        return loss

    def sample_idx(self, probs, train):
        gen_method = 'greedy'

        BnT, _ = probs.size()
        if gen_method == 'greedy':
            _, index = probs.topk(1, dim=1) # Bn, 1
        elif gen_method == 'sampling':
            m = torch.distributions.categorical.Categorical(probs)
            index = m.sample()
        elif gen_method == 'topk_sampling':
            topk_probs, topk_ids = probs.topk(self.k, dim=1) # best_probs : Bn, k
            topk_probs = topk_probs / torch.sum(topk_probs, dim=1).unsqueeze(1).repeat(1,self.k)
            m = torch.distributions.categorical.Categorical(topk_probs)
            tmp_index = m.sample()
            index = topk_ids[torch.arange(BnT), tmp_index.squeeze()].unsqueeze(1)
        return index

    def translate(self, enc_out, x_mask, CRT_coeff, train=True,\
                                 start_time=time.time()):
        Bn, Tx = x_mask.size()

        pad = (torch.ones((Bn,1))*Const.PAD).type(torch.long).cuda()
        pad_onehot = one_hot(pad, self.class_tensor, self.trg_words_n).squeeze() # Bn, C

        EOSs = torch.zeros((Bn, 1)).cuda()

        y_hat0 = (torch.ones((Bn,1))*Const.BOS).type(torch.long).cuda()
        dec_seq = copy.deepcopy(y_hat0)
        y_hat = one_hot(y_hat0, self.class_tensor, self.trg_words_n)
        dec_mask = CudaVariable(torch.ones((Bn,1))).type(torch.cuda.LongTensor)

        tmp_max_len = self.max_len

        times = torch.ones(Bn)*(time.time() - start_time)
        for yi in range(tmp_max_len):

            dec_out = self.decoder(dec_seq, dec_mask, enc_out, x_mask) # Bn, Tx, C
            dec_out = self.logit_layer(dec_out)
            Bn, T, C = dec_out.size()
            tmp_dec_out = dec_out.reshape(Bn*T, C)
            probs = F.softmax(tmp_dec_out, dim=1) # Bn*T, C
            index = self.sample_idx(probs, train=train) # Bn*T, 1

            last_index = index.reshape(Bn, T, 1)[:,-1,:]
            tmp_EOSs = torch.gt(EOSs,0).type(torch.long)
            tmp_dec_seq = (1-tmp_EOSs)*last_index + (tmp_EOSs)*pad
            #dec_seq = torch.cat((dec_seq, last_index), dim=1)
            dec_seq = torch.cat((dec_seq, tmp_dec_seq), dim=1) # Bn, T+1
            dec_mask = torch.cat((dec_mask, (1-tmp_EOSs)), dim=1) # Bn, T+1

            #EOS1 = torch.eq(last_index, Const.EOS).view(Bn, 1)
            EOS1 = torch.eq(tmp_dec_seq, Const.EOS).view(Bn, 1)
            for b in range(Bn):
                if EOSs[b] > 0:
                    continue
                elif EOS1[b] > 0:
                    times[b] = time.time() - start_time
            EOSs = EOSs + EOS1
            if yi > 0 and torch.sum(torch.gt(EOSs,0)) >= Bn:
                break

        for b in range(Bn):
            if EOSs[b] > 0:
                continue
            times[b] = time.time() - start_time

        return dec_seq, dec_mask, enc_out, times


class NormalNMT(nn.Module):
    def __init__(self, args=None, mean_batch=1):
        super(NormalNMT, self).__init__()
    
        self.src_lang = args.src_lang
        self.trg_lang = args.trg_lang

        self.model = Transformer(args=args)

        self.batch_sizes = []
        self.mean_batch = mean_batch

        self.test_max_len = 250

    def compute_norm(self, variable, mask):
        B, T, E = variable.size()
        norm = LA.norm(variable, dim=2)
        norm = (mask*norm).sum() / (mask.sum())
        return norm

    def compute_std(self, variable, mask):
        B, T, E = variable.size()
        mask = mask.unsqueeze(-1).repeat(1,1,E)

        variable = (mask*variable).sum(dim=1) / (mask.sum(dim=1))

        std = torch.std(variable, dim=0)
        std = std.mean()
        return std

    def compute_mean_batch_sizes(self, new_B):
        self.batch_sizes.append(new_B)

        if len(self.batch_sizes) > 1000: # 100 is arbitrarily selected number
            self.batch_sizes = self.batch_sizes[-1000:]

        self.mean_batch = np.mean(np.array(self.batch_sizes))

    def translation(self, x_data, x_mask, y_data, y_mask):
        enc_out = self.model.encoder(x_data, x_mask)

        trans_loss = self.model.decode_forward(enc_out, x_mask, y_data, y_mask)
        trans_loss = torch.sum(trans_loss)/self.mean_batch
        return trans_loss, enc_out

    def add_eos_padding(self, sample, B_max, max_len):
        B, T = sample.size()

        if B < B_max:
            add_batch_eos = torch.ones((B_max-B, T), dtype=torch.long).cuda() * Const.EOS
            sample = torch.cat((sample, add_batch_eos), dim=0) # B_max, T
        if T < max_len:
            add_time_eos = torch.ones((B_max, max_len-T), dtype=torch.long).cuda() * Const.EOS
            sample = torch.cat((sample, add_time_eos), dim=1) # B_max, max_len
        return sample

    def generation(self, x_data, x_mask, multi_gpu=False, B_max=0, max_len=0):
        start_time = time.time()
        enc_out = self.model.encoder(x_data, x_mask)

        gen_y_data, gen_y_mask, _, times =\
                         self.model.translate(enc_out, x_mask, 1.0,\
                                                             train=False, start_time=start_time)
        if multi_gpu == False:
            return gen_y_data.detach().cpu().numpy()[:,1:],\
                    enc_out, times
        else:
            gen_y_data = gen_y_data.detach()[:,1:]
            gen_y_data = self.add_eos_padding(gen_y_data, B_max, max_len)
            return gen_y_data, enc_out, times

    def forward(self, *inputs, **kwargs):
        mode = inputs[-1]
        if mode == 'training':
            self.model.train()

            x_data, x_mask, y_data, y_mask = inputs[0],inputs[1],inputs[2],inputs[3]

            x_data = CudaVariable(torch.LongTensor(x_data))
            x_mask = CudaVariable(torch.LongTensor(x_mask))
            y_data = CudaVariable(torch.LongTensor(y_data))
            y_mask = CudaVariable(torch.LongTensor(y_mask))

            B, _ = x_data.size()
            self.compute_mean_batch_sizes(B)

            trans_loss, enc_out = self.translation(x_data, x_mask, y_data, y_mask)

            latent_norms = self.compute_norm(enc_out, x_mask)
            latent_stds = self.compute_std(enc_out, x_mask)
            return (trans_loss) 
        elif mode == 'validation':
            self.model.eval()

            x_data, x_mask, y_data, y_mask = inputs[0],inputs[1],inputs[2],inputs[3]

            x_data = CudaVariable(torch.LongTensor(x_data))
            x_mask = CudaVariable(torch.LongTensor(x_mask))
            y_data = CudaVariable(torch.LongTensor(y_data))
            y_mask = CudaVariable(torch.LongTensor(y_mask))

            gen_y_data, enc_out, times = self.generation(x_data, x_mask)

            avg_time = times.mean()

            trans_loss = self.model.decode_forward(enc_out, x_mask, y_data, y_mask)
            trans_loss = torch.sum(trans_loss)/self.mean_batch
            return (gen_y_data, trans_loss, avg_time)
        elif mode == 'multi_validation':
            self.model.eval()

            x_data, x_mask, y_data, y_mask, B_max = inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]

            x_data = CudaVariable(torch.LongTensor(x_data))
            x_mask = CudaVariable(torch.LongTensor(x_mask))
            y_data = CudaVariable(torch.LongTensor(y_data))
            y_mask = CudaVariable(torch.LongTensor(y_mask))

            gen_y_data, enc_out, times = self.generation(x_data, x_mask, multi_gpu=True,\
                                                         B_max=B_max, max_len=self.test_max_len)

            avg_time = times.mean()

            trans_loss = self.model.decode_forward(enc_out, x_mask, y_data, y_mask)
            trans_loss = torch.sum(trans_loss)/self.mean_batch

            y_data = self.add_eos_padding(y_data, B_max, self.test_max_len)

            return (gen_y_data, trans_loss, avg_time, y_data)


