# H. Choi, hchoi@handong.edu

import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import eye_, xavier_uniform_, xavier_normal_, constant_

from libs.layers import CudaVariable, GaussianNoise
import nmt_const as Const

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")
print ('Is there any GPU ', use_cuda)


def get_scale(n1, n2=None): # for Xavier uniform initialization
    if n2 == None: # with only one side.
        scale = math.sqrt(3)/math.sqrt(n1)
    else:
        scale = math.sqrt(6)/math.sqrt(n1+n2)
    return scale # * 0.7 # for layer_norm(out + x)


class myEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(myEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def reset_parameters(self):
        scale = get_scale(self.embedding_dim)
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
        #print('sizes', qnew.size(), k.size(), v.size())
        if attn_mask is not None: #  B ? T -> B ? ? T 
            attn_mask = attn_mask.unsqueeze(1)
        
        output = self.sdp_attn(qnew, k, v, mask=attn_mask) # Bn H Ty E
        output = output.transpose(1, 2).contiguous().view(Bn, Tq, -1) # Bn Ty H*E
        output = self.dropout(self.out_proj(output))

        return self.layer_norm(output + q)

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
        return self.layer_norm(output + x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = 200
        self.get_pe(self.max_len)

    def forward(self, x):
        #self.max_len = 200 # don't need! but necessary for compatability with the previous model. 
        if self.training or x.size(1) <= self.max_len:
            pass
        else:
            #self.d_model = 512 # don't need! but 
            self.get_pe(x.size(1))
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

    def get_pe(self, max_len):
        scale = get_scale(self.d_model)
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
                    dim_wemb=512, dim_model=512, dim_ff=1024, drop_p=0., emb_noise=0.):
        super(TM_Encoder, self).__init__()

        # first in : 
        self.src_emb = myEmbedding(src_words_n, dim_wemb)#, padding_idx=Const.PAD)
        self.pos_enc = PositionalEncoding(dim_wemb)
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
            dim_wemb=512, dim_model=512, dim_ff=1024, drop_p=0., emb_noise=0.):
        super(TM_Decoder, self).__init__()

        # first in 
        self.dec_emb = myEmbedding(trg_words_n, dim_wemb)#, padding_idx=Const.PAD)
        self.pos_enc = PositionalEncoding(dim_wemb)
        self.layer_norm = nn.LayerNorm(dim_wemb)
        # repeated layer
        self.layer_stack = nn.ModuleList([
            TM_DecoderLayer(dim_model, dim_ff, n_head, dk, dv, drop_p=drop_p)
            for _ in range(n_layers)])
        # readout 
        self.trg_word_proj = myLinear(dim_model, trg_words_n)
        #self.trg_word_proj.weight = self.dec_emb.weight # Share the weight 

    def get_subsequent_mask(self, seq):
        s0, s1 = seq.size()
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
        dec_out = self.trg_word_proj(dec_out)   

        return dec_out # B Ty E

class Transformer(nn.Module):
    def __init__(self, args=None):
        super(Transformer, self).__init__()
      
        src_words_n, trg_words_n = args.src_words_n, args.trg_words_n
        dim_wemb, dim_model = args.dim_model, args.dim_model
        drop_p = args.dropout_p
        dim_ff, n_layers = args.tm_dim_ff, args.tm_n_layers
        n_head, dk, dv = args.tm_n_head, args.tm_dk, args.tm_dv
        assert dim_model == dim_wemb, 'dim_model == dim_wemb for residual connections'

        self.encoder=TM_Encoder(src_words_n, n_layers=n_layers, n_head=n_head,
                dim_wemb=dim_model, dim_model=dim_model, dim_ff=dim_ff, drop_p=drop_p, 
                emb_noise=args.emb_noise)
        self.decoder=TM_Decoder(trg_words_n, n_layers=n_layers, n_head=n_head,
                dim_wemb=dim_model, dim_model=dim_model, dim_ff=dim_ff, drop_p=drop_p, 
                emb_noise=args.emb_noise)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x_data, x_mask, y_data, y_mask):
        x_data = CudaVariable(torch.LongTensor(x_data)).transpose(0,1) # B T
        x_mask = CudaVariable(torch.FloatTensor(x_mask)).transpose(0,1) # B T
        y_data = CudaVariable(torch.LongTensor(y_data)).transpose(0,1) # B T
        y_mask = CudaVariable(torch.FloatTensor(y_mask)).transpose(0,1) # B T

        y_target = y_data[:,1:] # label
        y_mask = y_mask[:,1:]
        y_in = y_data[:,:-1] # input as teacher forcing
        Bn, Ty = y_in.size()

        # encode and decode
        enc_out = self.encoder(x_data, x_mask) # B Tx E
        out = self.decoder(y_in, y_mask, enc_out, x_mask) # B Ty E(num of words)

        # loss
        loss = self.criterion(out.view(-1, out.size(2)), y_target.contiguous().view(-1)) 
        loss = torch.sum(loss * y_mask.contiguous().view(-1))/Bn 

        return loss

    def forward2(self, x_data, x_mask, y_data, y_mask):
        x_data = CudaVariable(torch.LongTensor(x_data)).transpose(0,1) # B T
        x_mask = CudaVariable(torch.FloatTensor(x_mask)).transpose(0,1) # B T
        y_data = CudaVariable(torch.LongTensor(y_data)).transpose(0,1) # B T
        y_mask = CudaVariable(torch.FloatTensor(y_mask)).transpose(0,1) # B T

        y_target = y_data[:,1:] # label
        y_mask = y_mask[:,1:]
        y_in = y_data[:,:-1] # input as teacher forcing
        Bn, Ty = y_in.size()

        # encode and decode
        enc_out = self.encoder(x_data, x_mask) # B Tx E
        out = self.decoder(y_in, y_mask, enc_out, x_mask) # B Ty E(num of words)

        # loss
        loss = self.criterion(out.view(-1, out.size(2)), y_target.contiguous().view(-1)) 
        loss = loss * y_mask.contiguous().view(-1) 
        loss = torch.sum(loss.view(Bn, Ty), dim=1)/torch.sum(y_target, dim=1)

        return loss
