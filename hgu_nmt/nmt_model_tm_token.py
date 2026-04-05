import numpy as np
import os
import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_ 

from libs.layers import CudaVariable, GaussianNoise
import nmt_const as Const

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")


def get_scale(n1, n2=None): # for Xavier uniform initialization
    if n2 == None: # with only one side.
        scale = math.sqrt(3)/math.sqrt(n1)
    else:
        scale = math.sqrt(6)/math.sqrt(n1+n2)
    return scale # * 0.7 # for layer_norm(out + x)

def get_scale_para(n1, n2=None): # for Xavier uniform initialization
    scale = get_scale(n1, n2)*torch.ones(1)
    return nn.Parameter(scale.to(device)) # for layer_norm(out + x)

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
        if self.in_features == self.out_features:
            self.weight.data.copy_(torch.eye(self.in_features))
        else:
            scale = get_scale(self.in_features, self.out_features)
            self.weight.data.uniform_(-scale, scale)
            
        if self.bias is not None:
            self.bias.data.zero_()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk, drop_p=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.temp_const = float(dk) # ** 0.5 # TEST it was better without 0.5 (layer_norm, init) # 논문과 확인
        self.temper = nn.Parameter(torch.ones(1)) # dkParam option
        #self.temper = nn.Parameter(torch.ones(1), requires_grad=True)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, q, k, v, mask=None): # B H T E

        #attn = torch.matmul(q, k.transpose(-2, -1)) / (self.temp_const) # B H Tq Tk
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.temper * self.temp_const) # B H Tq Tk
        #print("temper shape: ", self.temper.shape)
        if mask is not None:
            attn = attn.masked_fill(mask < 0.1, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        
        attn = self.dropout(attn) # B H Tq Tk 
        output = torch.matmul(attn, v) # B H Tq E 
        
        return output 

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, dim_model, dk, dv, drop_p=0.):
        super(MultiHeadAttention, self).__init__()
        
        self.n_head, self.dk, self.dv = n_head, dk, dv
        
        self.mat_qs = myLinear(dim_model, n_head * dk, bias=False)
        self.mat_ks = myLinear(dim_model, n_head * dk, bias=False)
        self.mat_vs = myLinear(dim_model, n_head * dv, bias=False)
        
        self.sdp_attn = ScaledDotProductAttention(dk, drop_p=drop_p)
        
        self.out_proj = myLinear(n_head * dv, dim_model)
        self.dropout = nn.Dropout(p=drop_p)
        self.layer_norm = nn.LayerNorm(dim_model)

        # 로깅 설정
        self.collect_res_ratio = False
        # (layer_id, n_tokens_used, sent_mean_ratio)
        self.res_ratio_log = []
        self.layer_id = "UNK"

    def forward(self, q, k, v, attn_mask=None, token_mask=None):
        n_head, dk, dv = self.n_head, self.dk, self.dv
        Bn, Tq, Tk, Tv = q.size(0), q.size(1), k.size(1), v.size(1)

        qnew = self.mat_qs(q).view(Bn, Tq, n_head, dk).transpose(1, 2)  # (B,H,Tq,dk)
        k = self.mat_ks(k).view(Bn, Tk, n_head, dk).transpose(1, 2)     # (B,H,Tk,dk)
        v = self.mat_vs(v).view(Bn, Tv, n_head, dv).transpose(1, 2)     # (B,H,Tv,dv)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # (B,1,Tq,Tk) 형태 맞추기

        output = self.sdp_attn(qnew, k, v, mask=attn_mask)              # (B,H,Tq,dv)
        output = output.transpose(1, 2).contiguous().view(Bn, Tq, -1)   # (B,Tq,H*dv)
        output = self.out_proj(output)                                  # (B,Tq,dim_model)
        output = self.dropout(output)

        # 더해질 값(branch)과 잔차(residual)
        branch = self.layer_norm(output)  # 더해질 값
        residual = q                      # 잔차 경로

        # 토큰별 ratio -> 문장별 평균 ratio 로깅
        if self.collect_res_ratio:
            eps = 1e-12

            # 토큰별 norm
            res_norm = residual.norm(dim=-1)  # (B,T)
            br_norm  = branch.norm(dim=-1)    # (B,T)

            # 토큰별 ratio
            ratio_tok = br_norm / (res_norm + eps)  # (B,T)

            if token_mask is None:
                # 문장별 평균 ratio
                sent_ratio = ratio_tok.mean(dim=1)  # (B,)
                sent_n = torch.full(
                    (ratio_tok.size(0),),
                    ratio_tok.size(1),
                    device=ratio_tok.device
                )
            else:
                m = token_mask.float()               # (B,T) 0/1
                denom = m.sum(dim=1).clamp_min(1.0)  # (B,)
                sent_ratio = (ratio_tok * m).sum(dim=1) / denom
                sent_n = denom

            # 배치의 각 문장마다 1개 값 저장
            sent_ratio_list = sent_ratio.detach().cpu().tolist()
            sent_n_list = sent_n.detach().cpu().tolist()
            for r, n_used in zip(sent_ratio_list, sent_n_list):
                self.res_ratio_log.append((self.layer_id, int(n_used), float(r)))

        return branch + residual


class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, drop_p=0.):
        super().__init__()
        self.layer1 = myLinear(dim_model, dim_ff)
        self.layer2 = myLinear(dim_ff, dim_model)
        self.dropout = nn.Dropout(p=drop_p)
        self.layer_norm = nn.LayerNorm(dim_model)

        # 로깅 설정
        self.collect_res_ratio = False
        # (layer_id, n_tokens_used, sent_mean_ratio)
        self.res_ratio_log = []
        self.layer_id = "UNK"

    def forward(self, x, token_mask=None):
        output = self.layer2(F.relu(self.layer1(x)))
        output = self.dropout(output)

        branch = self.layer_norm(output)
        residual = x

        # 토큰별 ratio -> 문장별 평균 ratio 로깅
        if self.collect_res_ratio:
            eps = 1e-12

            res_norm = residual.norm(dim=-1)  # (B,T)
            br_norm  = branch.norm(dim=-1)    # (B,T)
            ratio_tok = br_norm / (res_norm + eps)

            if token_mask is None:
                sent_ratio = ratio_tok.mean(dim=1)  # (B,)
                sent_n = torch.full(
                    (ratio_tok.size(0),),
                    ratio_tok.size(1),
                    device=ratio_tok.device
                )
            else:
                m = token_mask.float()
                denom = m.sum(dim=1).clamp_min(1.0)  # (B,)
                sent_ratio = (ratio_tok * m).sum(dim=1) / denom
                sent_n = denom

            sent_ratio_list = sent_ratio.detach().cpu().tolist()
            sent_n_list = sent_n.detach().cpu().tolist()
            for r, n_used in zip(sent_ratio_list, sent_n_list):
                self.res_ratio_log.append((self.layer_id, int(n_used), float(r)))

        return branch + residual


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.scale = get_scale_para(self.d_model)

    def forward(self, x):
        return x + self.get_pe(x.size(1))

    def get_pe(self, x_len):
        pe = torch.zeros(x_len, self.d_model).to(device)
        position = torch.arange(0, x_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) * self.scale

class TM_EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ff, n_head, dk, dv, drop_p=0.):
        super(TM_EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p)
        self.ff_layer = FeedForward(dim_model, dim_ff, drop_p=drop_p)

    def forward(self, enc_in, x_mask=None, token_mask=None):
        enc_out = self.self_attn(enc_in, enc_in, enc_in, attn_mask=x_mask, token_mask=token_mask)
        enc_out = self.ff_layer(enc_out, token_mask=token_mask)
        return enc_out

class TM_Encoder(nn.Module):
    def __init__(self, src_words_n, n_layers=6, n_head=8, dk=64, dv=64,
                 dim_wemb=512, dim_model=512, dim_ff=1024, drop_p=0., emb_noise=0.):
        super(TM_Encoder, self).__init__()
        self.src_emb = myEmbedding(src_words_n, dim_wemb)
        self.pos_enc = PositionalEncoding(dim_wemb)
        self.emb_dropout = nn.Dropout(p=drop_p)
        self.layer_stack = nn.ModuleList([
            TM_EncoderLayer(dim_model, dim_ff, n_head, dk, dv, drop_p=drop_p)
            for _ in range(n_layers)])
        
        for i, layer in enumerate(self.layer_stack):
            layer.self_attn.layer_id = f"Enc_SelfAttn_L{i}"
            layer.ff_layer.layer_id = f"Enc_FFN_L{i}"
        

    def forward(self, src_seq, src_mask):
        token_mask = src_mask
        src_mask = src_mask.unsqueeze(1)

        enc_out = self.src_emb(src_seq)
        enc_out = self.pos_enc(enc_out)
        enc_out = self.emb_dropout(enc_out)

        for enc_layer in self.layer_stack:
            enc_out = enc_layer(enc_out, x_mask=src_mask, token_mask=token_mask)
           
        return enc_out
        #return enc_out, total_entropy,layer_entropies

class TM_DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ff, n_head, dk, dv, drop_p=0.):
        super(TM_DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p)
        self.enc_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p)
        self.ff_layer = FeedForward(dim_model, dim_ff, drop_p=drop_p)


    def forward(self, dec_in, enc_out, y_mask=None, enc_mask=None, token_mask=None):
        dec_out = self.self_attn(dec_in, dec_in, dec_in, attn_mask=y_mask, token_mask=token_mask)
       
        dec_out = self.enc_attn(dec_out, enc_out, enc_out, attn_mask=enc_mask, token_mask=token_mask)
       
        dec_out = self.ff_layer(dec_out, token_mask=token_mask)
       
        return dec_out

class TM_Decoder(nn.Module):
    def __init__(self, trg_words_n, n_layers=6, n_head=8, dk=64, dv=64,
                 dim_wemb=512, dim_model=512, dim_ff=1024, drop_p=0., emb_noise=0.):
        super(TM_Decoder, self).__init__()
        self.trg_emb = myEmbedding(trg_words_n, dim_wemb)
        self.pos_enc = PositionalEncoding(dim_wemb)
        self.emb_dropout = nn.Dropout(p=drop_p)
        self.layer_stack = nn.ModuleList([
            TM_DecoderLayer(dim_model, dim_ff, n_head, dk, dv, drop_p=drop_p)
            for _ in range(n_layers)])
        self.add_noise = GaussianNoise(mean=0.0, sigma=emb_noise)
        self.trg_word_proj = myLinear(dim_model, trg_words_n)

        for i, layer in enumerate(self.layer_stack):
            layer.self_attn.layer_id = f"Dec_SelfAttn_L{i}"
            layer.enc_attn.layer_id = f"Dec_CrossAttn_L{i}"
            layer.ff_layer.layer_id = f"Dec_FFN_L{i}"

    def get_subsequent_mask(self, seq):
        s0, s1 = seq.size()
        mask = torch.tril(torch.ones((s1, s1), device=seq.device), diagonal=0)
        return mask.type(torch.cuda.FloatTensor).unsqueeze(0)

    def forward(self, trg_seq, trg_mask, enc_out, src_mask):
        src_mask = src_mask.unsqueeze(1)
        token_mask = trg_mask

        trg_mask = trg_mask.unsqueeze(1)
        dec_out = self.trg_emb(trg_seq)
        dec_out = self.pos_enc(dec_out)
        dec_out = self.emb_dropout(dec_out)

        mh_attn_sub_mask = self.get_subsequent_mask(trg_seq)
        y_mask = trg_mask * mh_attn_sub_mask

        
        for dec_layer in self.layer_stack:
            dec_out = dec_layer(dec_out, enc_out, y_mask=y_mask, enc_mask=src_mask, token_mask=token_mask)

        dec_out = self.add_noise(dec_out)
        dec_out = self.trg_word_proj(dec_out)
        return dec_out
        # return dec_out, total_self_entropy, total_cross_entropy, self_layer_entropies, cross_layer_entropies

class Transformer(nn.Module):
    def __init__(self, src_words_n, trg_words_n, args=None):
        super(Transformer, self).__init__()
        assert args.dim_model == args.dim_wemb, 'dim_model == dim_wemb for residual connections'

        self.encoder = TM_Encoder(src_words_n, n_layers=args.n_layers, n_head=args.n_head,
                                  dim_wemb=args.dim_wemb, dim_model=args.dim_model, dim_ff=args.dim_ff,
                                  drop_p=args.drop_p, emb_noise=args.emb_noise, dk=args.dk, dv=args.dv)
        self.decoder = TM_Decoder(trg_words_n, n_layers=args.n_layers, n_head=args.n_head,
                                  dim_wemb=args.dim_wemb, dim_model=args.dim_model, dim_ff=args.dim_ff,
                                  drop_p=args.drop_p, emb_noise=args.emb_noise, dk=args.dk, dv=args.dv)
        self.criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smooth)
        

    def forward(self, x_data, x_mask, y_data, y_mask):
        x_data = CudaVariable(torch.LongTensor(x_data)).transpose(0, 1)
        x_mask = CudaVariable(torch.FloatTensor(x_mask)).transpose(0, 1)
        y_data = CudaVariable(torch.LongTensor(y_data)).transpose(0, 1)
        y_mask = CudaVariable(torch.FloatTensor(y_mask)).transpose(0, 1)

        #x_data = x_data[:,1:] # remove BOS
        #x_mask = x_mask[:,1:] # remove BOS
        
        y_target = y_data[:, 1:] # label # remove BOS
        y_mask = y_mask[:, 1:]  # remove BOS
        y_in = y_data[:, :-1]   # input as teacher forcing
        Bn, Ty = y_in.size()

        #encode and decode
        # enc_out,enc_entropy, enc_layer_entropies = self.encoder(x_data, x_mask) # B Tx E
        enc_out= self.encoder(x_data, x_mask)
        # dec_out, dec_self_entropy, dec_cross_entropy, dec_self_layer_entropies, dec_cross_layer_entropies = self.decoder(y_in, y_mask, enc_out, x_mask) # B Ty E
        dec_out= self.decoder(y_in, y_mask, enc_out, x_mask)

        #loss
        loss = self.criterion(dec_out.view(-1, dec_out.size(2)), y_target.contiguous().view(-1))
        loss = torch.sum(loss * y_mask.contiguous().view(-1)) / Bn
        
        #print(f"Encoder entropy: {enc_entropy}")  # Transformer.forward() 내부에 추가
        #print(f"Decoder self entropy: {dec_self_entropy}")
        #print(f"Decoder cross entropy: {dec_cross_entropy}")
        
        # return loss, enc_entropy, dec_self_entropy, dec_cross_entropy, enc_layer_entropies, dec_self_layer_entropies, dec_cross_layer_entropies
        return loss

