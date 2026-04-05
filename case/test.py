import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import wandb
import matplotlib.pyplot as plt
from libs.layers import CudaVariable, GaussianNoise
import nmt_const as Const

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_scale(n1, n2=None):
    if n2 == None:
        scale = math.sqrt(3)/math.sqrt(n1)
    else:
        scale = math.sqrt(6)/math.sqrt(n1+n2)
    return scale

def get_scale_para(n1, n2=None):
    scale = get_scale(n1, n2)*torch.ones(1)
    return nn.Parameter(scale.to(device))

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

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, drop_p=0.):
        super().__init__()
        self.layer1 = myLinear(dim_model, dim_ff)
        self.layer2 = myLinear(dim_ff, dim_model)
        self.dropout = nn.Dropout(p=drop_p)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        xnew = self.layer_norm(x)
        output = self.layer2(F.relu(self.layer1(xnew)))
        output = self.dropout(output)
        return output + x

def calculate_attention_entropy(attn, mask):
    """
    마스크를 고려한 어텐션 엔트로피 계산
    e = -attn * log(attn)      # BHTT 형태의 엔트로피
    e = sum(e, dim=-1)         # BHT
    e = sum(e * mask) / sum(mask)  # 마스크 적용
    """
    # 1. 각 위치의 엔트로피 계산: -attn * log(attn)
    entropy = -attn * torch.log(attn + 1e-9)  # BHTT
    
    if mask is not None:
        # 2. 마스크 확장 (B, T) -> (B, 1, T, 1)
        expanded_mask = mask.unsqueeze(1).unsqueeze(-1)
        # 3. 마스크 적용하여 합계 계산 (BHTT -> BHT)
        masked_entropy = entropy * expanded_mask
        entropy_sum = masked_entropy.sum(dim=-1)  # BHT
        # 4. 마스크 합으로 나누어 평균
        mask_sum = expanded_mask.sum(dim=-2)  # B11
        entropy = entropy_sum / (mask_sum + 1e-9)
    else:
        entropy = entropy.sum(dim=-1)  # BHT
    
    return entropy

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk, drop_p=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.temp_const = float(dk) ** 0.5
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temp_const

        if mask is not None:
            attn = attn.masked_fill(mask < 0.1, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        entropy = calculate_attention_entropy(attn, mask)
        attn = self.dropout(attn)
        
        return torch.matmul(attn, v), entropy

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

    def forward(self, q, k, v, attn_mask=None):
        qnew = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)
        
        n_head, dk, dv = self.n_head, self.dk, self.dv
        Bn, Tq, Tk, Tv = qnew.size(0), qnew.size(1), k.size(1), v.size(1)
         
        qnew = self.mat_qs(qnew).view(Bn, Tq, n_head, dk).transpose(1,2)
        k = self.mat_ks(k).view(Bn, Tk, n_head, dk).transpose(1,2)
        v = self.mat_vs(v).view(Bn, Tv, n_head, dv).transpose(1,2)
         
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            
        output, entropy = self.sdp_attn(qnew, k, v, mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(Bn, Tq, -1)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output + q, entropy

class TM_EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ff, n_head, dk, dv, drop_p=0.):
        super(TM_EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p)
        self.ff_layer = FeedForward(dim_model, dim_ff, drop_p=drop_p)

    def forward(self, enc_in, x_mask=None):
        enc_out, entropy = self.self_attn(enc_in, enc_in, enc_in, attn_mask=x_mask)
        enc_out = self.ff_layer(enc_out)
        return enc_out, entropy

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

    def forward(self, src_seq, src_mask):
        src_mask = src_mask.unsqueeze(1)
        enc_out = self.src_emb(src_seq)
        enc_out = self.pos_enc(enc_out)
        enc_out = self.emb_dropout(enc_out)
        
        layer_entropies = []
        for enc_layer in self.layer_stack:
            enc_out, entropy = enc_layer(enc_out, x_mask=src_mask)
            layer_entropies.append(entropy)
            
        return enc_out, layer_entropies

class TM_DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ff, n_head, dk, dv, drop_p=0.):
        super(TM_DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p)
        self.enc_attn = MultiHeadAttention(n_head, dim_model, dk, dv, drop_p=drop_p)
        self.ff_layer = FeedForward(dim_model, dim_ff, drop_p=drop_p)

    def forward(self, dec_in, enc_out, y_mask=None, enc_mask=None):
        dec_out, self_entropy = self.self_attn(dec_in, dec_in, dec_in, attn_mask=y_mask)
        dec_out, cross_entropy = self.enc_attn(dec_out, enc_out, enc_out, attn_mask=enc_mask)
        dec_out = self.ff_layer(dec_out)
        return dec_out, self_entropy, cross_entropy

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

    def get_subsequent_mask(self, seq):
        s0, s1 = seq.size()
        mask = torch.tril(torch.ones((s1, s1), device=seq.device), diagonal=0)
        return mask.type(torch.cuda.FloatTensor).unsqueeze(0)

    def forward(self, trg_seq, trg_mask, enc_out, src_mask):
        src_mask = src_mask.unsqueeze(1)
        trg_mask = trg_mask.unsqueeze(1)
        dec_out = self.trg_emb(trg_seq)
        dec_out = self.pos_enc(dec_out)
        dec_out = self.emb_dropout(dec_out)

        mh_attn_sub_mask = self.get_subsequent_mask(trg_seq)
        y_mask = trg_mask * mh_attn_sub_mask

        layer_self_entropies = []
        layer_cross_entropies = []
        
        for dec_layer in self.layer_stack:
            dec_out, self_entropy, cross_entropy = dec_layer(
                dec_out, enc_out, y_mask=y_mask, enc_mask=src_mask)
            layer_self_entropies.append(self_entropy)
            layer_cross_entropies.append(cross_entropy)

        dec_out = self.add_noise(dec_out)
        dec_out = self.trg_word_proj(dec_out)
        return dec_out, layer_self_entropies, layer_cross_entropies

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
        
        # 레이어별 엔트로피 저장용 버퍼
        self.entropy_buffer = {
            'encoder_layers': [[] for _ in range(args.n_layers)],
            'decoder_self_layers': [[] for _ in range(args.n_layers)],
            'decoder_cross_layers': [[] for _ in range(args.n_layers)],
            'steps': []
        }

    def forward(self, x_data, x_mask, y_data, y_mask, step=None):
        x_data = CudaVariable(torch.LongTensor(x_data)).transpose(0, 1)
        x_mask = CudaVariable(torch.FloatTensor(x_mask)).transpose(0, 1)
        y_data = CudaVariable(torch.LongTensor(y_data)).transpose(0, 1)
        y_mask = CudaVariable(torch.FloatTensor(y_mask)).transpose(0, 1)
        
        y_target = y_data[:, 1:]
        y_mask = y_mask[:, 1:]
        y_in = y_data[:, :-1]
        Bn, Ty = y_in.size()

        # 인코더/디코더 실행 및 엔트로피 수집
        enc_out, enc_entropies = self.encoder(x_data, x_mask)
        dec_out, dec_self_entropies, dec_cross_entropies = self.decoder(
            y_in, y_mask, enc_out, x_mask)

        # 레이어별 엔트로피 저장
        if step is not None:
            self.entropy_buffer['steps'].append(step)
            for i, ent in enumerate(enc_entropies):
                self.entropy_buffer['encoder_layers'][i].append(ent.mean().item())
            for i, ent in enumerate(dec_self_entropies):
                self.entropy_buffer['decoder_self_layers'][i].append(ent.mean().item())
            for i, ent in enumerate(dec_cross_entropies):
                self.entropy_buffer['decoder_cross_layers'][i].append(ent.mean().item())
        
        # Loss 계산
        loss = self.criterion(dec_out.view(-1, dec_out.size(2)), y_target.contiguous().view(-1))
        loss = torch.sum(loss * y_mask.contiguous().view(-1)) / Bn

        return loss


    def log_entropies(self, step):
        """WandB에 레이어별 엔트로피 로깅"""
        if step % 5000 == 0 and len(self.entropy_buffer['steps']) > 0:
            # 각 레이어별 평균 엔트로피 계산 및 로깅
            for i in range(len(self.encoder.layer_stack)):
                wandb.log({
                    f'encoder_layer_{i+1}_entropy': np.mean(self.entropy_buffer['encoder_layers'][i]),
                    f'decoder_self_layer_{i+1}_entropy': np.mean(self.entropy_buffer['decoder_self_layers'][i]),
                    f'decoder_cross_layer_{i+1}_entropy': np.mean(self.entropy_buffer['decoder_cross_layers'][i]),
                    'step': step
                })
            
            # 버퍼 초기화
            self.entropy_buffer = {
                'encoder_layers': [[] for _ in range(len(self.encoder.layer_stack))],
                'decoder_self_layers': [[] for _ in range(len(self.decoder.layer_stack))],
                'decoder_cross_layers': [[] for _ in range(len(self.decoder.layer_stack))],
                'steps': []
            }

    def plot_entropy_analysis(self, save_path='entropy_analysis.png'):
        """레이어별 엔트로피 분석 그래프 생성"""
        plt.figure(figsize=(15, 12))
        
        # 1. 인코더 레이어별 엔트로피
        plt.subplot(3, 1, 1)
        for i in range(len(self.encoder.layer_stack)):
            plt.plot(self.entropy_buffer['steps'], 
                    self.entropy_buffer['encoder_layers'][i],
                    label=f'Layer {i+1}')
        plt.title('Encoder Self-Attention Entropy by Layer')
        plt.xlabel('Steps')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True)

        # 2. 디코더 self-attention 레이어별 엔트로피
        plt.subplot(3, 1, 2)
        for i in range(len(self.decoder.layer_stack)):
            plt.plot(self.entropy_buffer['steps'],
                    self.entropy_buffer['decoder_self_layers'][i],
                    label=f'Layer {i+1}')
        plt.title('Decoder Self-Attention Entropy by Layer')
        plt.xlabel('Steps')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True)

        # 3. 크로스 어텐션 레이어별 엔트로피
        plt.subplot(3, 1, 3)
        for i in range(len(self.decoder.layer_stack)):
            plt.plot(self.entropy_buffer['steps'],
                    self.entropy_buffer['decoder_cross_layers'][i],
                    label=f'Layer {i+1}')
        plt.title('Cross-Attention Entropy by Layer')
        plt.xlabel('Steps')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def calculate_layer_wise_entropy_stats(self):
        """레이어별 엔트로피 통계 계산"""
        stats = {
            'encoder': [],
            'decoder_self': [],
            'decoder_cross': []
        }
        
        # 각 레이어별 평균 엔트로피 계산
        for i in range(len(self.encoder.layer_stack)):
            stats['encoder'].append(np.mean(self.entropy_buffer['encoder_layers'][i]))
            stats['decoder_self'].append(np.mean(self.entropy_buffer['decoder_self_layers'][i]))
            stats['decoder_cross'].append(np.mean(self.entropy_buffer['decoder_cross_layers'][i]))
            
        return stats