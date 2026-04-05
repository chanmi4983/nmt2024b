import torch 
import math
from torch.nn.parameter import Parameter
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def CudaVariable(X):
    return Variable(X).to(device)

def CudaVariableNoGrad(X):
    return Variable(X, requires_grad=False).to(device)

def get_scale(nin, nout):
    return  math.sqrt(6)/math.sqrt(nin+nout) # Xavier

class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, sigma=1.0):
        super().__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0.0:
            noise = torch.randn_like(x, device=x.device, requires_grad=False)
            return x + (noise+self.mean)*self.sigma
        else:
            return x 

class PositionalEncoding(nn.Module):
    def __init__(self, words_n, dim_wemb, n_pos=200):
        super(PositionalEncoding, self).__init__()

        self.n, self.dim = words_n, dim_wemb
        self.register_buffer('pos_table', self.get_pos_enc_table(n_pos, dim_wemb))

    def get_pos_enc_table(self, n_pos, dim):
        def get_pos_angle_vec(pos):
            return [pos / np.power(10000, 2*(hj//2)/dim) for hj in range(dim)]

        pos_enc_table = np.array([get_pos_angle_vec(pos_i) for pos_i in range(n_pos)])
        pos_enc_table[:, 0::2] = np.sin(pos_enc_table[:, 0::2])  # dim 2i
        pos_enc_table[:, 1::2] = np.cos(pos_enc_table[:, 1::2])  # dim 2i+1
        
        # added by hchoi
        pos_enc_table = pos_enc_table * np.sqrt(6)/np.sqrt(self.n + self.dim) # Xavier

        return torch.FloatTensor(pos_enc_table).unsqueeze(0)

    def forward(self, x, index=-1):
        if index >= 0:
            out = x + self.pos_table[:, index] # for the rnn decoding step
        else:
            out = x + self.pos_table[:, :x.size(1)]#.clone().detach()
        return out

class PositionalEncodingNew(nn.Module): # from pytorch
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncodingNew, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # T E 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe) # torch.Size([1, 200, 300]) without transpose above

    def forward(self, x):
        #x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class myEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(myEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def reset_parameters(self):
        #scale = get_scale(self.num_embeddings, self.embedding_dim)
        scale = math.sqrt(6)/math.sqrt(1+self.embedding_dim)
        self.weight.data.uniform_(-scale, scale)

class myLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(myLinear, self).__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        #if self.in_features == self.out_features:
        #    self.weight.data.copy_(torch.eye(self.in_features))
        #else:
        scale = get_scale(self.in_features, self.out_features)
        self.weight.data.uniform_(-scale, scale)

        if self.bias is not None:
            self.bias.data.zero_()

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-7):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid))
        self.b_2 = nn.Parameter(torch.zeros(d_hid))

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


# bidirectional LSTM using two nn.LSTMCell
class biLSTM(nn.Module): 
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(biLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.step_f = nn.LSTMCell(input_size, hidden_size)
        self.step_r = nn.LSTMCell(input_size, hidden_size)
    
    def forward(self, x_data, x_mask=None):
        if self.batch_first:
            x_data = x_data.transpose(0, 1)

        ht_f, ct_f, ht_r, ct_r = self.init_hidden(x_data.size(1))

        step_f = range(0, x_data.size(0)) # 'forward'
        step_r = range(x_data.size(0)-1,-1,-1) # 'backward' or 'reverse'

        output_f = []
        output_r = []
        for (i,j) in zip(step_f, step_r):
            ht_f_tmp, ct_f_tmp = self.step_f(x_data[i], (ht_f, ct_f))
            ht_r_tmp, ct_r_tmp = self.step_r(x_data[j], (ht_r, ct_r))
            if x_mask is not None:
                x_m = x_mask[i]
                ht_f = ht_f_tmp*x_m.unsqueeze(1) + ht_f*((1.-x_m).unsqueeze(1))
                ct_f = ct_f_tmp*x_m.unsqueeze(1) + ct_f*((1.-x_m).unsqueeze(1))
                x_m = x_mask[j]
                ht_r = ht_r_tmp*x_m.unsqueeze(1) + ht_r*((1.-x_m).unsqueeze(1))
                ct_r = ct_r_tmp*x_m.unsqueeze(1) + ct_r*((1.-x_m).unsqueeze(1))
            else:
                ht_f, ct_f, ht_r, ct_r = ht_f_tmp, ct_f_tmp, ht_r_tmp, ct_r_tmp

            output_f.append(ht_f)
            output_r.insert(0, ht_r)

        output_f = torch.stack(output_f)
        output_r = torch.stack(output_r)
        output = torch.cat((output_f, output_r), dim=2)

        if self.batch_first:
            output = output.transpose(0, 1)
        return output
        
    def init_hidden(self, Bn):
        h_f = CudaVariable(torch.zeros(Bn, self.hidden_size))
        c_f = CudaVariable(torch.zeros(Bn, self.hidden_size))
        h_r = CudaVariable(torch.zeros(Bn, self.hidden_size))
        c_r = CudaVariable(torch.zeros(Bn, self.hidden_size))
        return h_f, c_f, h_r, c_r


# naive LSTM from scratch
class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, direction='f', batch_first=False, plus=False):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.direction = direction
        self.plus = plus

        self.input_weights = myLinear(input_size, 4*hidden_size, bias=True)
        self.hidden_weights = myLinear(hidden_size, 4*hidden_size, bias=False)
        if self.plus: 
            self.plus_layer = myLinear(hidden_size, hidden_size, bias=True)

    def get_gates_cell(self, gates, xt):
        ig, fg, og, ct = gates.chunk(4, 1)
        return ig, fg, og, ct
        
    def step(self, xt, htm, ctm, x_m=None):
        gates = self.input_weights(xt) + self.hidden_weights(htm)
        ig, fg, og, ct = self.get_gates_cell(gates, xt)

        ig = torch.sigmoid(ig)
        fg = torch.sigmoid(fg)
        og = torch.sigmoid(og)
        ct = torch.tanh(ct)  # o_t

        ct = fg*ctm + ig*ct
        ht = og*torch.tanh(ct)

        if self.plus: 
            ht = torch.tanh(self.plus_layer(ht))

        if x_m is not None:
            ct = ct*x_m.unsqueeze(1) + ctm*((1.-x_m).unsqueeze(1))
            ht = ht*x_m.unsqueeze(1) + htm*((1.-x_m).unsqueeze(1))

        return ht, ct

    def forward(self, x_data, x_mask=None, hidden=None):
        if self.batch_first:
            x_data = x_data.transpose(0, 1)

        if hidden is None:
            ht, ct = self.init_hidden(x_data.size(1))
        else:
            ht, ct = hidden
        
        if self.direction == 'f':
            step_range = range(0, x_data.size(0)) # 'forward'
        else:
            step_range = range(x_data.size(0)-1,-1,-1) # 'backward' or 'reverse'

        output = []
        for i in step_range:
            if x_mask is None:
                ht, ct = self.step(x_data[i], ht, ct)
            else:
                ht, ct = self.step(x_data[i], ht, ct, x_m=x_mask[i])
            if self.direction == 'f':
                output.append(ht)
            else:
                output.insert(0, ht)
        output = torch.stack(output) # list 2 tensor

        if self.batch_first:
            output = output.transpose(0, 1)

        return output

    def init_hidden(self, Bn):
        h0 = CudaVariable(torch.zeros(Bn, self.hidden_size))
        c0 = CudaVariable(torch.zeros(Bn, self.hidden_size))
        return h0, c0

class myLSTMCell(myLSTM):
    def __init__(self, input_size, hidden_size, direction='f', batch_first=False, plus=False):
        super(myLSTMCell, self).__init__(input_size, hidden_size, direction, batch_first, plus)

    def forward(self, xt, tms, x_m=None):
        return self.step(xt, tms[0], tms[1], x_m)

class myLSTMg(myLSTM):
    def __init__(self, input_size, hidden_size, direction='f', batch_first=False, plus=False):
        super(myLSTMg, self).__init__(input_size, hidden_size, direction, batch_first, plus)

        self.input_weights = myLinear(input_size, 3, bias=True)
        self.hidden_weights = myLinear(hidden_size, 3, bias=False)
        self.cell_weight = myLinear(input_size, hidden_size, bias=True)

    def get_gates_cell(self, gates, xt):
        ig, fg, og = gates.chunk(3, 1)
        ct = self.cell_weight(xt) 
        return ig, fg, og, ct

class myLSTMgCell(myLSTMg):
    def __init__(self, input_size, hidden_size, direction='f', batch_first=False):
        super(myLSTMgCell, self).__init__(input_size, hidden_size, direction, batch_first)

    def forward(self, xt, tms, x_m=None):
        return self.step(xt, tms[0], tms[1], x_m)

# persistant recurrent unit
class PRU(nn.Module):
    def __init__(self, input_size, hidden_size, direction='f', batch_first=False, plus=False):
        super(PRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.direction = direction
        self.plus = plus

        self.input_weights = myLinear(input_size, 3*hidden_size, bias=True)
        self.hidden_weights = myLinear(hidden_size, 3*hidden_size, bias=False)
        self.cell_weight = myLinear(input_size, hidden_size, bias=True)
        if self.plus: 
            self.plus_layer = myLinear(hidden_size, hidden_size, bias=True)
            
    def step(self, xt, htm, ctm, x_m=None):
        gates = self.input_weights(xt) + self.hidden_weights(htm)
        ig, fg, og = gates.chunk(3, 1)
        ct = self.cell_weight(xt) 

        ig = torch.sigmoid(ig)
        fg = torch.sigmoid(fg)
        og = torch.sigmoid(og)
        ct = torch.tanh(ct)  # o_t

        ct = fg*ctm + ig*ct
        ht = og*torch.tanh(ct)
        
        if self.plus: 
            ht = torch.tanh(self.plus_layer(ht))

        if x_m is not None:
            ct = ct*x_m.unsqueeze(1) + ctm*((1.-x_m).unsqueeze(1))
            ht = ht*x_m.unsqueeze(1) + htm*((1.-x_m).unsqueeze(1))

        return ht, ct

    def forward(self, x_data, x_mask=None, hidden=None):
        if self.batch_first:
            x_data = x_data.transpose(0, 1)

        if hidden is None:
            ht, ct = self.init_hidden(x_data.size(1))
        else:
            ht, ct = hidden
        
        if self.direction == 'f':
            step_range = range(0, x_data.size(0)) # 'forward'
        else:
            step_range = range(x_data.size(0)-1,-1,-1) # 'backward' or 'reverse'

        output = []
        for i in step_range:
            if x_mask is None:
                ht, ct = self.step(x_data[i], ht, ct)
            else:
                ht, ct = self.step(x_data[i], ht, ct, x_m=x_mask[i])
            if self.direction == 'f':
                output.append(ht)
            else:
                output.insert(0, ht)
        output = torch.stack(output) # list 2 tensor

        if self.batch_first:
            output = output.transpose(0, 1)

        return output

    def init_hidden(self, Bn):
        h0 = CudaVariable(torch.zeros(Bn, self.hidden_size))
        c0 = CudaVariable(torch.zeros(Bn, self.hidden_size))
        return h0, c0

class PRUCell(PRU):
    def __init__(self, input_size, hidden_size, direction='f', batch_first=False, plus=False):
        super(PRUCell, self).__init__(input_size, hidden_size, direction, batch_first, plus)

    def forward(self, xt, tms, x_m=None):
        return self.step(xt, tms[0], tms[1], x_m)

class PRUg(PRU):
    def __init__(self, input_size, hidden_size, direction='f', batch_first=False, plus=False):
        super(PRUg, self).__init__(input_size, hidden_size, direction, batch_first, plus)

        self.input_weights = myLinear(input_size, 3, bias=True)
        self.hidden_weights = myLinear(hidden_size, 3, bias=False)
            
class PRUgCell(PRUg):
    def __init__(self, input_size, hidden_size, direction='f', batch_first=False, plus=False):
        super(PRUgCell, self).__init__(input_size, hidden_size, direction, batch_first, plus)

    def forward(self, xt, tms, x_m=None):
        return self.step(xt, tms[0], tms[1], x_m)

