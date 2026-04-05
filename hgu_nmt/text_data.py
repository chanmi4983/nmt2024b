# -*- coding: utf-8 -*-

# H. Choi, hchoi@handong.edu
import io
import six; from six.moves import cPickle as pkl
import gzip
import numpy as np
import time
from libs.utils import timeSince
from picklable_itertools.extras import equizip

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return io.open(filename, mode, encoding="utf-8")

def read_dict(dic_file, const_id=None):
    with open(dic_file, 'rb') as f:
        word_dict = pkl.load(f, encoding="utf-8")
    new_dict = dict()
    for kk, vv in word_dict.items():
        new_dict[kk] = vv+2 # in the dict file, <s>/</s>=0, <unk>=1
    if const_id is None:
        new_dict['<pad>'] = 0
        new_dict['<s>'] = 1
        new_dict['</s>'] = 2
        new_dict['<unk>'] = 3
    else:
        new_dict['<pad>'] = const_id.PAD
        new_dict['<s>'] = const_id.BOS
        new_dict['</s>'] = const_id.EOS
        new_dict['<unk>'] = const_id.UNK
    return new_dict

class TextPairIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, src_dict, trg_dict,
                 batch_size=128, maxlen=100, 
                 ahead=1, resume_num=0, just_one_epoch=0, const_id=None):
        self.const_id = const_id

        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')

        self.src_dict2 = read_dict(src_dict, const_id=const_id)
        self.trg_dict2 = read_dict(trg_dict, const_id=const_id)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.end_of_data = False
        self.just_one_epoch = just_one_epoch

        self.x_buf =[]
        self.y_buf =[]
        self.buf_remain = 0
        self.cur_line_num=0
        self.ahead=ahead
        self.iters = 0 

        if resume_num > 0:
            self.iters = resume_num
            self.cur_line_num=resume_num * self.batch_size
            for i in range(resume_num * self.batch_size):
                ss = self.source.readline()
                tt = self.target.readline()

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.cur_line_num=0
        self.iters = self.iters + 1

    def __next__(self):
        if self.buf_remain == 0:
            self.x_buf = []
            self.y_buf = []
            i = 0
            while True:
                ss = self.source.readline()
                tt = self.target.readline()
                if ss == "" or tt == "":
                    if self.just_one_epoch:
                        if len(self.x_buf) > 0: 
                            break
                        else:
                            raise StopIteration 
                    else:
                        self.reset()
                    ss = self.source.readline()
                    tt = self.target.readline()

                ss = ss.strip().split()
                tt = tt.strip().split()

                ss = [self.src_dict2.get(key, self.const_id.UNK) for key in ss] 
                tt = [self.trg_dict2.get(key, self.const_id.UNK) for key in tt]

                self.cur_line_num = self.cur_line_num + 1

                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                self.x_buf.append(ss)
                self.y_buf.append(tt)

                i = i + 1
                if i >= self.batch_size*self.ahead:
                    break

            self.buf_remain = self.ahead
            #self.buf_remain = (i-1)/self.batch_size + 1

            if self.ahead > 1: # takes 0 sec for ahead=1000, 1 sec for ahead=2500
                len_xy = [(len(x), len(y), x, y) for x, y in equizip(self.x_buf, self.y_buf)]
                sorted_len_xy = sorted(len_xy, key=lambda xy: (xy[0], xy[1]))
                self.x_buf = [xy[2] for xy in sorted_len_xy]
                self.y_buf = [xy[3] for xy in sorted_len_xy]

        # with self.buf_remain as index
        br = self.ahead-self.buf_remain
        bs = self.batch_size

        source = self.x_buf[br*bs:(br+1)*bs]
        target = self.y_buf[br*bs:(br+1)*bs]

        self.buf_remain = self.buf_remain - 1

        x_data, x_mask, y_data, y_mask = self.prepare_text_pair(source, target)
        self.iters = self.iters + 1
        return x_data, x_mask, y_data, y_mask, self.cur_line_num, self.iters

    # batch preparation
    def prepare_text_pair(self, seqs_x, seqs_y):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]
        lengths_y = [len(s) for s in seqs_y]

        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS
        maxlen_y = np.max(lengths_y) + 2 # for BOS and EOS

        #print(maxlen_x, maxlen_y)

        x_data = np.ones((maxlen_x, n_samples)).astype('int64')*self.const_id.PAD
        y_data = np.ones((maxlen_y, n_samples)).astype('int64')*self.const_id.PAD
        x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
        y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
        for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
            x_data[1:lengths_x[idx]+1, idx] = s_x
            x_data[0, idx] = self.const_id.BOS
            x_data[lengths_x[idx]+1, idx] = self.const_id.EOS
            x_mask[:lengths_x[idx]+2, idx] = 1. 

            y_data[1:lengths_y[idx]+1, idx] = s_y
            y_data[0, idx] = self.const_id.BOS
            y_data[lengths_y[idx]+1, idx] = self.const_id.EOS
            y_mask[:lengths_y[idx]+2, idx] = 1. # extra +2 for BOS/EOS

        return x_data, x_mask, y_data, y_mask


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, data_dict, 
                 batch_size=128, maxlen=50, 
                 ahead=1, resume_num=0, just_one_epoch=0, const_id=None):
        self.source_name = source
        self.const_id = const_id

        self.source = fopen(source, 'r')
        self.data_dict2 = read_dict(data_dict, const_id=const_id)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.end_of_data = False

        self.just_one_epoch = just_one_epoch
        self.x_buf =[]
        self.buf_remain = 0
        self.cur_line_num=0
        self.ahead=ahead
        self.iters = 0

        if resume_num > 0:
            self.cur_line_num=resume_num
            for i in range(resume_num):
                ss = self.source.readline()

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.cur_line_num=0
        self.iters = self.iters + 1

    def __next__(self):

        if self.buf_remain == 0:
            self.x_buf = []
            i = 0
            while True:
                ss = self.source.readline()
                if ss == "":
                    if self.just_one_epoch:
                        if len(self.x_buf) > 0:
                            break
                        else:
                            raise StopIteration # validation
                    else:
                        self.reset()
                    ss = self.source.readline()

                ss = ss.strip().split()
                ss = [self.data_dict2.get(key, self.const_id.UNK) for key in ss] 
                self.cur_line_num = self.cur_line_num + 1

                if len(ss) > self.maxlen:
                    continue

                self.x_buf.append(ss)

                if len(self.x_buf) >= self.batch_size*self.ahead:
                    break

            self.buf_remain = self.ahead

            if self.ahead > 1:
                len_xs = [(len(x), x) for x in self.x_buf]
                sorted_len_xs = sorted(len_xs, key=lambda xs: xs[0])
                self.x_buf = [xs[1] for xs in sorted_len_xs]

        # with self.buf_remain as index
        br = self.ahead-self.buf_remain
        bs = self.batch_size

        source = self.x_buf[br*bs:(br+1)*bs]

        self.buf_remain = self.buf_remain - 1

        x_data, x_mask = self.prepare_text(source)
        self.iters = self.iters + 1
        return x_data, x_mask, self.cur_line_num, self.iters

    # batch preparation, returns padded batch and mask
    def prepare_text(self, seqs_x):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]
        n_samples = len(seqs_x)

        maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS

        x_data = np.ones((maxlen_x, n_samples)).astype('int64')*self.const_id.PAD 
        x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
        for idx, s_x in enumerate(seqs_x):
            x_data[1:lengths_x[idx]+1, idx] = s_x
            x_data[0, idx] = self.const_id.BOS
            x_data[lengths_x[idx]+1, idx] = self.const_id.EOS
            x_mask[:lengths_x[idx]+2, idx] = 1. 

        return x_data, x_mask


if __name__ == "__main__":
    import nmt_const as Const
    from libs.utils import ids2words


    base_dir = '/home/torch/data/data_en_fi_org/subword/'
    src_file = base_dir + 'TR_data.en.tok.shuf.sub'
    trg_file = base_dir + 'TR_data.fi.tok.shuf.sub'
    src_dict = base_dir + 'vocab.en.pkl'
    trg_dict = base_dir + 'vocab.fi.pkl'

    src_dict2 = read_dict(src_dict, const_id=Const)
    src_inv_dict = dict()
    for kk, vv in src_dict2.items():
        src_inv_dict[vv] = kk

    train_iter = TextPairIterator(src_file, trg_file, src_dict, trg_dict,
                         batch_size=3, maxlen=1000, 
                         ahead=1, resume_num=0, const_id=Const, just_one_epoch=1)

    idx = 0
    for x, xm, y, ym, tmp1, tmp2, i in train_iter:
        #print (len(x))
        #print (len(y))
        #print (x)
        #print (xm)
        sentence = ids2words(src_inv_dict, x[:,0], eos_id=Const.EOS)
        print(sentence)
        sentence = ids2words(src_inv_dict, x[:,1], eos_id=Const.EOS)
        print(sentence)
        sentence = ids2words(src_inv_dict, x[:,2], eos_id=Const.EOS)
        print(sentence)
        idx = idx + 1
        if idx >= 3:
            break
        
    valid_iter = TextIterator(src_file, src_dict, 
                    batch_size=1, maxlen=1000, ahead=1, resume_num=0, 
                    const_id=Const)
    for x, xm, tmp, i in valid_iter:
        #print (len(x))
        #print (x)
        sentence = ids2words(src_inv_dict, x[:,0], eos_id=Const.EOS)
        print(sentence)
        print (xm)
        idx = idx + 1
        if idx >= 3:
            break
