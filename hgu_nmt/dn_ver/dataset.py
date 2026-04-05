import os
import argparse

import io
import six; from six.moves import cPickle as pkl
import numpy as np
import random
import math
import copy
import time
import shutil

import nmt_const as Const
from libs.utils import timeSince, ids2words, save_checkpoint, CountDown, load_metrics, PostProcess

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

def read_dict(dic_file, const_id=None, ctc_dict=False):
    with open(dic_file, 'rb') as f:
        src_dict = pkl.load(f, encoding="utf-8")
    src_dict2 = dict()
    for kk, vv in src_dict.items():
        src_dict2[kk] = vv+2 # in the dict file, <s>/</s>=0, <unk>=1
    if const_id is None:
        src_dict2['<pad>'] = 0
        src_dict2['<s>'] = 1
        src_dict2['</s>'] = 2
        src_dict2['<unk>'] = 3
    else:
        src_dict2[const_id.PAD_WORD] = const_id.PAD
        src_dict2[const_id.BOS_WORD] = const_id.BOS
        src_dict2[const_id.EOS_WORD] = const_id.EOS
        src_dict2[const_id.UNK_WORD] = const_id.UNK

    return src_dict2


class MultiTextPairIterator():
    def __init__(self, addresses, src_dict, trg_dict, batch_sizes, maxlen, ahead,\
                 seed=0, rank=0, world_size=1, ctc_dict=False):
        self.addresses = addresses

        self.vocab_dict = {}
        self.vocab_dict['src'] = read_dict(src_dict, const_id=Const, ctc_dict=ctc_dict)
        self.vocab_dict['trg'] = read_dict(trg_dict, const_id=Const, ctc_dict=ctc_dict)

        self.batch_sizes = batch_sizes
        self.rank_batch_sizes = [b // world_size for b in batch_sizes]
        print("RANK {} | Batch sizes : {}".format(rank, self.rank_batch_sizes))

        self.maxlen = maxlen
        self.ahead = ahead
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.sampling_seed = seed

        # pre-process
        self.n_datasets = len(addresses)
        self.data_idxes = [0] * self.n_datasets
        self.offsets = [0] * self.n_datasets
        self.epochs = [0] * self.n_datasets

        self.dataset_types = []
        for n in range(self.n_datasets):
            (source_addr, target_addr) = self.addresses[n]
            dataset_type = self.type_of_dataset(source_addr, target_addr)
            self.dataset_types += [dataset_type]

        self.cursors = {}
        self.cursor_idxes = {}
        self.buffers = {}
        for n in range(self.n_datasets):
            (source_addr, target_addr) = self.addresses[n]
            dataset_type = self.dataset_types[n]
            print("Reading the cursors for {}-th {}-type dataset".format(n, dataset_type))
            self.cursors[('src',n)], self.cursors[('trg',n)], self.cursor_idxes[n] =\
                 self.init_cursors(source_addr, target_addr, dataset_type)
            print("N data : ", len(self.cursor_idxes[n]))
            self.shuffle_cursors(n, self.seed)
            self.buffers[('src',n)], self.buffers[('trg',n)], self.buffers[('idx',n)] = [], [], []
            self.fill_buffer(n)


    def __iter__(self):
        return self

    def fopen(self, filename, mode='r'):
        if filename.endswith('.gz'):
            return gzip.open(filename, mode)
        return io.open(filename, mode, encoding="utf-8")

    def prepare_text(self, seqs_x):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]

        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS

        x_data = np.ones((n_samples, maxlen_x)).astype('int64')*Const.PAD
        x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        for idx, s_x in enumerate(seqs_x):
            x_data[idx, 1:lengths_x[idx]+1] = s_x
            x_data[idx, 0] = Const.BOS
            x_data[idx, lengths_x[idx]+1] = Const.EOS
            x_mask[idx, :lengths_x[idx]+2] = 1.

        return x_data, x_mask

    def prepare_text_pair(self, seqs_x, seqs_y):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]
        lengths_y = [len(s) for s in seqs_y]

        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS
        maxlen_y = np.max(lengths_y) + 2 # for BOS and EOS

        x_data = np.ones((n_samples, maxlen_x)).astype('int64')*Const.PAD
        y_data = np.ones((n_samples, maxlen_y)).astype('int64')*Const.PAD
        x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        y_mask = np.zeros((n_samples, maxlen_y)).astype('float32')
        for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
            x_data[idx, 1:lengths_x[idx]+1] = s_x
            x_data[idx, 0] = Const.BOS
            x_data[idx, lengths_x[idx]+1] = Const.EOS
            x_mask[idx, :lengths_x[idx]+2] = 1.

            y_data[idx, 1:lengths_y[idx]+1] = s_y
            y_data[idx, 0] = Const.BOS
            y_data[idx, lengths_y[idx]+1] = Const.EOS
            y_mask[idx, :lengths_y[idx]+2] = 1. # extra +2 for BOS/EOS

        return x_data, x_mask, y_data, y_mask

    def type_of_dataset(self, source_addr, target_addr):
        dataset_type = ''
        if source_addr is None and target_addr is not None:
            dataset_type = 'trg_mono'
        elif source_addr is not None and target_addr is None:
            dataset_type = 'src_mono'
        elif source_addr is not None and target_addr is not None:
            dataset_type = 'bi'
        else:
            raise SyntaxError("Both of dataset addresses are None..")
        return dataset_type

    def init_cursors(self, source_addr, target_addr, dataset_type):

        def init_bi_cursors(source, target, maxlen):
            src_cursors = []
            trg_cursors = []
            prev_src_cursor, prev_trg_cursor = 0, 0
            source.seek(0)
            target.seek(0)
            while True:
                ss = source.readline()
                tt = target.readline()

                if ss == "" and tt == "":
                    break

                ss = ss.strip().split()
                tt = tt.strip().split()

                if len(ss) <= maxlen and len(tt) <= maxlen:
                    src_cursors.append(prev_src_cursor)
                    trg_cursors.append(prev_trg_cursor)

                prev_src_cursor = source.tell()
                prev_trg_cursor = target.tell()

            return src_cursors, trg_cursors

        def init_mono_cursors(source, maxlen):
            src_cursors = []
            prev_src_cursor = 0
            source.seek(0)
            while True:
                ss = source.readline()

                if ss == "":
                    break

                ss = ss.strip().split()

                if len(ss) <= maxlen:
                    src_cursors.append(prev_src_cursor)
                prev_src_cursor = source.tell()

            return src_cursors

        src_cursors = []
        trg_cursors = []
        if dataset_type == 'bi':
            source = self.fopen(source_addr, 'r')
            target = self.fopen(target_addr, 'r')
            src_cursors, trg_cursors = init_bi_cursors(source, target, self.maxlen)
            source.close()
            target.close()
        elif dataset_type == 'src_mono':
            source = self.fopen(source_addr, 'r')
            src_cursors = init_mono_cursors(source, self.maxlen)
            source.close()
        elif dataset_type == 'trg_mono':
            target = self.fopen(target_addr, 'r')
            trg_cursors = init_mono_cursors(target, self.maxlen)
            target.close()
        cursor_idxes = np.arange(0,max(len(src_cursors),len(trg_cursors))).tolist()
        return src_cursors, trg_cursors, cursor_idxes

    def shuffle_cursors(self, dataset_idx, seed):
        random.seed(seed)
        random.shuffle(self.cursor_idxes[dataset_idx])

    def fill_buffer(self, dataset_idx):

        def load_bi_buffer(source, target, cursor_idxes, src_cursors, trg_cursors, offset, ahead,\
                     src_dict, trg_dict):
            if ahead == 0:
                return [], [], []

            tmp_cursor_idxes = cursor_idxes[offset:offset+ahead]
            src_cursors = [src_cursors[idx] for idx in tmp_cursor_idxes]
            trg_cursors = [trg_cursors[idx] for idx in tmp_cursor_idxes]

            new_src_buffer = []
            new_trg_buffer = []

            for tmp_src_cursor, tmp_trg_cursor in zip(src_cursors, trg_cursors):
                source.seek(tmp_src_cursor)
                target.seek(tmp_trg_cursor)

                ss = source.readline()
                tt = target.readline()

                if ss == "" or tt == "":
                    raise SyntaxError("While filling buffer, I met the last line. Check the cursors")

                ss = ss.strip().split()
                tt = tt.strip().split()

                ss = [src_dict.get(key, Const.UNK) for key in ss]
                tt = [trg_dict.get(key, Const.UNK) for key in tt]

                new_src_buffer.append(ss)
                new_trg_buffer.append(tt)

            return new_src_buffer, new_trg_buffer, tmp_cursor_idxes

        def load_mono_buffer(source, cursor_idxes, src_cursors, offset, ahead, src_dict):
            if ahead == 0:
                return [], []

            tmp_cursor_idxes = cursor_idxes[offset:offset+ahead]
            src_cursors = [src_cursors[idx] for idx in tmp_cursor_idxes]

            new_src_buffer = []

            for tmp_src_cursor in src_cursors:
                source.seek(tmp_src_cursor)

                ss = source.readline()

                if ss == "":
                    raise SyntaxError("While filling buffer, I met the last line. Check the cursors")

                ss = ss.strip().split()

                ss = [src_dict.get(key, Const.UNK) for key in ss]

                new_src_buffer.append(ss)

            return new_src_buffer, tmp_cursor_idxes

        def load_buffer_main(self, source_addr, target_addr, cursor_idxes, src_cursors, trg_cursors,\
                            offset, ahead, src_dict, trg_dict, dataset_type):
            src_buffer = []
            trg_buffer = []
            idx_buffer = []
            if dataset_type == 'bi':
                source = self.fopen(source_addr, 'r')
                target = self.fopen(target_addr, 'r')
                src_buffer, trg_buffer, idx_buffer =\
                        load_bi_buffer(source, target, cursor_idxes, src_cursors, trg_cursors,\
                            offset, ahead, src_dict, trg_dict)
                source.close()
                target.close()
            elif dataset_type == 'src_mono':
                source = self.fopen(source_addr, 'r')
                src_buffer, idx_buffer =\
                        load_mono_buffer(source, cursor_idxes, src_cursors, offset, ahead, src_dict)
                source.close()
            elif dataset_type == 'trg_mono':
                target = self.fopen(target_addr, 'r')
                trg_buffer, idx_buffer =\
                        load_mono_buffer(target, cursor_idxes, trg_cursors, offset, ahead, trg_dict)
                target.close()
            return src_buffer, trg_buffer, idx_buffer

        n_data = len(self.cursor_idxes[dataset_idx])
        offset = self.offsets[dataset_idx]
        n_remain_dataset = n_data - offset

        (source_addr, target_addr) = self.addresses[dataset_idx]
        dataset_type = self.dataset_types[dataset_idx]

        # First try to fill buffer
        if offset + self.ahead < n_data:
            ahead = self.ahead
            end_of_epoch = False
        else:
            ahead = n_remain_dataset
            end_of_epoch = True

        self.buffers[('src',dataset_idx)], self.buffers[('trg',dataset_idx)],\
        self.buffers[('idx',dataset_idx)]\
             = load_buffer_main(self, source_addr, target_addr, self.cursor_idxes[dataset_idx],\
                            self.cursors[('src',dataset_idx)], self.cursors[('trg',dataset_idx)],\
                            self.offsets[dataset_idx], ahead, self.vocab_dict['src'],\
                            self.vocab_dict['trg'], dataset_type)

        # Second try to fill buffer after shuffling the cursor idxes
        if end_of_epoch == True:
            # initialize the offset
            self.offsets[dataset_idx] = 0
            # update the epoch
            self.epochs[dataset_idx] += 1
            # shuffle the cursor with the same seed of other ranks
            self.shuffle_cursors(dataset_idx, self.seed + self.epochs[dataset_idx])

            # fill the remain part of the buffer
            ahead = self.ahead - n_remain_dataset

            additional_src_buffer, additional_trg_buffer, additional_idx_buffer\
                 = load_buffer_main(self, source_addr, target_addr, self.cursor_idxes[dataset_idx],\
                                self.cursors[('src',dataset_idx)], self.cursors[('trg',dataset_idx)],\
                                self.offsets[dataset_idx], ahead, self.vocab_dict['src'],\
                                self.vocab_dict['trg'], dataset_type)

            self.buffers[('src',dataset_idx)] += additional_src_buffer
            self.buffers[('trg',dataset_idx)] += additional_trg_buffer
            self.buffers[('idx',dataset_idx)] += additional_idx_buffer

        self.offsets[dataset_idx] += ahead

        if dataset_type == 'bi':
            if len(self.buffers[('src',dataset_idx)]) != self.ahead or\
                len(self.buffers[('trg',dataset_idx)]) != self.ahead:
                raise SyntaxError("New bi buffer does not contain data as the amount of ahead")
        elif dataset_type == 'src_mono':
            if len(self.buffers[('src',dataset_idx)]) != self.ahead:
                raise SyntaxError("New src mono buffer does not contain data as the amount of ahead")
        elif dataset_type == 'trg_mono':
            if len(self.buffers[('trg',dataset_idx)]) != self.ahead:
                raise SyntaxError("New trg mono buffer does not contain data as the amount of ahead")

    def decide_batch_sizes(self):
        # Warning : This function is made for Backtranslation (Bilingual, Src Monolingual, Trg Monolingual)
        batch_sizes = copy.deepcopy(self.batch_sizes)
        rank_batch_sizes = copy.deepcopy(self.rank_batch_sizes)
        return batch_sizes, rank_batch_sizes


    def __next__(self):

        def load_bi_batch(src_buffer, trg_buffer, idx_buffer, begin_idx, end_idx):
            src_minibatch = src_buffer[begin_idx:end_idx]
            trg_minibatch = trg_buffer[begin_idx:end_idx]
            idx_minibatch = idx_buffer[begin_idx:end_idx]
            return src_minibatch, trg_minibatch, idx_minibatch

        def load_mono_batch(src_buffer, idx_buffer, data_idx, batch_size):
            src_minibatch = src_buffer[begin_idx:end_idx]
            idx_minibatch = idx_buffer[begin_idx:end_idx]
            return src_minibatch, idx_minibatch

        def load_batch_main(src_buffer, trg_buffer, idx_buffer, begin_idx, end_idx, dataset_type):
            src_minibatch = []
            trg_minibatch = []
            idx_minibatch = []
            if dataset_type == 'bi':
                src_minibatch, trg_minibatch, idx_minibatch = load_bi_batch(\
                            src_buffer, trg_buffer, idx_buffer, begin_idx, end_idx)
            elif dataset_type == 'src_mono':
                src_minibatch, idx_minibatch = load_mono_batch(\
                            src_buffer, idx_buffer, begin_idx, end_idx)
            elif dataset_type == 'trg_mono':
                trg_minibatch, idx_minibatch = load_mono_batch(\
                            trg_buffer, idx_buffer, begin_idx, end_idx)
            return src_minibatch, trg_minibatch, idx_minibatch

        # Initialize output minibatches
        src_minibatches = [0] * self.n_datasets
        trg_minibatches = [0] * self.n_datasets
        idx_minibatches = [0] * self.n_datasets

        # Loop for every datasets
        for n in range(self.n_datasets):
            dataset_type = self.dataset_types[n]
            
            batch_sizes, rank_batch_sizes = self.decide_batch_sizes()
            batch_size = batch_sizes[n]
            rank_batch_size = rank_batch_sizes[n]
            data_idx = self.data_idxes[n]
            rank_data_idx = data_idx + self.rank*rank_batch_size

            n_buffer = len(self.buffers[('idx',n)])

            # First try to load batch
            if data_idx + batch_size <= n_buffer:
                begin_idx = rank_data_idx
                end_idx = rank_data_idx+rank_batch_size
                refill_flag = False
            else:
                if rank_data_idx <= n_buffer:
                    begin_idx = rank_data_idx
                    end_idx = min(begin_idx+rank_batch_size, n_buffer)
                else:
                    begin_idx = rank_data_idx
                    end_idx = begin_idx
                refill_flag = True

            src_minibatches[n], trg_minibatches[n], idx_minibatches[n] = \
                        load_batch_main(self.buffers[('src',n)], self.buffers[('trg',n)],\
                                        self.buffers[('idx',n)], begin_idx, end_idx, dataset_type)


            # Second try to load batch from new buffer
            n_current_minibatch = len(idx_minibatches[n])
            if refill_flag == True:
                begin_idx = max(end_idx - n_buffer, 0)
                end_idx = begin_idx + rank_batch_size - n_current_minibatch

                # refill buffer
                self.fill_buffer(n)

                additional_src_minibatch, additional_trg_minibatch, additional_idx_minibatch = \
                        load_batch_main(self.buffers[('src',n)], self.buffers[('trg',n)],\
                                        self.buffers[('idx',n)], begin_idx, end_idx, dataset_type)

                src_minibatches[n] += additional_src_minibatch
                trg_minibatches[n] += additional_trg_minibatch
                idx_minibatches[n] += additional_idx_minibatch

            # update data idx
            self.data_idxes[n] = data_idx + batch_size if refill_flag == False\
                             else data_idx + batch_size - n_buffer

            # check the minibatch satisfies the required batch size
            if dataset_type == 'bi':
                if len(src_minibatches[n]) != rank_batch_sizes[n] or\
                   len(trg_minibatches[n]) != rank_batch_sizes[n]:
                    raise SyntaxError("Generated bilingual minibatch does not satisfy the batch size")
            elif dataset_type == 'src_mono':
                if len(src_minibatches[n]) != rank_batch_sizes[n]:
                    raise SyntaxError("Generated src monolingual minibatch does not satisfy the batch size")
            elif dataset_type == 'trg_mono':
                if len(trg_minibatches[n]) != rank_batch_sizes[n]:
                    raise SyntaxError("Generated trg monolingual minibatch does not satisfy the batch size")

        # preprocess the minibatches (padding, mask and etc.)
        src_data = [0] * self.n_datasets
        src_mask = [0] * self.n_datasets
        trg_data = [0] * self.n_datasets
        trg_mask = [0] * self.n_datasets
        for n in range(self.n_datasets):
            dataset_type = self.dataset_types[n]
            if dataset_type == 'bi':
                src_data[n], src_mask[n], trg_data[n], trg_mask[n] =\
                    self.prepare_text_pair(src_minibatches[n], trg_minibatches[n])
            elif dataset_type == 'src_mono':
                src_data[n], src_mask[n] = self.prepare_text(src_minibatches[n])
                trg_data[n], trg_mask[n] = [0], [0]
            elif dataset_type == 'trg_mono':
                src_data[n], src_mask[n] = [0], [0]
                trg_data[n], trg_mask[n] = self.prepare_text(trg_minibatches[n])


        # return the total minibatches
        return src_data, src_mask, trg_data, trg_mask, idx_minibatches # (N, B, Tmax)

class MultiTextPairIterator_TokenBased():
    def __init__(self, addresses, src_dict, trg_dict, token_sizes, ahead, max_lengths,\
                 seed=0, rank=0, world_size=1, sorting=False, ctc_dict=False):
        self.addresses = addresses

        self.vocab_dict = {}
        self.vocab_dict['src'] = read_dict(src_dict, const_id=Const, ctc_dict=ctc_dict)
        self.vocab_dict['trg'] = read_dict(trg_dict, const_id=Const, ctc_dict=ctc_dict)

        self.token_sizes = token_sizes
        self.rank_token_sizes = [t // world_size for t in token_sizes]
        print("RANK {} | Token sizes : {}".format(rank, self.rank_token_sizes))

        self.ahead = ahead
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.sampling_seed = seed
        self.sorting = sorting

        # pre-process
        self.n_datasets = len(addresses)
        self.sample_idxes = [0] * self.n_datasets
        self.offsets = [0] * self.n_datasets
        self.epochs = [0] * self.n_datasets

        self.dataset_types = []
        for n in range(self.n_datasets):
            (source_addr, target_addr) = self.addresses[n]
            dataset_type = self.type_of_dataset(source_addr, target_addr)
            self.dataset_types += [dataset_type]

        self.cursors = {}
        self.cursor_idxes = {}
        self.buffers = {}
        self.buffer_lens = {}
        for n in range(self.n_datasets):
            (source_addr, target_addr) = self.addresses[n]
            dataset_type = self.dataset_types[n]
            print("Reading the cursors for {}-th {}-type dataset".format(n, dataset_type))
            self.cursors[('src',n)], self.cursors[('trg',n)], self.cursor_idxes[n] =\
                 self.init_cursors(source_addr, target_addr, dataset_type, max_lengths[n])
            print("N data : ", len(self.cursor_idxes[n]))
            self.shuffle_cursors(n, self.seed)
            self.buffers[('src',n)], self.buffers[('trg',n)], self.buffers[('idx',n)] = [], [], []
            self.buffer_lens[('src',n)], self.buffer_lens[('trg',n)] = [], []
            self.fill_buffer(n)


    def __iter__(self):
        return self

    def fopen(self, filename, mode='r'):
        if filename.endswith('.gz'):
            return gzip.open(filename, mode)
        return io.open(filename, mode, encoding="utf-8")

    def prepare_text(self, seqs_x):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]

        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS

        x_data = np.ones((n_samples, maxlen_x)).astype('int64')*Const.PAD
        x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        for idx, s_x in enumerate(seqs_x):
            x_data[idx, 1:lengths_x[idx]+1] = s_x
            x_data[idx, 0] = Const.BOS
            x_data[idx, lengths_x[idx]+1] = Const.EOS
            x_mask[idx, :lengths_x[idx]+2] = 1.

        return x_data, x_mask

    def prepare_text_pair(self, seqs_x, seqs_y):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]
        lengths_y = [len(s) for s in seqs_y]

        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS
        maxlen_y = np.max(lengths_y) + 2 # for BOS and EOS

        x_data = np.ones((n_samples, maxlen_x)).astype('int64')*Const.PAD
        y_data = np.ones((n_samples, maxlen_y)).astype('int64')*Const.PAD
        x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        y_mask = np.zeros((n_samples, maxlen_y)).astype('float32')
        for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
            x_data[idx, 1:lengths_x[idx]+1] = s_x
            x_data[idx, 0] = Const.BOS
            x_data[idx, lengths_x[idx]+1] = Const.EOS
            x_mask[idx, :lengths_x[idx]+2] = 1.

            y_data[idx, 1:lengths_y[idx]+1] = s_y
            y_data[idx, 0] = Const.BOS
            y_data[idx, lengths_y[idx]+1] = Const.EOS
            y_mask[idx, :lengths_y[idx]+2] = 1. # extra +2 for BOS/EOS

        return x_data, x_mask, y_data, y_mask

    def type_of_dataset(self, source_addr, target_addr):
        dataset_type = ''
        if source_addr is None and target_addr is not None:
            dataset_type = 'trg_mono'
        elif source_addr is not None and target_addr is None:
            dataset_type = 'src_mono'
        elif source_addr is not None and target_addr is not None:
            dataset_type = 'bi'
        else:
            raise SyntaxError("Both of dataset addresses are None..")
        return dataset_type

    def init_cursors(self, source_addr, target_addr, dataset_type, max_length):

        def init_bi_cursors(source, target, max_length):
            src_cursors = []
            trg_cursors = []
            prev_src_cursor, prev_trg_cursor = 0, 0
            source.seek(0)
            target.seek(0)

            while True:
                ss = source.readline()
                tt = target.readline()

                if ss == "" and tt == "":
                    break

                ss = ss.strip().split()
                tt = tt.strip().split()

                if len(ss) <= max_length and len(tt) <= max_length:
                    src_cursors.append(prev_src_cursor)
                    trg_cursors.append(prev_trg_cursor)

                prev_src_cursor = source.tell()
                prev_trg_cursor = target.tell()

            return src_cursors, trg_cursors

        def init_mono_cursors(source, max_length):
            src_cursors = []
            prev_src_cursor = 0
            source.seek(0)
            while True:
                ss = source.readline()

                if ss == "":
                    break

                ss = ss.strip().split()

                if len(ss) <= max_length:
                    src_cursors.append(prev_src_cursor)
                prev_src_cursor = source.tell()

            return src_cursors

        src_cursors = []
        trg_cursors = []
        if dataset_type == 'bi':
            source = self.fopen(source_addr, 'r')
            target = self.fopen(target_addr, 'r')
            src_cursors, trg_cursors = init_bi_cursors(source, target, max_length)
            source.close()
            target.close()
        elif dataset_type == 'src_mono':
            source = self.fopen(source_addr, 'r')
            src_cursors = init_mono_cursors(source, max_length)
            source.close()
        elif dataset_type == 'trg_mono':
            target = self.fopen(target_addr, 'r')
            trg_cursors = init_mono_cursors(target, max_length)
            target.close()
        cursor_idxes = np.arange(0,max(len(src_cursors),len(trg_cursors))).tolist()
        return src_cursors, trg_cursors, cursor_idxes


    def shuffle_cursors(self, dataset_idx, seed):
        random.seed(seed)
        random.shuffle(self.cursor_idxes[dataset_idx])

    def fill_buffer(self, dataset_idx):

        def compute_lens(data_buffer):
            buffer_lens = []
            for i in range(len(data_buffer)):
                buffer_lens.append(len(data_buffer[i]))
            return buffer_lens

        def load_bi_buffer(source, target, cursor_idxes, src_cursors, trg_cursors, offset, ahead,\
                     src_dict, trg_dict, sorting):
            if ahead == 0:
                return [], [], [], [], []

            tmp_cursor_idxes = cursor_idxes[offset:offset+ahead]
            src_cursors = [src_cursors[idx] for idx in tmp_cursor_idxes]
            trg_cursors = [trg_cursors[idx] for idx in tmp_cursor_idxes]

            new_src_buffer = []
            new_trg_buffer = []

            for tmp_src_cursor, tmp_trg_cursor in zip(src_cursors, trg_cursors):
                source.seek(tmp_src_cursor)
                target.seek(tmp_trg_cursor)

                ss = source.readline()
                tt = target.readline()

                if ss == "" or tt == "":
                    raise SyntaxError("While filling buffer, I met the last line. Check the cursors")

                ss = ss.strip().split()
                tt = tt.strip().split()

                ss = [src_dict.get(key, Const.UNK) for key in ss]
                tt = [trg_dict.get(key, Const.UNK) for key in tt]

                new_src_buffer.append(ss)
                new_trg_buffer.append(tt)

            if sorting == True:
                src_len_args = [len(tmp) for tmp in new_src_buffer]
                _, src_len_args = torch.sort(torch.tensor(src_len_args))
                src_len_args = src_len_args.tolist()

                new_src_buffer = [new_src_buffer[i] for i in src_len_args]
                new_trg_buffer = [new_trg_buffer[i] for i in src_len_args]
                tmp_cursor_idxes = [tmp_cursor_idxes[i] for i in src_len_args]

            new_src_buffer_len = compute_lens(new_src_buffer)
            new_trg_buffer_len = compute_lens(new_trg_buffer)

            return new_src_buffer, new_src_buffer_len,\
                   new_trg_buffer, new_trg_buffer_len, tmp_cursor_idxes

        def load_mono_buffer(source, cursor_idxes, src_cursors, offset, ahead, src_dict, sorting):
            if ahead == 0:
                return [], [], []

            tmp_cursor_idxes = cursor_idxes[offset:offset+ahead]
            src_cursors = [src_cursors[idx] for idx in tmp_cursor_idxes]

            new_src_buffer = []

            for tmp_src_cursor in src_cursors:
                source.seek(tmp_src_cursor)

                ss = source.readline()

                if ss == "":
                    raise SyntaxError("While filling buffer, I met the last line. Check the cursors")

                ss = ss.strip().split()

                ss = [src_dict.get(key, Const.UNK) for key in ss]

                new_src_buffer.append(ss)

            if sorting == True:
                src_len_args = [len(tmp) for tmp in new_src_buffer]
                _, src_len_args = torch.sort(torch.tensor(src_len_args))
                src_len_args = src_len_args.tolist()

                new_src_buffer = [new_src_buffer[i] for i in src_len_args]
                tmp_cursor_idxes = [tmp_cursor_idxes[i] for i in src_len_args]
        
            new_src_buffer_len = compute_lens(new_src_buffer)

            return new_src_buffer, new_src_buffer_len, tmp_cursor_idxes

        def load_buffer_main(self, source_addr, target_addr, cursor_idxes, src_cursors, trg_cursors,\
                            offset, ahead, src_dict, trg_dict, dataset_type, sorting):
            src_buffer = []
            src_buffer_len = []
            trg_buffer = []
            trg_buffer_len = []
            idx_buffer = []
            if dataset_type == 'bi':
                source = self.fopen(source_addr, 'r')
                target = self.fopen(target_addr, 'r')
                src_buffer, src_buffer_len, trg_buffer, trg_buffer_len, idx_buffer =\
                        load_bi_buffer(source, target, cursor_idxes, src_cursors, trg_cursors,\
                            offset, ahead, src_dict, trg_dict, sorting)
                source.close()
                target.close()
            elif dataset_type == 'src_mono':
                source = self.fopen(source_addr, 'r')
                src_buffer, src_buffer_len, idx_buffer =\
                        load_mono_buffer(source, cursor_idxes, src_cursors, offset, ahead,\
                                         src_dict, sorting)
                source.close()
            elif dataset_type == 'trg_mono':
                target = self.fopen(target_addr, 'r')
                trg_buffer, trg_buffer_len, idx_buffer =\
                        load_mono_buffer(target, cursor_idxes, trg_cursors, offset, ahead,\
                                         trg_dict, sorting)
                target.close()
            return src_buffer, src_buffer_len, trg_buffer, trg_buffer_len, idx_buffer

        # Leave only unused samples from buffer
        remain_src_buffer, remain_src_buffer_lens, remain_trg_buffer, remain_trg_buffer_lens,\
        remain_idx_buffer = [], [], [], [], []
        
        dataset_type = self.dataset_types[dataset_idx]
        sample_idx = self.sample_idxes[dataset_idx]
        if dataset_type == 'bi':
            remain_src_buffer = self.buffers[('src',dataset_idx)][sample_idx:]
            remain_src_buffer_lens = self.buffer_lens[('src',dataset_idx)][sample_idx:]
            remain_trg_buffer = self.buffers[('trg',dataset_idx)][sample_idx:]
            remain_trg_buffer_lens = self.buffer_lens[('trg',dataset_idx)][sample_idx:]
            remain_idx_buffer = self.buffers[('idx',dataset_idx)][sample_idx:]
            if len(remain_src_buffer) > self.ahead:
                raise SyntaxError("Remaining 'bi' buffer has too many unused samples..")
        elif dataset_type == 'src_mono':
            remain_src_buffer = self.buffers[('src',dataset_idx)][sample_idx:]
            remain_src_buffer_lens = self.buffer_lens[('src',dataset_idx)][sample_idx:]
            remain_idx_buffer = self.buffers[('idx',dataset_idx)][sample_idx:]
            if len(remain_src_buffer) > self.ahead:
                raise SyntaxError("Remaining 'src_mono' buffer has too many unused samples..")
        elif dataset_type == 'trg_mono':
            remain_trg_buffer = self.buffers[('trg',dataset_idx)][sample_idx:]
            remain_trg_buffer_lens = self.buffer_lens[('trg',dataset_idx)][sample_idx:]
            remain_idx_buffer = self.buffers[('idx',dataset_idx)][sample_idx:]
            if len(remain_trg_buffer) > self.ahead:
                raise SyntaxError("Remaining 'trg_mono' buffer has too many unused samples..")
           
        self.sample_idxes[dataset_idx] = 0
 

        n_data = len(self.cursor_idxes[dataset_idx])
        offset = self.offsets[dataset_idx]
        n_remain_dataset = n_data - offset

        (source_addr, target_addr) = self.addresses[dataset_idx]
        dataset_type = self.dataset_types[dataset_idx]

        # First try to fill buffer
        if offset + self.ahead <= n_data:
            ahead = self.ahead
            end_of_epoch = False
        else:
            ahead = n_remain_dataset
            end_of_epoch = True

        additional_src_buffer, additional_src_buffer_lens,\
        additional_trg_buffer, additional_trg_buffer_lens,\
        additional_idx_buffer\
             = load_buffer_main(self, source_addr, target_addr, self.cursor_idxes[dataset_idx],\
                            self.cursors[('src',dataset_idx)], self.cursors[('trg',dataset_idx)],\
                            self.offsets[dataset_idx], ahead, self.vocab_dict['src'],\
                            self.vocab_dict['trg'], dataset_type, self.sorting)

        remain_src_buffer += additional_src_buffer
        remain_src_buffer_lens += additional_src_buffer_lens
        remain_trg_buffer += additional_trg_buffer
        remain_trg_buffer_lens += additional_trg_buffer_lens
        remain_idx_buffer += additional_idx_buffer

        # Second try to fill buffer after shuffling the cursor idxes
        if end_of_epoch == True:
            # initialize the offset
            self.offsets[dataset_idx] = 0
            # update the epoch
            self.epochs[dataset_idx] += 1
            # shuffle the cursor with the same seed of other ranks
            self.shuffle_cursors(dataset_idx, self.seed + self.epochs[dataset_idx])

            # fill the remain part of the buffer
            ahead = self.ahead - n_remain_dataset

            additional_src_buffer, additional_src_buffer_lens,\
            additional_trg_buffer, additional_trg_buffer_lens,\
            additional_idx_buffer\
                 = load_buffer_main(self, source_addr, target_addr, self.cursor_idxes[dataset_idx],\
                            self.cursors[('src',dataset_idx)], self.cursors[('trg',dataset_idx)],\
                            self.offsets[dataset_idx], ahead, self.vocab_dict['src'],\
                            self.vocab_dict['trg'], dataset_type, self.sorting)

            remain_src_buffer += additional_src_buffer
            remain_src_buffer_lens += additional_src_buffer_lens
            remain_trg_buffer += additional_trg_buffer
            remain_trg_buffer_lens += additional_trg_buffer_lens
            remain_idx_buffer += additional_idx_buffer

        self.buffers[('src',dataset_idx)] = remain_src_buffer
        self.buffer_lens[('src',dataset_idx)] = remain_src_buffer_lens
        self.buffers[('trg',dataset_idx)] = remain_trg_buffer
        self.buffer_lens[('trg',dataset_idx)] = remain_trg_buffer_lens
        self.buffers[('idx',dataset_idx)] = remain_idx_buffer

        self.offsets[dataset_idx] += ahead


        if dataset_type == 'bi':
            if len(self.buffers[('src',dataset_idx)]) < self.ahead or\
                len(self.buffers[('trg',dataset_idx)]) < self.ahead:
                raise SyntaxError("New bi buffer does not contain data as the amount of ahead")
        elif dataset_type == 'src_mono':
            if len(self.buffers[('src',dataset_idx)]) < self.ahead:
                raise SyntaxError("New src mono buffer does not contain data as the amount of ahead")
        elif dataset_type == 'trg_mono':
            if len(self.buffers[('trg',dataset_idx)]) < self.ahead:
                raise SyntaxError("New trg mono buffer does not contain data as the amount of ahead")

        dist.barrier()

    def decide_batch_sizes(self, dataset_idx, dataset_type, sample_idx, max_rank_tokens):

        def decide_bi_batch_sizes(src_buffer_lens, trg_buffer_lens, sample_idx, max_rank_tokens,\
                                 world_size):
            refill_flag = False
            total_batch_size = 0
            rank_batch_sizes = [0] * self.world_size

            tmp_sample_idx = copy.deepcopy(sample_idx)
            for tmp_rank in range(world_size):
                tmp_max_token = max_rank_tokens[tmp_rank]
                tmp_src_batch_prediction = []
                tmp_trg_batch_prediction = []
                while(True):
                    if tmp_sample_idx >= len(src_buffer_lens) or \
                       tmp_sample_idx >= len(trg_buffer_lens):
                        refill_flag = True
                        return 0, 0, refill_flag
                    else:
                        new_src_batch_prediction = copy.deepcopy(tmp_src_batch_prediction)
                        new_trg_batch_prediction = copy.deepcopy(tmp_trg_batch_prediction)

                        new_src_batch_prediction.append(src_buffer_lens[tmp_sample_idx])
                        new_trg_batch_prediction.append(trg_buffer_lens[tmp_sample_idx])

                        new_src_total_tokens = max(new_src_batch_prediction)\
                                             * len(new_src_batch_prediction)
                        new_trg_total_tokens = max(new_trg_batch_prediction)\
                                             * len(new_trg_batch_prediction)
                        if new_src_total_tokens <= tmp_max_token and \
                           new_trg_total_tokens <= tmp_max_token:
                            tmp_src_batch_prediction = copy.deepcopy(new_src_batch_prediction)
                            tmp_trg_batch_prediction = copy.deepcopy(new_trg_batch_prediction)
                            tmp_sample_idx += 1
                        else:
                            if len(tmp_src_batch_prediction) == 0:
                                refill_flag = True
                                return 0, 0, refill_flag
                            break
                rank_batch_sizes[tmp_rank] = len(tmp_src_batch_prediction)
            total_batch_size = sum(rank_batch_sizes)
            return total_batch_size, rank_batch_sizes, refill_flag

        def decide_mono_batch_sizes(buffer_lens, sample_idx, max_rank_tokens, world_size):
            refill_flag = False
            total_batch_size = 0
            rank_batch_sizes = [0] * self.world_size

            tmp_sample_idx = copy.deepcopy(sample_idx)
            for tmp_rank in range(world_size):
                tmp_max_token = max_rank_tokens[tmp_rank]
                tmp_batch_prediction = []
                while(True):
                    if tmp_sample_idx >= len(buffer_lens):
                        refill_flag = True
                        total_batch_size = sum(rank_batch_sizes)
                        return total_batch_size, rank_batch_sizes, refill_flag
                    else:
                        new_batch_prediction = copy.deepcopy(tmp_batch_prediction)
                        new_batch_prediction.append(buffer_lens[tmp_sample_idx])
                        new_total_tokens = max(new_batch_prediction) * len(new_batch_prediction)
                        if new_total_tokens <= tmp_max_token:
                            tmp_batch_prediction = copy.deepcopy(new_batch_prediction)
                            tmp_sample_idx += 1
                        else:
                            if len(tmp_batch_prediction) == 0:
                                refill_flag = True
                                total_batch_size = sum(rank_batch_sizes)
                                return total_batch_size, rank_batch_sizes, refill_flag
                            break
                rank_batch_sizes[tmp_rank] = len(tmp_batch_prediction)
            total_batch_size = sum(rank_batch_sizes)
            return total_batch_size, rank_batch_sizes, refill_flag

        if dataset_type == 'bi':
            src_buffer_lens = self.buffer_lens[('src', dataset_idx)]
            trg_buffer_lens = self.buffer_lens[('trg', dataset_idx)]
            total_batch_size, rank_batch_sizes, refill_flag =\
                            decide_bi_batch_sizes(src_buffer_lens, trg_buffer_lens,\
                                                sample_idx, max_rank_tokens, self.world_size)

        elif dataset_type == 'src_mono':
            buffer_lens = self.buffer_lens[('src', dataset_idx)]
            total_batch_size, rank_batch_sizes, refill_flag =\
                            decide_mono_batch_sizes(buffer_lens, sample_idx,\
                                                    max_rank_tokens, self.world_size)
        elif dataset_type == 'trg_mono':
            buffer_lens = self.buffer_lens[('trg', dataset_idx)]
            total_batch_size, rank_batch_sizes, refill_flag =\
                            decide_mono_batch_sizes(buffer_lens, sample_idx,\
                                                    max_rank_tokens, self.world_size)
        return total_batch_size, rank_batch_sizes, refill_flag


    def __next__(self):

        def load_bi_batch(src_buffer, trg_buffer, idx_buffer, begin_idx, end_idx):
            src_minibatch = src_buffer[begin_idx:end_idx]
            trg_minibatch = trg_buffer[begin_idx:end_idx]
            idx_minibatch = idx_buffer[begin_idx:end_idx]
            return src_minibatch, trg_minibatch, idx_minibatch

        def load_mono_batch(src_buffer, idx_buffer, data_idx, batch_size):
            src_minibatch = src_buffer[begin_idx:end_idx]
            idx_minibatch = idx_buffer[begin_idx:end_idx]
            return src_minibatch, idx_minibatch

        def load_batch_main(src_buffer, trg_buffer, idx_buffer, begin_idx, end_idx, dataset_type):
            src_minibatch = []
            trg_minibatch = []
            idx_minibatch = []
            if dataset_type == 'bi':
                src_minibatch, trg_minibatch, idx_minibatch = load_bi_batch(\
                            src_buffer, trg_buffer, idx_buffer, begin_idx, end_idx)
            elif dataset_type == 'src_mono':
                src_minibatch, idx_minibatch = load_mono_batch(\
                            src_buffer, idx_buffer, begin_idx, end_idx)
            elif dataset_type == 'trg_mono':
                trg_minibatch, idx_minibatch = load_mono_batch(\
                            trg_buffer, idx_buffer, begin_idx, end_idx)
            return src_minibatch, trg_minibatch, idx_minibatch

        # Initialize output minibatches
        src_minibatches = [0] * self.n_datasets
        trg_minibatches = [0] * self.n_datasets
        idx_minibatches = [0] * self.n_datasets

        # Loop for every datasets
        for n in range(self.n_datasets):
            dataset_type = self.dataset_types[n]
            sample_idx = self.sample_idxes[n]
            max_rank_tokens = [self.rank_token_sizes[n]] * self.world_size
            
            total_batch_size, rank_batch_sizes, refill_flag =\
                        self.decide_batch_sizes(n, dataset_type, sample_idx, max_rank_tokens)

            if refill_flag == True:
                self.fill_buffer(n)
                
                sample_idx = self.sample_idxes[n]
                total_batch_size, rank_batch_sizes, refill_flag = self.decide_batch_sizes(n,\
                                                         dataset_type, sample_idx, max_rank_tokens)
                if refill_flag == True:
                    raise SyntaxError("Refill flag is True after refilling buffer")

            rank_batch_size = rank_batch_sizes[self.rank]
            rank_sample_idx = sample_idx + sum(rank_batch_sizes[:self.rank])

            begin_idx = rank_sample_idx
            end_idx = begin_idx + rank_batch_size

            src_minibatches[n], trg_minibatches[n], idx_minibatches[n] = \
                        load_batch_main(self.buffers[('src',n)], self.buffers[('trg',n)],\
                                        self.buffers[('idx',n)], begin_idx, end_idx, dataset_type)

            # update data idx
            self.sample_idxes[n] = sample_idx + total_batch_size

            # check the minibatch satisfies the required batch size
            if dataset_type == 'bi':
                if len(src_minibatches[n]) != rank_batch_size or\
                   len(trg_minibatches[n]) != rank_batch_size:
                    raise SyntaxError("Generated bilingual minibatch does not satisfy the batch size")
            elif dataset_type == 'src_mono':
                if len(src_minibatches[n]) != rank_batch_size:
                    raise SyntaxError("Generated src monolingual minibatch does not satisfy the batch size")
            elif dataset_type == 'trg_mono':
                if len(trg_minibatches[n]) != rank_batch_size:
                    raise SyntaxError("Generated trg monolingual minibatch does not satisfy the batch size")

        # preprocess the minibatches (padding, mask and etc.)
        src_data = [0] * self.n_datasets
        src_mask = [0] * self.n_datasets
        trg_data = [0] * self.n_datasets
        trg_mask = [0] * self.n_datasets
        for n in range(self.n_datasets):
            dataset_type = self.dataset_types[n]
            if dataset_type == 'bi':
                src_data[n], src_mask[n], trg_data[n], trg_mask[n] =\
                    self.prepare_text_pair(src_minibatches[n], trg_minibatches[n])
            elif dataset_type == 'src_mono':
                src_data[n], src_mask[n] = self.prepare_text(src_minibatches[n])
                trg_data[n], trg_mask[n] = [0], [0]
            elif dataset_type == 'trg_mono':
                src_data[n], src_mask[n] = [0], [0]
                trg_data[n], trg_mask[n] = self.prepare_text(trg_minibatches[n])

        # return the total minibatches
        return src_data, src_mask, trg_data, trg_mask, idx_minibatches # (N, B, Tmax)

class TestPairIterator_TokenBased():
    def __init__(self, address, src_dict, trg_dict, token_size, max_length,\
                 rank=0, world_size=1, sorting=True, ctc_dict=False):
        self.address = address

        self.vocab_dict = {}
        self.vocab_dict['src'] = read_dict(src_dict, const_id=Const, ctc_dict=ctc_dict)
        self.vocab_dict['trg'] = read_dict(trg_dict, const_id=Const, ctc_dict=ctc_dict)

        self.token_size = token_size
        self.rank_token_size = token_size // world_size

        self.ahead = 9999999
        self.rank = rank
        self.world_size = world_size
        self.sorting = sorting
        self.max_length = max_length

        # pre-process
        self.sample_idx = 0
        self.epoch_run = True

        self.cursor = {}
        self.cursor_idx = []
        self.buffer = {}
        self.buffer_len = {}

        (source_addr, target_addr) = self.address
        self.cursor['src'], self.cursor['trg'], self.cursor_idx =\
             self.init_cursors(source_addr, target_addr, self.max_length)
        print("Test N data : ", len(self.cursor_idx))

        self.buffer['src'], self.buffer['trg'], self.buffer['idx'] = [], [], []
        self.buffer_len['src'], self.buffer_len['trg'] = [], []

        self.initialize()
        self.max_rank_batch_size, self.avg_rank_batch_size, self.num_iloop, self.rank_token_size\
                         = self.auto_tune_and_inspect()
        print("RANK {} | Test Token size : {}".format(rank, self.rank_token_size))

    def initialize(self):
        self.sample_idx = 0
        self.epoch_run = True
        self.fill_buffer()

    def __iter__(self):
        return self

    def fopen(self, filename, mode='r'):
        if filename.endswith('.gz'):
            return gzip.open(filename, mode)
        return io.open(filename, mode, encoding="utf-8")

    def prepare_text(self, seqs_x):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]

        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS

        x_data = np.ones((n_samples, maxlen_x)).astype('int64')*Const.PAD
        x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        for idx, s_x in enumerate(seqs_x):
            x_data[idx, 1:lengths_x[idx]+1] = s_x
            x_data[idx, 0] = Const.BOS
            x_data[idx, lengths_x[idx]+1] = Const.EOS
            x_mask[idx, :lengths_x[idx]+2] = 1.

        return x_data, x_mask

    def prepare_text_pair(self, seqs_x, seqs_y):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]
        lengths_y = [len(s) for s in seqs_y]

        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS
        maxlen_y = np.max(lengths_y) + 2 # for BOS and EOS

        x_data = np.ones((n_samples, maxlen_x)).astype('int64')*Const.PAD
        y_data = np.ones((n_samples, maxlen_y)).astype('int64')*Const.PAD
        x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        y_mask = np.zeros((n_samples, maxlen_y)).astype('float32')
        for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
            x_data[idx, 1:lengths_x[idx]+1] = s_x
            x_data[idx, 0] = Const.BOS
            x_data[idx, lengths_x[idx]+1] = Const.EOS
            x_mask[idx, :lengths_x[idx]+2] = 1.

            y_data[idx, 1:lengths_y[idx]+1] = s_y
            y_data[idx, 0] = Const.BOS
            y_data[idx, lengths_y[idx]+1] = Const.EOS
            y_mask[idx, :lengths_y[idx]+2] = 1. # extra +2 for BOS/EOS

        return x_data, x_mask, y_data, y_mask


    def init_cursors(self, source_addr, target_addr, max_length):

        def init_bi_cursors(source, target, max_length):
            src_cursors = []
            trg_cursors = []
            prev_src_cursor, prev_trg_cursor = 0, 0
            source.seek(0)
            target.seek(0)

            while True:
                ss = source.readline()
                tt = target.readline()

                if ss == "" and tt == "":
                    break

                ss = ss.strip().split()
                tt = tt.strip().split()

                if len(ss) <= max_length and len(tt) <= max_length:
                    src_cursors.append(prev_src_cursor)
                    trg_cursors.append(prev_trg_cursor)

                prev_src_cursor = source.tell()
                prev_trg_cursor = target.tell()

            return src_cursors, trg_cursors

        src_cursors = []
        trg_cursors = []

        source = self.fopen(source_addr, 'r')
        target = self.fopen(target_addr, 'r')
        src_cursors, trg_cursors = init_bi_cursors(source, target, max_length)
        source.close()
        target.close()

        cursor_idx = np.arange(0,max(len(src_cursors),len(trg_cursors))).tolist()
        return src_cursors, trg_cursors, cursor_idx

    def fill_buffer(self):

        def compute_lens(data_buffer):
            buffer_lens = []
            for i in range(len(data_buffer)):
                buffer_lens.append(len(data_buffer[i]))
            return buffer_lens

        def load_bi_buffer(source, target, cursor_idx, src_cursors, trg_cursors,\
                     src_dict, trg_dict, sorting):

            tmp_cursor_idx = copy.deepcopy(cursor_idx)
            src_cursors = [src_cursors[idx] for idx in tmp_cursor_idx]
            trg_cursors = [trg_cursors[idx] for idx in tmp_cursor_idx]

            new_src_buffer = []
            new_trg_buffer = []

            for tmp_src_cursor, tmp_trg_cursor in zip(src_cursors, trg_cursors):
                source.seek(tmp_src_cursor)
                target.seek(tmp_trg_cursor)

                ss = source.readline()
                tt = target.readline()

                if ss == "" or tt == "":
                    raise SyntaxError("While filling buffer, I met the last line. Check the cursors")

                ss = ss.strip().split()
                tt = tt.strip().split()

                ss = [src_dict.get(key, Const.UNK) for key in ss]
                tt = [trg_dict.get(key, Const.UNK) for key in tt]

                new_src_buffer.append(ss)
                new_trg_buffer.append(tt)

            if sorting == True:
                src_len_args = [len(tmp) for tmp in new_src_buffer]
                _, src_len_args = torch.sort(torch.tensor(src_len_args))
                src_len_args = src_len_args.tolist()

                new_src_buffer = [new_src_buffer[i] for i in src_len_args]
                new_trg_buffer = [new_trg_buffer[i] for i in src_len_args]
                tmp_cursor_idx = [tmp_cursor_idx[i] for i in src_len_args]

            new_src_buffer_len = compute_lens(new_src_buffer)
            new_trg_buffer_len = compute_lens(new_trg_buffer)

            return new_src_buffer, new_src_buffer_len,\
                   new_trg_buffer, new_trg_buffer_len, tmp_cursor_idx

        src_buffer = []
        src_buffer_len = []
        trg_buffer = []
        trg_buffer_len = []
        idx_buffer = []

        (source_addr, target_addr) = self.address
        source = self.fopen(source_addr, 'r')
        target = self.fopen(target_addr, 'r')
        src_buffer, src_buffer_len, trg_buffer, trg_buffer_len, idx_buffer =\
                load_bi_buffer(source, target, self.cursor_idx,\
                            self.cursor[('src')], self.cursor[('trg')],\
                            self.vocab_dict['src'], self.vocab_dict['trg'], self.sorting)
        source.close()
        target.close()

        self.buffer['src'] = src_buffer
        self.buffer_len['src'] = src_buffer_len
        self.buffer['trg'] = trg_buffer
        self.buffer_len['trg'] = trg_buffer_len
        self.buffer['idx'] = idx_buffer

        dist.barrier()

    def decide_batch_sizes(self, sample_idx, max_rank_tokens):

        def decide_bi_batch_sizes(src_buffer_len, trg_buffer_len, sample_idx, max_rank_tokens,\
                                 world_size):
            epoch_done = False
            total_batch_size = 0
            rank_batch_sizes = [0] * self.world_size

            tmp_sample_idx = copy.deepcopy(sample_idx)
            for tmp_rank in range(world_size):
                tmp_max_token = max_rank_tokens[tmp_rank]
                tmp_src_batch_prediction = []
                tmp_trg_batch_prediction = []
                while(True):
                    if tmp_sample_idx >= len(src_buffer_len) or \
                       tmp_sample_idx >= len(trg_buffer_len):
                        epoch_done = True
                        break
                    else:
                        new_src_batch_prediction = copy.deepcopy(tmp_src_batch_prediction)
                        new_trg_batch_prediction = copy.deepcopy(tmp_trg_batch_prediction)

                        new_src_batch_prediction.append(src_buffer_len[tmp_sample_idx])
                        new_trg_batch_prediction.append(trg_buffer_len[tmp_sample_idx])

                        new_src_total_tokens = max(new_src_batch_prediction)\
                                             * len(new_src_batch_prediction)
                        new_trg_total_tokens = max(new_trg_batch_prediction)\
                                             * len(new_trg_batch_prediction)
                        if new_src_total_tokens <= tmp_max_token and \
                           new_trg_total_tokens <= tmp_max_token:
                            tmp_src_batch_prediction = copy.deepcopy(new_src_batch_prediction)
                            tmp_trg_batch_prediction = copy.deepcopy(new_trg_batch_prediction)
                            tmp_sample_idx += 1
                        else:
                            if len(tmp_src_batch_prediction) == 0:
                                raise SyntaxError("Unexpected:Rank{} does not have enough data".format(tmp_rank))
                            break
                rank_batch_sizes[tmp_rank] = len(tmp_src_batch_prediction)
            total_batch_size = sum(rank_batch_sizes)

            if tmp_sample_idx >= len(src_buffer_len) or tmp_sample_idx >= len(trg_buffer_len):
                epoch_done = True

            return total_batch_size, rank_batch_sizes, epoch_done

        src_buffer_len = self.buffer_len['src']
        trg_buffer_len = self.buffer_len['trg']
        total_batch_size, rank_batch_sizes, epoch_done =\
                        decide_bi_batch_sizes(src_buffer_len, trg_buffer_len,\
                                            sample_idx, max_rank_tokens, self.world_size)
        return total_batch_size, rank_batch_sizes, epoch_done

    def auto_tune_and_inspect(self):

        def verify_token_size(test_token_size):
            sample_idx = 0
            max_rank_batch_size = 0
            avg_rank_batch_size = 0
            num_iloop = 0
            max_rank_tokens = [test_token_size] * self.world_size
            while(True):
                total_batch_size, rank_batch_sizes, epoch_done =\
                            self.decide_batch_sizes(sample_idx, max_rank_tokens)
                for tmp_rank, rank_batch_size in enumerate(rank_batch_sizes):
                    if rank_batch_size == 0:
                        return 0, 0, 0, False

                if epoch_done == True:
                    break
                if max(rank_batch_sizes) > max_rank_batch_size:
                    max_rank_batch_size = max(rank_batch_sizes)
                avg_rank_batch_size += total_batch_size / self.world_size
                sample_idx += total_batch_size
                num_iloop += 1
            avg_rank_batch_size /= num_iloop
            return max_rank_batch_size, avg_rank_batch_size, num_iloop, True

        n_trial = 1
        tmp_rank_token_size = copy.deepcopy(self.rank_token_size)
        for i in range(self.rank_token_size):
            print("{} iterations to tune token_size for the given test dataset".format(i), end="\r")
            max_rank_batch_size, avg_rank_batch_size, num_iloop, pass_flag =\
                                             verify_token_size(tmp_rank_token_size)
            if pass_flag == True:
                return max_rank_batch_size, avg_rank_batch_size, num_iloop, tmp_rank_token_size
            else:
                tmp_rank_token_size -= 1
        raise SyntaxError("Current test_token_size setting does not pass the inspection")

    def __next__(self):

        def load_bi_batch(src_buffer, trg_buffer, idx_buffer, begin_idx, end_idx):
            src_minibatch = src_buffer[begin_idx:end_idx]
            trg_minibatch = trg_buffer[begin_idx:end_idx]
            idx_minibatch = idx_buffer[begin_idx:end_idx]
            return src_minibatch, trg_minibatch, idx_minibatch

        def load_batch_main(src_buffer, trg_buffer, idx_buffer, begin_idx, end_idx, dataset_type):
            src_minibatch = []
            trg_minibatch = []
            idx_minibatch = []

            src_minibatch, trg_minibatch, idx_minibatch = load_bi_batch(\
                        src_buffer, trg_buffer, idx_buffer, begin_idx, end_idx)
            return src_minibatch, trg_minibatch, idx_minibatch

        if self.epoch_run == False:
            return [0], [0], [0], [0], [0], False

        # Initialize output minibatches
        src_minibatches = [0]
        trg_minibatches = [0]
        idx_minibatches = [0]

        # Loop for every datasets
        sample_idx = self.sample_idx
        max_rank_tokens = [self.rank_token_size] * self.world_size

        total_batch_size, rank_batch_sizes, epoch_done =\
                    self.decide_batch_sizes(sample_idx, max_rank_tokens)

        self.epoch_done = epoch_done

        rank_batch_size = rank_batch_sizes[self.rank]
        rank_sample_idx = sample_idx + sum(rank_batch_sizes[:self.rank])

        begin_idx = rank_sample_idx
        end_idx = begin_idx + rank_batch_size

        src_minibatches[0], trg_minibatches[0], idx_minibatches[0] = \
                    load_bi_batch(self.buffer['src'], self.buffer['trg'],\
                                    self.buffer['idx'], begin_idx, end_idx)

        # update data idx
        self.sample_idx = sample_idx + total_batch_size

        # check the minibatch satisfies the required batch size
        if len(src_minibatches[0]) != rank_batch_size or\
           len(trg_minibatches[0]) != rank_batch_size:
            raise SyntaxError("Generated bilingual minibatch does not satisfy the batch size")
        if len(src_minibatches[0]) == 0 or\
           len(trg_minibatches[0]) == 0:
            raise SyntaxError("Minibatch contains no sample")

        # preprocess the minibatches (padding, mask and etc.)
        src_data = [0]
        src_mask = [0]
        trg_data = [0]
        trg_mask = [0]

        src_data[0], src_mask[0], trg_data[0], trg_mask[0] =\
            self.prepare_text_pair(src_minibatches[0], trg_minibatches[0])

        # return the total minibatches
        return src_data, src_mask, trg_data, trg_mask, idx_minibatches, not self.epoch_done # (1, B, Tmax)


class MyTestDataset(Dataset):
    def __init__(self, source, target, ref_source, ref_target, token_ref_source, token_ref_target,\
                     src_dict, trg_dict, ahead, ctc_dict=False):
        super(MyTestDataset, self).__init__()
        self.source = source
        self.ref_source = ref_source
        self.token_ref_source = token_ref_source
        self.target = target
        self.ref_target = ref_target
        self.token_ref_target = token_ref_target

        self.src_dict2 = read_dict(src_dict, const_id=Const, ctc_dict=ctc_dict)
        self.trg_dict2 = read_dict(trg_dict, const_id=Const, ctc_dict=ctc_dict)

        self.source_dataset = []
        self.target_dataset = []
        self.buffer_indexes = []

        self.offset = 0

        self.epoch = 0

        self.init_process(0)

        self.ahead = ahead

    def fopen(self, filename, mode='r'):
        if filename.endswith('.gz'):
            return gzip.open(filename, mode)
        return io.open(filename, mode, encoding="utf-8")

    def init_process(self, epoch):
        self.src_seeks, self.trg_seeks = self.find_seeks(self.source, self.target)
        if len(self.src_seeks) != len(self.trg_seeks):
            raise SyntaxError("Seek points of bilingual datasets are not matched..")
        source_references, target_references = self.read_references(\
                                self.ref_source, self.ref_target)
        source_tok_references, target_tok_references = self.read_references(\
                                self.token_ref_source, self.token_ref_target)
        if len(self.trg_seeks) != len(source_references):
            raise SyntaxError("Target-side input and source-side reference have different length")
        if len(self.src_seeks) != len(target_references):
            raise SyntaxError("Source-side input and target-side reference have different length")
        self.source_references = [source_references]
        self.target_references = [target_references]
        self.source_token_references = [source_tok_references]
        self.target_token_references = [target_tok_references]

        self.epoch = epoch

    def find_seeks(self, source, target):
        source = self.fopen(source, 'r')
        target = self.fopen(target, 'r')

        src_seeks, trg_seeks = [], []
        prev_src_seek, prev_trg_seek = 0, 0
        while True:
            ss = source.readline()
            tt = target.readline()

            if ss == "" and tt == "":
                break

            ss = ss.strip().split()
            tt = tt.strip().split()

            src_seeks.append(prev_src_seek)
            trg_seeks.append(prev_trg_seek)

            prev_src_seek = source.tell()
            prev_trg_seek = target.tell()

        return src_seeks, trg_seeks

    def read_references(self, ref_source, ref_target):
        ref_source = self.fopen(ref_source, 'r')
        ref_target = self.fopen(ref_target, 'r')

        source_references, target_references = [], []
        while True:
            ss = ref_source.readline()
            tt = ref_target.readline()

            if ss == "" and tt == "":
                break

            source_references.append(ss.strip())
            target_references.append(tt.strip())

        return source_references, target_references

    def buffer(self, const_id):
        source = self.fopen(self.source, 'r')
        target = self.fopen(self.target, 'r')
        self.source_dataset = []
        self.target_dataset = []
        self.buffer_indexes = []

        ahead = self.ahead
        offset = self.offset
        src_seeks = self.src_seeks
        trg_seeks = self.trg_seeks

        tmp_src_seeks = src_seeks[offset:min(offset+ahead,len(src_seeks))]
        tmp_trg_seeks = trg_seeks[offset:min(offset+ahead,len(trg_seeks))]

        for src_seek, trg_seek in zip(tmp_src_seeks, tmp_trg_seeks):
            source.seek(src_seek)
            target.seek(trg_seek)

            ss = source.readline()
            tt = target.readline()

            if ss == "" or tt == "":
                raise SyntaxError("ss or tt read the end of the file while buffering the bilingual dataset")
            ss = ss.strip().split()
            tt = tt.strip().split()

            ss = [self.src_dict2.get(key, const_id.UNK) for key in ss]
            tt = [self.trg_dict2.get(key, const_id.UNK) for key in tt]

            self.source_dataset.append(ss)
            self.target_dataset.append(tt)
            self.buffer_indexes.append(self.offset)
            self.offset += 1

        # Buffer system read until the end of the file
        if offset+ahead >= len(self.src_seeks):
            self.offset = 0
            self.epoch += 1

    def __len__(self):
        return len(self.src_seeks)

    def __getitem__(self, idx):
        if idx not in self.buffer_indexes:
            self.buffer(Const)

        buffer_idx = self.buffer_indexes.index(idx)
        source = torch.tensor(self.source_dataset[buffer_idx])
        target = torch.tensor(self.target_dataset[buffer_idx])
        return source, target



def prepare_text_pair(batch):
    const_id = Const

    seqs_x = [x for (x, y) in batch]
    seqs_y = [y for (x, y) in batch]

    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS
    maxlen_y = np.max(lengths_y) + 2 # for BOS and EOS

    x_data = np.ones((n_samples, maxlen_x)).astype('int64')*const_id.PAD
    y_data = np.ones((n_samples, maxlen_y)).astype('int64')*const_id.PAD
    x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
    y_mask = np.zeros((n_samples, maxlen_y)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x_data[idx, 1:lengths_x[idx]+1] = s_x
        x_data[idx, 0] = const_id.BOS
        x_data[idx, lengths_x[idx]+1] = const_id.EOS
        x_mask[idx, :lengths_x[idx]+2] = 1.

        y_data[idx, 1:lengths_y[idx]+1] = s_y
        y_data[idx, 0] = const_id.BOS
        y_data[idx, lengths_y[idx]+1] = const_id.EOS
        y_mask[idx, :lengths_y[idx]+2] = 1. # extra +2 for BOS/EOS

    return x_data, x_mask, y_data, y_mask



def CallNormalIterator(data_dir, lang1_addr, lang2_addr, lang1_dict_addr, lang2_dict_addr,\
                     batch_size, maxlen, ahead, seed=0, rank=0, world_size=1, ctc_dict=False):

    # set dataset addresses
    lang1_addr = data_dir + lang1_addr
    lang2_addr = data_dir + lang2_addr

    addresses = [(lang1_addr, lang2_addr)]

    dist.barrier()

    # set vocab. dict addresses
    lang1_dict_addr = data_dir + lang1_dict_addr
    lang2_dict_addr = data_dir + lang2_dict_addr

    # set the batch sizes
    batch_sizes = [batch_size]

    # call and return MultiTextPairIterator
    iterator = MultiTextPairIterator(addresses, lang1_dict_addr, lang2_dict_addr,\
                batch_sizes, maxlen, ahead, seed, rank, world_size, ctc_dict)
    return iterator

def CallNormalIterator_TokenBased(data_dir, lang1_addr, lang2_addr, lang1_dict_addr, lang2_dict_addr,\
                     token_size, ahead, seed=0, rank=0, world_size=1, sorting=False, ctc_dict=False):

    # set dataset addresses
    lang1_addr = data_dir + lang1_addr
    lang2_addr = data_dir + lang2_addr

    addresses = [(lang1_addr, lang2_addr)]

    dist.barrier()

    # set vocab. dict addresses
    lang1_dict_addr = data_dir + lang1_dict_addr
    lang2_dict_addr = data_dir + lang2_dict_addr

    # set the token sizes
    token_sizes = [token_size]

    max_lengths = [150]

    # call and return MultiTextPairIterator
    iterator = MultiTextPairIterator_TokenBased(addresses, lang1_dict_addr, lang2_dict_addr,\
                token_sizes, ahead, max_lengths, seed, rank, world_size, sorting, ctc_dict)
    return iterator

def CallTestIterator_TokenBased(data_dir, lang1_addr, lang2_addr, lang1_dict_addr,\
                    lang2_dict_addr, token_size, rank=0, world_size=1, sorting=True,\
                    ctc_dict=False):
    # set dataset addresses
    lang1_addr = data_dir + lang1_addr
    lang2_addr = data_dir + lang2_addr

    address = (lang1_addr, lang2_addr)

    dist.barrier()

    # set vocab. dict addresses
    lang1_dict_addr = data_dir + lang1_dict_addr
    lang2_dict_addr = data_dir + lang2_dict_addr

    max_length = 248 # 250 - 2 for BOS & EOS

    # call and return MultiTextPairIterator
    iterator = TestPairIterator_TokenBased(address, lang1_dict_addr, lang2_dict_addr,\
                token_size, max_length, rank, world_size, sorting, ctc_dict)
    return iterator


def CallTestIterator_SacreBLEU(data_dir, data_lang1_addr, data_lang2_addr, \
                     ref_lang1_addr, ref_lang2_addr, \
                     token_ref_lang1_addr, token_ref_lang2_addr, \
                     lang1_dict_addr, lang2_dict_addr, batch_size, ctc_dict=False):
    # set dataset addresses
    data_lang1_addr = data_dir + data_lang1_addr
    data_lang2_addr = data_dir + data_lang2_addr
    ref_lang1_addr = data_dir + ref_lang1_addr
    ref_lang2_addr = data_dir + ref_lang2_addr
    token_ref_lang1_addr = data_dir + token_ref_lang1_addr
    token_ref_lang2_addr = data_dir + token_ref_lang2_addr

    # set vocab. dict addresses
    lang1_dict_addr = data_dir + lang1_dict_addr
    lang2_dict_addr = data_dir + lang2_dict_addr

    dataset = MyTestDataset(data_lang1_addr, data_lang2_addr, ref_lang1_addr, ref_lang2_addr, \
                        token_ref_lang1_addr, token_ref_lang2_addr,\
                        lang1_dict_addr, lang2_dict_addr, ahead=1000, ctc_dict=ctc_dict)

    # call and return MultiTextPairIterator
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=False,\
                        num_workers=1, collate_fn=prepare_text_pair,\
                        pin_memory=True)

    return iterator



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--port", type=int, default=24378)

    args = parser.parse_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    rank = args.local_rank

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(rank)

    data_dir='/home/milab_data/data/wmt14_en_de/'
    train_src_file='bpe.test.2014.en'
    train_trg_file='bpe.test.2014.de'
    mono_train_src_file='bpe.monolingual.dedup.small.en'
    mono_train_trg_file='bpe.monolingual.dedup.small.de'
    src_dict='vocab.32000.joined.pkl'
    trg_dict=src_dict

    dataset_seed=0

    batch_size=64
    max_length=250
    token_size=3600
    bimono_ratio=1
    ahead=1000
    sorting=True



    #test_iter = CallBTIterator(data_dir, train_src_file, train_trg_file,\
    #                        mono_train_src_file, mono_train_trg_file,\
    #                        src_dict, trg_dict, batch_size, (1,bimono_ratio),\
    #                        max_length, ahead, seed=dataset_seed,\
    #                        rank=rank, world_size=args.world_size)

    test_iter = CallBTIterator_TokenBased(data_dir, train_src_file, train_trg_file,\
                                        mono_train_src_file, mono_train_trg_file,\
                                        src_dict, trg_dict, token_size, (1,bimono_ratio),\
                                        ahead, seed=dataset_seed, rank=rank,\
                                        world_size=args.world_size, sorting=sorting)

    N_data = [len(test_iter.cursor_idxes[0]), len(test_iter.cursor_idxes[1]),\
              len(test_iter.cursor_idxes[2])]
    index_counts = [[0]*(N_data[0]+1), [0]*(N_data[1]+1), [0]*(N_data[2]+1)]
    max_samples = token_size

    start_time = time.time()
    times = [0, 0, 0]
    times_per_epoch = [0, 0, 0]

    batch_histories = [[], [], []]
    padding_ratio_histories = [[], [], []]

    if rank == 0: print("Iteration Starts")
    prev_epochs = copy.deepcopy(test_iter.epochs)
    print_every = 1000
    for iloop, [x_data, x_mask, y_data, y_mask, idxes] in enumerate(test_iter):
        for n in range(3):
            if prev_epochs[n] != test_iter.epochs[n]:
                #print("New Epoch !!!!!!!!!!!!!!!!!!!!!!!")
                times[n] = time.time() - start_time
                #print("times : ", times[n])
                times_per_epoch[n] = times[n] / test_iter.epochs[n] if test_iter.epochs[n] > 0 else 0
                #print("times per epoch : ", times_per_epoch[n])
                prev_epochs[n] = copy.deepcopy(test_iter.epochs[n])

                batch_histories[n] = []
                padding_ratio_histories[n] = []

        if rank == 0 and iloop % print_every == 0:
            print("=======================================")
            print("ILOOP|EPOCHS : {}|{}".format(iloop, test_iter.epochs))
            print("Times per epoch : {}".format(times_per_epoch))

        bi_x_data, bi_x_mask, bi_y_data, bi_y_mask = x_data[0], x_mask[0], y_data[0], y_mask[0]
        mono_x_data, mono_x_mask, mono_y_data, mono_y_mask = x_data[1], x_mask[1],\
                                                             y_data[2], y_mask[2]

        B_bi = len(bi_x_data)
        B_x_mono = len(mono_x_data)
        B_y_mono = len(mono_y_data)

        batch_histories[0].append(B_bi)
        batch_histories[1].append(B_x_mono)
        batch_histories[2].append(B_y_mono)

        if rank == 0 and iloop % print_every == 0:
            means = [np.mean(batch_histories[0]), np.mean(batch_histories[1]),\
                     np.mean(batch_histories[2])]
            stds = [np.std(batch_histories[0]), np.std(batch_histories[1]),\
                     np.std(batch_histories[2])]
            print("Batchsize statistics")
            print("--- Mean : {}".format(means))
            print("--- Std : {}".format(stds))

        N_bi = B_bi * len(bi_x_data[0])
        N_x_mono = B_x_mono * len(mono_x_data[0])
        N_y_mono = B_y_mono * len(mono_y_data[0])
        N_bi_pad = N_bi - sum([sum(tmp) for tmp in bi_x_mask])
        N_x_mono_pad = N_x_mono - sum([sum(tmp) for tmp in mono_x_mask])
        N_y_mono_pad = N_y_mono - sum([sum(tmp) for tmp in mono_y_mask])

        padding_ratio_histories[0].append(N_bi_pad/N_bi)
        padding_ratio_histories[1].append(N_x_mono_pad/N_x_mono)
        padding_ratio_histories[2].append(N_y_mono_pad/N_y_mono)

        if rank == 0 and iloop % print_every == 0:
            means = [np.mean(padding_ratio_histories[0]), np.mean(padding_ratio_histories[1]),\
                     np.mean(padding_ratio_histories[2])]
            stds = [np.std(padding_ratio_histories[0]), np.std(padding_ratio_histories[1]),\
                     np.std(padding_ratio_histories[2])]
            print("Padding Ratio statistics")
            print("--- Mean : {}".format(means))
            print("--- Std : {}".format(stds))

        bi_idxes, mono_x_idxes, mono_y_idxes = idxes[0], idxes[1], idxes[2]

        bi_idxes = torch.tensor(bi_idxes, dtype=torch.long).cuda()
        bi_idxes += 1
        bi_idxes = torch.cat((bi_idxes, torch.tensor([0]*(max_samples-len(bi_idxes))).cuda()))
        gathered_bi_idxes = [torch.zeros_like(bi_idxes, dtype=torch.long, device=bi_idxes.device) for _ in range(args.world_size)]

        if rank == 0:
            dist.gather(bi_idxes, gathered_bi_idxes)
        else:
            dist.gather(bi_idxes, dst=0)
        dist.barrier()

        mono_x_idxes = torch.tensor(mono_x_idxes, dtype=torch.long).cuda()
        mono_x_idxes += 1
        mono_x_idxes = torch.cat((mono_x_idxes, torch.tensor([0]*(max_samples-len(mono_x_idxes))).cuda()))
        gathered_mono_x_idxes = [torch.zeros_like(mono_x_idxes, dtype=torch.long, device=mono_x_idxes.device) for _ in range(args.world_size)]

        if rank == 0:
            dist.gather(mono_x_idxes, gathered_mono_x_idxes)
        else:
            dist.gather(mono_x_idxes, dst=0)
        dist.barrier()

        mono_y_idxes = torch.tensor(mono_y_idxes, dtype=torch.long).cuda()
        mono_y_idxes += 1
        mono_y_idxes = torch.cat((mono_y_idxes, torch.tensor([0]*(max_samples-len(mono_y_idxes))).cuda()))
        gathered_mono_y_idxes = [torch.zeros_like(mono_y_idxes, dtype=torch.long, device=mono_y_idxes.device) for _ in range(args.world_size)]

        if rank == 0:
            dist.gather(mono_y_idxes, gathered_mono_y_idxes)
        else:
            dist.gather(mono_y_idxes, dst=0)
        dist.barrier()

        if rank == 0:
            total_bi_idxes = []
            total_mono_x_idxes = []
            total_mono_y_idxes = []
            for n in range(len(gathered_bi_idxes)):
                total_bi_idxes += gathered_bi_idxes[n].tolist()
            for n in range(len(gathered_mono_x_idxes)):
                total_mono_x_idxes += gathered_mono_x_idxes[n].tolist()
            for n in range(len(gathered_mono_y_idxes)):
                total_mono_y_idxes += gathered_mono_y_idxes[n].tolist()

        # Verification for equal sampling ratio
        if rank == 0:
            for sampled_idx in total_bi_idxes:
                index_counts[0][sampled_idx] += 1
            for sampled_idx in total_mono_x_idxes:
                index_counts[1][sampled_idx] += 1
            for sampled_idx in total_mono_y_idxes:
                index_counts[2][sampled_idx] += 1
            for n in range(3):
                if max(index_counts[n][1:]) - min(index_counts[n][1:]) > 1:
                    print("Max : ", max(index_counts[n][1:]))
                    print("Min : ", min(index_counts[n][1:]))
                    raise SyntaxError("Non-equal sampling ratio in {} dataset".format(n))
        dist.barrier()
