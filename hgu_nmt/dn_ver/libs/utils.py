# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import numpy as np
from collections import OrderedDict
import re
import shutil
import torch
import argparse
import ast
import neptune

# This function is borrowed from https://dodonam.tistory.com/185
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def check_korean(line):
    line_list = list(line)
    for ch in line_list:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', ch) is not None: 
            return True
    return False

def convert(line, no_jongsung=' '):
    # Unicode Korean Character: from 44032 to 55199
    KR_BEGIN, CHOSUNG, JUNGSUNG = 44032, 588, 28

    # Chosung list 00 ~ 18
    chosungs = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 
                    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    # Jungsung list 00 ~ 20
    jungsungs = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 
                    'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
                    'ㅡ', 'ㅢ', 'ㅣ']

    # Jongsung list 00 ~ 27 + 1(none)
    jongsungs = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 
                    'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
                    'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 
                    'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    line_list = list(line)

    result = list()
    for ch in line_list:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', ch) is not None: 
            char_code = ord(ch) - KR_BEGIN
            c_wmthar1 = int(char_code / CHOSUNG)
            result.append(chosungs[char1])

            char2 = int((char_code - (CHOSUNG*char1)) / JUNGSUNG)
            result.append(jungsungs[char2])

            char3 = int((char_code - (CHOSUNG*char1) - (JUNGSUNG*char2)))
            if char3==0:
                result.append(no_jongsung)
            else:
                result.append(jongsungs[char3])
        else:
            result.append(ch)
    # result
    output = "".join(result)
    return output

def read_pn_list(dict_file):
    pn_list = []
    with open(dict_file, 'r', encoding="utf-8") as dict_f:
        for line in dict_f:
            if line[0] == '#' or len(line.strip()) <= 0:
                continue
            if "::" not in line:
                continue
            en_w, kr_w = line.strip().split("::")
            if en_w == '' or kr_w =='':
                print('broken line:', line)
                continue
            pn_list.append((en_w, kr_w))
    return pn_list

def read_pn_dict(dict_file):
    pn_dict = OrderedDict()
    with open(dict_file, 'r', encoding="utf-8") as dict_f:
        for line in dict_f:
            if line[0] == '#' or len(line.strip()) <= 0:
                continue
            if "::" not in line:
                continue
            en_w, kr_w = line.strip().split("::")
            if en_w == '' or kr_w =='':
                print('broken line:', line)
                continue
            pn_dict[en_w] = kr_w

    return pn_dict

def symbolize_test(input_sen, PN_list, lang):
    input_sen = ' '+input_sen+' '
    temp_dict = OrderedDict()
    i = 0
    for (ew, kw) in PN_list: # PN_dict is (eng, kor)
        temp = "__P" + str(i)
        if lang == "k2e":
            if ' '+kw in input_sen:
                input_sen = input_sen.replace(' '+kw, ' '+temp)
                temp_dict[temp] = ew 
                i = i + 1
        else: 
            if ' '+ew+' ' in input_sen:
                input_sen = input_sen.replace(' '+ew+' ', ' '+temp+' ')
                temp_dict[temp] = kw
                i = i + 1
    return input_sen.strip(), temp_dict

def time_format(s):
    h = math.floor(s / 3600)
    m = math.floor((s-3600*h) / 60)
    s = s - h*3600 - m*60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (time_format(s))

def ids2words(dict_map_inv, raw_data, sep=' ', eos_id=0, unk_sym='<unk>'):
    str_text = ''
    for vv in raw_data:
        if vv == eos_id:
            break
        if vv in dict_map_inv:
            str_text = str_text + sep + dict_map_inv[vv]
        else:
            str_text = str_text + sep + unk_sym
    return str_text.strip()

def unbpe(sentence):
    #sentence = sentence.replace('<s>', '').strip()
    #sentence = sentence.replace('</s>', '').strip()
    sentence = sentence.replace('@@ ', '')
    sentence = sentence.replace('@@', '')
    return sentence

def PostProcess(sentence, PAD_TOKEN):
    sentence = sentence.replace('<s>', '').strip()
    sentence = sentence.replace('</s>', '').strip()
    sentence = sentence.replace(PAD_TOKEN+' ', '').strip()
    sentence = sentence.replace(PAD_TOKEN, '').strip()
    return sentence

def unCTCBlank(sentence, BLANK_TOKEN):
    sentence = sentence.replace(BLANK_TOKEN+' ', '').strip()
    sentence = sentence.replace(BLANK_TOKEN, '').strip()
    return sentence

def equizip(*iterables):
    iterators = [iter(x) for x in iterables]
    while True:
        try:
            first_value = iterators[0].__next__()
            try:
                other_values = [x.__next__() for x in iterators[1:]]
            except StopIteration:
                raise IterableLengthMismatch
            else:
                values = [first_value] + other_values
                yield tuple(values)
        except StopIteration:
            for iterator in iterators[1:]:
                try:
                    extra_value = iterator.__next__()
                except StopIteration:
                    pass # this is what we expect
                else:
                    raise IterableLengthMismatch
            raise StopIteration

def CountDown(sec):
    left_time = sec
    for i in range(sec):
        print("{} sec left..".format(left_time), end="\r")
        time.sleep(1)
        left_time -= 1

def DeleteFiles(*args):
    delete_flags = []
    for address in args:
        if os.path.exists(address):
            delete_flags.append(1)
        else:
            delete_flags.append(0)
    delete_list = [ address for i, address in enumerate(args) if delete_flags[i] == 1 ]
    if len(delete_list) > 0:
        print("Existed {} files will be deleted (5 sec later)".format(delete_list))
        CountDown(5)
        for address in delete_list:
            os.remove(address)


def make_subdir(*args, save_dir, resume, rank):
    subdir = ''
    for (arg, name) in args:
        subdir += arg if type(arg) is str else str(arg)
        subdir += name + '_'

    subdir_path = save_dir + subdir + '/'
    if resume == 0 and rank == 0:
        if not os.path.exists(subdir_path):
            print("{} directory is made".format(subdir_path))
            os.makedirs(subdir_path)
        else:
            print("(in make_subdir)WARNING:Exist {} directory will be deleted!".format(subdir_path))
            CountDown(5)
            shutil.rmtree(subdir_path, ignore_errors=False)
            os.makedirs(subdir_path)

    model_file = save_dir + subdir + '/model'
    return subdir, model_file

def save_checkpoint(save_dir, subdir, save_list, n_saved_checkpoints):
    subdir_path = save_dir + subdir + '/'
    model_file_list = [f for f in os.listdir(subdir_path)\
                         if f[-4:] == '.pth' and f[-9:] != '.best.pth']

    max_num = 0
    for file_name in model_file_list:
        model_num = int(file_name.strip('model').strip('.pth'))
        if model_num > max_num:
            max_num = model_num
    print("Max Checkpoint : ", max_num)

    if max_num > n_saved_checkpoints:
        for num in range(max_num-n_saved_checkpoints):
            tmp_model_path = subdir_path + 'model' + str(num) + '.pth'
            if os.path.exists(tmp_model_path):
                os.remove(tmp_model_path)

    new_num = max_num + 1
    torch.save(save_list, subdir_path + 'model' + str(new_num) + '.pth')

def load_checkpoint(load_dir, subdir, raw_model, raw_optimizer, opt_scheduled=True, load_latest=True, load_model_name=None):

    def load_model_optimizer(load_file_name, raw_model, raw_optimizer):
        if os.path.exists(load_file_name):
            load_chk_point = torch.load(load_file_name, map_location='cpu')

            for key, value in load_chk_point['state_dict'].copy().items():
                if 'pos_enc.pe' in key:
                    del load_chk_point['state_dict'][key]
            raw_model.load_state_dict(load_chk_point['state_dict'], strict=False)
            if opt_scheduled:
                raw_optimizer._optimizer.load_state_dict(load_chk_point['optimizer'])
                raw_optimizer._optimizer.param_groups[0]['capturable'] = True # This line forces the value to True, we don't know how it will affect the training
                raw_optimizer.n_steps = load_chk_point['scheduler']['n_steps']
            else:
                raw_optimizer.load_state_dict(load_chk_point['optimizer'])
                raw_optimizer.param_groups[0]['capturable'] = True
        else:
            raise SyntaxError("There is no pre-trained model named : {}".format(load_file_name))

        return raw_model, raw_optimizer, load_chk_point['iloop']

    if load_latest == False:
        if load_model_name is None:
            raise SyntaxError("load_model_name should be given if you are not going to load the latest checkpoint.")
        load_file_name = load_dir + subdir + '/' + load_model_name
    else:
        subdir_path = load_dir + subdir + '/'
        model_file_list = [f for f in os.listdir(subdir_path)\
                            if f[-4:] == '.pth' and f[-9:] != '.best.pth']
        max_num = 0
        for file_name in model_file_list:
            model_num = int(file_name.strip('model').strip('.pth'))
            if model_num > max_num:
                max_num = model_num
        load_file_name = subdir_path + 'model' + str(max_num) + '.pth'
        
    loaded_model, loaded_optimizer, resume_iloop = load_model_optimizer(load_file_name,\
                                                                     raw_model, raw_optimizer)

    return loaded_model, loaded_optimizer, resume_iloop



def load_metrics(project_name, running_id, metrics):
    raise SyntaxError("This function is deprecated according to neptuen.ai")
    load_project = neptune.new.init(project=project_name, run=running_id)

    for metric in metrics:
        metric_values = load_project['logs/'+metric]\
                            .fetch_values(include_timestamp=False).to_numpy()[:,1]
        for value in metric_values:
            neptune.log_metric(metric, value)

