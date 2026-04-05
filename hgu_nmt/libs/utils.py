# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import numpy as np
from collections import OrderedDict
import re


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
            char1 = int(char_code / CHOSUNG)
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

def unbpe_fix(sentence):
    sentence = sentence.replace('<s>', '').strip()
    sentence = sentence.replace('</s>', '').strip()
    sentence = sentence.replace(' @@', '')
    sentence = sentence.replace('@@', '')
    return sentence

def unbpe(sentence):
    sentence = sentence.replace('<s>', '').strip()
    sentence = sentence.replace('</s>', '').strip()
    sentence = sentence.replace('@@ ', '')
    sentence = sentence.replace('@@', '')
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
