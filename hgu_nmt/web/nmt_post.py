import pickle as pkl
import sys
import argparse
import os
import torch
import torch.nn as nn
from text_data import read_dict
from datetime import datetime
from libs.utils import ids2words, unbpe, unbpe_fix, symbolize_test, read_pn_list, convert
from subprocess import Popen, PIPE, check_output, call
from io import open
import pickle
import nmt_const as Const
import numpy as np
from collections import OrderedDict

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def post_process(output, lang, temp_dict, args):
    # unBPE
    output = unbpe(output)

    # #숫자 기호화 되돌리기
    # mapping=open("mapping.sym","rb")
    # num_dict=pickle.load(mapping)

    # Desymbolization
    for key, val in temp_dict.items(): 
        if lang=='k2e':
            output = output.replace(key, val)
        else:
            # cases with '을/를' '은/는' '이/가'
            # if key is followed be postfix, 
            # and val has jongsung and the postfix is one of '를/는/가'
            jongsung = convert(val[-1], no_jongsung='-')
            if jongsung[-1] == '-' or val[-1] in "aefhiostuvwxyzABCDEFGHIJKOPQSTUVWXYZ": 
                # when there is no jongsung
                output = output.replace(key+'이 ', key+'가 ' )
                output = output.replace(key+'은 ', key+'는 ' )
                output = output.replace(key+'을 ', key+'를 ' )
            else:
                output = output.replace(key+'가 ', key+'이 ' )
                output = output.replace(key+'는 ', key+'은 ' )
                output = output.replace(key+'를 ', key+'을 ' )
            output = output.replace(key, val)

    out_before_detoken = output.replace("  ", " ").strip()

    # Detokenized
    if lang == "k2e": #TODO: use a python detokenizer
        with open(args.temp_dir + "/temp.txt", "w", encoding="utf8") as out_f:
            out_f.write(output)
        detoken_cmd = 'perl ' + args.detokenizer + ' en < ' + args.temp_dir + '/temp.txt'
        output=check_output(detoken_cmd, shell=True, encoding="utf-8")
    else: # TODO, use word divider for Korean
        output = output.replace(" ,", ",") 
        output = output.replace(" .", ".") 
        output = output.replace(" !", "!") 
        output = output.replace(" ?", "?") 

    output = output.replace("<unk>", "").replace("  ", " ").strip()

    return output, out_before_detoken

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=str, default='') 
    parser.add_argument("--output", type=str, default='') 
    parser.add_argument("--lang1", type=str, default='') 
    parser.add_argument("--temp_dir", type=str, default='./temp_dir') 
    parser.add_argument("--detokenizer", type=str, default='./tools/detokenizer.perl')
    args = parser.parse_args()

    if args.lang1 == "kr":
        lang = "k2e"
    else:
        lang = "e2k"

    dict_list = pkl.load(open(args.temp_dir+'/test_temp_dicts.pkl', 'rb'))
    fout = open(args.output, "w")
    with open(args.input, "r") as fp:  
        i = 0
        for line in fp:
            output, out_token = post_process(line.strip(), lang, dict_list[i], args)
            fout.write(output + "\n") 
            i = i + 1

    fout.close()

    print('post process done')
