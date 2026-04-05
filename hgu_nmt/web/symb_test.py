import pickle as pkl
import sys
import argparse
import os
from text_data import read_dict
from datetime import datetime
from libs.utils import ids2words, symbolize_test, read_pn_list, convert
from io import open
import nmt_const as Const
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=str, default='') 
    parser.add_argument("--output", type=str, default='') 
    parser.add_argument("--pn_dict", type=str, default='') 
    parser.add_argument("--lang1", type=str, default='') 
    parser.add_argument("--temp_dir", type=str, default='./temp_dir') 
    parser.add_argument("--detokenizer", type=str, default='./tools/detokenizer.perl')
    args = parser.parse_args()

    if args.lang1 == "kr":
        lang = "k2e"
    else:
        lang = "e2k"

    pn_dict = read_pn_list(args.pn_dict)

    dict_list = []
    fout = open(args.output, "w")
    with open(args.input, "r") as fp:  
        for line in fp:
            symbol_sen, temp_dict = symbolize_test(line.strip(), pn_dict, lang)
            dict_list.append(temp_dict) 
            fout.write(symbol_sen + "\n") 
    fout.close()

    pkl.dump(dict_list, open(args.temp_dir+'/test_temp_dicts.pkl', 'wb'), -1)
