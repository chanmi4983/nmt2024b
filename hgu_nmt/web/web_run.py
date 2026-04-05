#-*-coding: utf-8-*-

import sys
from flask import Flask, render_template, request, jsonify
import pickle as pkl
import argparse
import os
import torch
from text_data import read_dict
from nmt_trans_main import translate_tm
from nmt_model_tm import Transformer

import nmt_const as Const
from nmt_post import post_process
from datetime import datetime
from libs.utils import ids2words, symbolize_test, read_pn_list, convert
from subprocess import check_output
from io import open
from subword_nmt.apply_bpe import BPE
import numpy as np
from collections import OrderedDict

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--max_length", type=int, default=1000) #Max length of input src
parser.add_argument("--kr_dict", type=str, default='') 
parser.add_argument("--en_dict", type=str, default='') 
parser.add_argument("--pn_dict", type=str, default='') 
parser.add_argument("--stop_words", type=str, default='') 
parser.add_argument("--kr_bpe_code", type=argparse.FileType('r', encoding="utf-8"), default='') 
parser.add_argument("--en_bpe_code", type=argparse.FileType('r', encoding="utf-8"), default='') 
parser.add_argument("--temp_dir", type=str, default='./temp_dir')
parser.add_argument("--k2e_args", type=str, default='')
parser.add_argument("--e2k_args", type=str, default='')
parser.add_argument("--k2e_model_file", type=str, default='')
parser.add_argument("--e2k_model_file", type=str, default='')
parser.add_argument("--tokenizer", type=str, default='./tools/tokenizer.perl')
parser.add_argument("--detokenizer", type=str, default='./tools/detokenizer.perl')
parser.add_argument("--beam_width", type=int, default=1)
parser.add_argument("--host_addr", type=str, default='203.252.112.19')
parser.add_argument("--port_num", type=int, default=8888)
args = parser.parse_args()

# see https://www.fun-coding.org/flask_basic-2.html
# Usage : "python web_run.py" (see web.sh)
app = Flask(__name__)


def copyParams(module_src, module_dest):
    params_src = module_src.named_parameters()
    params_dest = module_dest.named_parameters()

    dict_dest = dict(params_dest)

    for name, param in params_src:
        if name in dict_dest:
            dict_dest[name].data.copy_(param.data)

def prepare_text(seqs_x, Const_id):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    n_samples = len(seqs_x)

    maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS

    x_data = np.ones((maxlen_x, n_samples)).astype('int64')*Const_id.PAD
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x_data[1:lengths_x[idx]+1, idx] = s_x
        x_data[0, idx] = Const_id.BOS
        x_data[lengths_x[idx]+1, idx] = Const_id.EOS
        x_mask[:lengths_x[idx]+2, idx] = 1.

    return x_data, x_mask


def setting(args):

    kr_dict = read_dict(args.kr_dict)
    en_dict = read_dict(args.en_dict)
    PN_list = read_pn_list(args.pn_dict)

    stop_words = []
    for line in open(args.stop_words):
        stop_words.append(line.strip())

    en_inv_dict = dict()
    for kk, vv in en_dict.items():
        en_inv_dict[vv] = kk

    kr_inv_dict = dict()
    for kk, vv in kr_dict.items():
        kr_inv_dict[vv] = kk

    # BPE
    kr_bpe = BPE(args.kr_bpe_code)
    en_bpe = BPE(args.en_bpe_code)
    print('dict and stopwords are loaded')

    torch.no_grad()
    torch.nn.Module.dump_patches = True
    k2e_args = pkl.load(open(args.k2e_args, 'rb'))
    e2k_args = pkl.load(open(args.e2k_args, 'rb'))

    kr_words_n = len(kr_dict)
    en_words_n = len(en_dict)

    k2e_model = Transformer(kr_words_n, en_words_n, args=k2e_args).to(device) 
    e2k_model = Transformer(en_words_n, kr_words_n, args=e2k_args).to(device) 

    chk_k2e = torch.load(args.k2e_model_file)
    chk_e2k = torch.load(args.e2k_model_file)
    k2e_model.load_state_dict(chk_k2e['state_dict'], strict=False)
    e2k_model.load_state_dict(chk_e2k['state_dict'], strict=False)

    k2e_model.eval()
    e2k_model.eval()

    print("models are loaded")

    if not os.path.exists(args.temp_dir):
        os.mkdir(args.temp_dir)

    return k2e_model, e2k_model, kr_bpe, en_bpe, kr_dict, kr_inv_dict, en_dict, en_inv_dict, PN_list, stop_words

@app.route("/")
def hello():
    return render_template("k2e.html", lang='k2e')

def add_newline_char(string, ch):
    string = string.replace(" " + ch + " &quot;", " " + ch + "&quot;") # ' . &quot;' --> ' .&quot;'
    string = string.replace(" " + ch + " ”", " " + ch + "”")

    string = string.replace(" " + ch + " ", " " + ch + " \n") # add new line

    string = string.replace(" " + ch + "&quot;", " " + ch + " &quot;") # ' .&quot;' --> ' . &quot;'
    string = string.replace(" " + ch + "”", " " + ch + " ”")

    string = string.replace(" " + ch + " &quot; ", " " + ch + " &quot;\n") # add new line
    string = string.replace(" " + ch + " ” ", " " + ch + " ”\n")
    return string

def add_newlines(string):
    new_str = string.strip()
    ch_list=['.', '?', '!']
    for ch in ch_list:
        new_str = add_newline_char(new_str, ch)
    return new_str

def translate_one(lang):
    if lang=="e2k":
        html_file = "e2k.html"
        trans_model = e2k_model
        src_dict = en_dict
        trg_inv_dict = kr_inv_dict
        bpe = en_bpe
    else: # lang=="k2e":
        html_file = "k2e.html"
        trans_model = k2e_model
        src_dict = kr_dict
        trg_inv_dict = en_inv_dict
        bpe = kr_bpe

    if request.method != 'POST':
        return render_template(html_file, lang=lang)

    if request.form['src'] == "":
        return render_template(html_file)

    # reading pn dictionary
    pn_with_mine = []
    my_pn_file = args.temp_dir + '/pn_' + request.remote_addr + '.txt'
    if os.path.exists(my_pn_file):
        pn_with_mine = read_pn_list(my_pn_file)
    pn_with_mine.extend(PN_list)

    log_file = open(args.temp_dir + "/nmt.log", "a", encoding="utf8")
    input_sen = request.form['src'].strip()
    print("Input: " + input_sen)

    date_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    log_file.write("IP & Date: " + request.remote_addr + ': '+ date_str + '[' + lang + "]\n")
    log_file.write("Input: " + input_sen + "\n")

    # Tokenize TODO: use a python function tokenizer
    input_file = args.temp_dir + '/input' + request.remote_addr + '.txt'
    with open(input_file, "w", encoding="utf8") as in_f:
        in_f.write(input_sen)
    token_cmd='perl ' + args.tokenizer + ' en < ' + input_file
    token_sen=check_output(token_cmd, shell=True, encoding="utf-8").strip()

    # Symbolize
    symbol_sen, temp_dict = symbolize_test(token_sen, pn_with_mine, lang)
    log_file.write("Symbolized: " + symbol_sen + "\n")

    # Translation
    ss = []
    new_line_num = []
    symbol_sen = add_newlines(symbol_sen)
    for line in symbol_sen.split('\n'):
        if len(line.strip().split())<=0: # new line
            new_line_num[-1] = new_line_num[-1] + 1 
            continue

        new_line_num.append(0)
        bpe_sen = bpe.segment(line).strip().replace("__P@@ ", "__P") 
        #print("BPEd: " + bpe_sen)
        ss.append([src_dict.get(key, Const.UNK) for key in bpe_sen.strip().split()])

    x_data, x_mask = prepare_text(ss, Const)

    samples = translate_tm(trans_model, x_data, x_mask, args.max_length)
    out_result = ''
    out_before_detoken_cat = ''
    for i in range(samples.shape[0]):
        output = ids2words(trg_inv_dict, samples[i,:], eos_id=Const.EOS) # from id to word
        #print("Translated: " + output)
        log_file.write("Translated: " + output + "\n")

        output, out_before_detoken = post_process(output, lang, temp_dict, args)
        out_before_detoken_cat = out_before_detoken_cat + ' ' + out_before_detoken
        if i == 0:
            out_result = output
        else:
            if new_line_num[i-1] == 0:
                out_result = out_result + ' ' + output
            elif new_line_num[i-1] == 1:
                out_result = out_result + '\n\n' + output
            else:
                out_result = out_result + '\n\n\n' + output

    #print("Output: " + out_result)
    log_file.write("Output: " + out_result + "\n")
    log_file.close()

    write_for_stat(token_sen, out_before_detoken_cat, lang, args, request)

    return render_template(html_file, lang=lang, src_contents=input_sen, trans_contents=out_result)

def write_for_stat(token_sen, out_before_detoken, lang, args, request):
    try:
        if lang == "e2k":
            with open(args.temp_dir + "/stat_" + request.remote_addr+ "_in.txt", "a", encoding="utf8") as in_eng:
                in_eng.write(token_sen + "\n")
        else:
            with open(args.temp_dir + "/stat_" + request.remote_addr+ "_out.txt", "a", encoding="utf8") as out_eng:
                out_eng.write(out_before_detoken + "\n")
    except:
        print('statistics file error')

@app.route('/lang_switch', methods=['POST', 'GET'])
def lang_switch(num=None):
    lang = request.form['lang']
    if lang=="e2k":
        html_file = "k2e.html"
        new_lang='k2e'
    else: # lang=="k2e":
        html_file = "e2k.html"
        new_lang='e2k'

    input_sen = request.form['src']
    output_sen = request.form['trg']
    return render_template(html_file, lang=new_lang, src_contents=output_sen, trans_contents=input_sen)

@app.route('/k2e_trans', methods=['POST', 'GET'])
def k2e_trans(num=None):
    return translate_one("k2e")

@app.route('/e2k_trans', methods=['POST', 'GET'])
def e2k_trans(num=None):
    return translate_one("e2k")

@app.route('/add_dict', methods=['POST', 'GET'])
def add_dict(num=None):
    if request.method == 'GET':
        return render_template("add_dict.html")
    if request.method == 'POST':
        if request.form['kr_word'] == "":
            return render_template("add_dict.html")
        if request.form['en_word'] == "":
            return render_template("add_dict.html")
        kr_word = request.form['kr_word']
        en_word = request.form['en_word']

        my_pn_file = args.temp_dir + '/pn_' + request.remote_addr + '.txt'
        with open(my_pn_file, 'a') as f:
            f.write(en_word + "::" + kr_word+"\n")

    return render_template("add_dict.html")

@app.route('/edit_dict', methods=['POST', 'GET'])
def edit_dict(num=None):
    if request.method == 'GET':
        my_pn_file = args.temp_dir + '/pn_' + request.remote_addr + '.txt'
        if os.path.exists(my_pn_file):
            with open(my_pn_file, 'r', encoding="utf-8") as f:
                word_list=f.readlines()
            word_list = ''.join(word_list)
            return render_template("edit_dict.html", src_contents=word_list)
        else:
            return render_template("edit_dict.html")

    if request.method == 'POST':
        if request.form['word_list'] == "":
            return render_template("edit_dict.html")
        word_list = request.form['word_list']

        my_pn_file = args.temp_dir + '/pn_' + request.remote_addr + '.txt'
        with open(my_pn_file, 'w') as f:
            f.write(word_list.strip()+"\n")

    return render_template("edit_dict.html", src_contents=word_list)

def count_words(lines):
    word_freq = OrderedDict()

    total_cnt = 0
    for line in lines:
        words = line.strip().replace("  ", " ").split(' ') 
        for word in words:
            word = word.strip().lower()
            if len(word) > 0 and word[0].isalnum():
                total_cnt = total_cnt + 1
                if word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1

    word_sorted = OrderedDict(sorted(word_freq.items(), key=lambda t: t[1], reverse=True))

    return word_sorted, total_cnt
        
def count_freq_words(my_file, max_w):
    if os.path.exists(my_file):
        with open(my_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        word_freq, total = count_words(lines)
        i = 0
        word_list = ''
        for (k,v) in word_freq.items():
            word_list = word_list + k + ': ' + str(v) + '\n'
            i = i + 1
            if i >= max_w:
                break
    else:
        word_list = " "
        total = 0

    word_list = 'Total Count: ' + str(total) + '\n' + word_list
    return word_list

@app.route('/check_stat', methods=['POST', 'GET'])
def check_stat(num=None):
    MAX_W = 100
    my_file_in = args.temp_dir + '/stat_' + request.remote_addr + '_in.txt'
    my_file_out = args.temp_dir + '/stat_' + request.remote_addr + '_out.txt'

    if request.method == 'GET':
        word_in = count_freq_words(my_file_in, MAX_W)
        word_out = count_freq_words(my_file_out, MAX_W)
        return render_template("check_stat.html", src_words=word_in, trg_words=word_out)
    else:
        if os.path.exists(my_file_in): os.remove(my_file_in)
        if os.path.exists(my_file_out): os.remove(my_file_out)
        return render_template("check_stat.html", src_words='Total Count: 0', trg_words='Total Count: 0')

if __name__ == "__main__":
    k2e_model, e2k_model, kr_bpe, en_bpe, kr_dict, kr_inv_dict, en_dict, en_inv_dict, PN_list, stop_words = setting(args)
    app.run(host=args.host_addr, port=args.port_num, threaded=True)
