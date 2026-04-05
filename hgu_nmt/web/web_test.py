#-*-coding: utf-8-*-

from flask import Flask, render_template, request, jsonify
import pickle as pkl
import sys
import argparse
import os
from text_data import read_dict
from libs.utils import ids2words, symbolize_test, read_pn_list, convert
from datetime import datetime
from io import open
import pickle
import numpy as np
from collections import OrderedDict
from flask_login import LoginManager, login_required
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--max_length", type=int, default=100) #Max length of input src
parser.add_argument("--kr_dict", type=str, default='') 
parser.add_argument("--en_dict", type=str, default='') 
parser.add_argument("--pn_dict", type=str, default='') 
parser.add_argument("--stop_words", type=str, default='') 
parser.add_argument("--kr_bpe_code", type=argparse.FileType('r', encoding="utf-8"), default='') 
parser.add_argument("--en_bpe_code", type=argparse.FileType('r', encoding="utf-8"), default='') 
parser.add_argument("--save_dir", type=str, default='')
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
app.secret_key = os.urandom(24)
login_manager = LoginManager()
login_manager.init_app(app)

class User:
    def __init__(self, user_id, email=None, passwd_hash=None,
                 authenticated=False):
        self.user_id = user_id
        self.email = email
        self.passwd_hash = passwd_hash
        self.authenticated = authenticated
    def __repr__(self):
        r = {
            'user_id': self.user_id,
            'email': self.email,
            'passwd_hash': self.passwd_hash,
            'authenticated': self.authenticated,
        }
        return str(r)
    def can_login(self, passwd_hash):
        return self.passwd_hash == passwd_hash
    def is_active(self):
        return True
    def get_id(self):
        return self.user_id
    def is_authenticated(self):
        return self.authenticated
    def is_anonymous(self):
        return False

USERS = {
    "user01": User("user01", passwd_hash='user_01'),
    "user02": User("user02", passwd_hash='user_02'),
    "user03": User("user03", passwd_hash='user_03'),
}
@login_manager.user_loader
def user_loader(user_id):
    return USERS[user_id]

class LoginForm(FlaskForm):
    username = StringField('Username')
    password = PasswordField('Password')
    submit = SubmitField('Submit')

@app.route('/login', methods=['POST'])
def login():
    form = LoginForm()
    return render_template('login.html', form=form)


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    user = current_user
    user.authenticated = False
    json_res = {'ok': True, 'msg': 'user <%s> logout' % user.user_id}
    logout_user()
    return jsonify(json_res)

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

    print('dict and stopwords are loaded')
    k2e_model = []
    e2k_model = []

    if not os.path.exists(args.temp_dir):
        os.mkdir(args.temp_dir)

    return k2e_model, e2k_model, kr_dict, kr_inv_dict, en_dict, en_inv_dict, PN_list, stop_words

@app.route("/")
def hello():
    return render_template("k2e.html")

def translate_one(lang):
    if lang=="k2e":
        html_file = "k2e.html"
        trans_model = k2e_model
        src_dict = kr_dict
        trg_inv_dict = en_inv_dict
        bpe_code = args.kr_bpe_code
    else:
        html_file = "e2k.html"
        trans_model = e2k_model
        src_dict = en_dict
        trg_inv_dict = kr_inv_dict
        bpe_code = args.en_bpe_code

    if request.method == 'GET':
        return render_template(html_file)

    input_sen = request.form['src'].replace("\", "").strip()
    if lang == "e2k":
        with open(args.temp_dir + "/stat_" + request.remote_addr+ "_in.txt", "a", encoding="utf8") as in_eng:
            in_eng.write(input_sen + "\n")

    return render_template(html_file)

@app.route('/k2e_trans', methods=['POST', 'GET'])
def k2e_trans(num=None):
    return translate_one("k2e")

@app.route('/e2k_trans', methods=['POST', 'GET'])
def e2k_trans(num=None):
    return translate_one("e2k")

@app.route('/add_dict', methods=['POST', 'GET'])
@login_required
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
@login_required
def edit_dict(num=None):
    user = request_loader(request)
    if user is None:
        return render_template("login")

    if request.method == 'GET':
        my_pn_file = args.temp_dir + '/pn_' + user.id + '.txt'
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

        my_pn_file = args.temp_dir + '/pn_' + user.id + '.txt'
        with open(my_pn_file, 'w') as f:
            f.write(word_list.strip()+"\n")

    return render_template("edit_dict.html", src_contents=word_list)

def count_words(lines):
    word_freq = OrderedDict()

    for line in lines:
        words = line.strip().replace("  ", " ").split(' ') 
        for word in words:
            word = word.strip().lower()
            if len(word) > 0 and (word not in stop_words) and word[0].isalnum():
                word_freq[word] = word_freq.get(word, 0) + 1

    word_sorted = OrderedDict(sorted(word_freq.items(), key=lambda t: t[1], reverse=True))

    return word_sorted
        
def count_freq_words(my_file, max_w):
    if os.path.exists(my_file):
        with open(my_file, 'r') as f:
            lines = f.readlines()
        word_freq = count_words(lines)    
        i = 0
        word_list = ''
        for (k,v) in word_freq.items():
            word_list = word_list + k + ': ' + str(v) + '\n'
            i = i + 1
            if i >= max_w: 
                break
    else: 
        word_list = "No words"

    return word_list

@app.route('/check_stat', methods=['POST', 'GET'])
@login_required
def check_stat(num=None):
    MAX_W = 100
    my_file = args.temp_dir + '/stat_' + request.remote_addr + '_in.txt'
    word_in = count_freq_words(my_file, MAX_W)

    my_file = args.temp_dir + '/stat_' + request.remote_addr + '_out.txt'
    word_out = count_freq_words(my_file, MAX_W)

    return render_template("check_stat.html", src_words=word_in, trg_words=word_out)

if __name__ == "__main__":

    k2e_model, e2k_model, kr_dict, kr_inv_dict, en_dict, en_inv_dict, PN_list, stop_words = setting(args)
    app.run(host=args.host_addr, port=args.port_num, threaded=True)
