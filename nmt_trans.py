# H. Choi, hchoi@handong.edu

import argparse
import pickle as pkl

import torch

from nmt_trans_main import translate_file 
from text_data import read_dict
from nmt_model_tm import Transformer

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
# Files
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--valid_src_file", type=str, default='')
parser.add_argument("--valid_trg_file", type=str, default='')
parser.add_argument("--valid_post", type=str, default='')
parser.add_argument("--src_dict", type=str, default='')
parser.add_argument("--trg_dict", type=str, default='')
parser.add_argument("--max_length", type=int, default=125) #200
parser.add_argument("--bleu_script", type=str, default='./tools/multi-bleu.perl')

# Translation
parser.add_argument("--model", type=str, default='tm')
parser.add_argument("--trans_model_file", type=str, default='')
parser.add_argument("--trans_args_file", type=str, default='')
parser.add_argument("--trans_file", type=str, default='')
parser.add_argument("--beam_width", type=int, default=1)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

# Read dictionaries
src_dict = read_dict(args.src_dict)
trg_dict = read_dict(args.trg_dict)
old_args = pkl.load(open(args.trans_args_file, 'rb'))
print('old args', old_args)


# Model loading
print('model load: ' + args.trans_model_file)
chk_point = torch.load(args.trans_model_file)

model = Transformer(len(src_dict), len(trg_dict), args=old_args).to(device)
model.load_state_dict(chk_point['state_dict'], strict=False)

# Translating.
print ('Translating from ' + args.valid_src_file+args.valid_post + ' into ' + args.trans_file)
print('Target ' + args.valid_trg_file)
ret = translate_file(model, args.valid_src_file+args.valid_post, args.src_dict, args.valid_trg_file, args.trg_dict, args.trans_file, args, valid=False)
