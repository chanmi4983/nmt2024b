# H. Choi, hchoi@handong.edu

import argparse
import pickle as pkl
import os
import torch
import nmt_const as Const


import wandb

from text_data import TextPairIterator, read_dict
from nmt_main import train_main
from nmt_trans import translate_file 
from text_data import read_dict
##from nmt_model_tmqv import Transformer
#from nmt_model_tm import Transformer
from nmt_model_scale import Transformer
#from nmt_lightning import LitAutoEncoder


parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
# Files
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--model", type=str, default='tm')
parser.add_argument("--model_file", type=str, default='')
parser.add_argument("--train_src_file", type=str, default='')
parser.add_argument("--train_trg_file", type=str, default='')
parser.add_argument("--valid_src_file", type=str, default='')
parser.add_argument("--valid_trg_file", type=str, default='')
parser.add_argument("--src_dict", type=str, default='')
parser.add_argument("--trg_dict", type=str, default='')
parser.add_argument("--max_length", type=int, default=125)
parser.add_argument("--bleu_script", type=str, default='./tools/multi-bleu.perl')

# training
parser.add_argument("--train", type=int, default=0)
parser.add_argument("--resume", type=int, default=0)
parser.add_argument("--resume_line", type=int, default=0)
parser.add_argument("--resume_file", type=str, default='')
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--grad_clip", type=float, default=10.0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--opt_start", type=int, default=4000)
parser.add_argument("--opt_scheduled", type=int, default=0)
parser.add_argument("--dropout_p", type=float, default=0.1)
parser.add_argument("--emb_noise", type=float, default=0.1)
parser.add_argument("--reload", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--print_every", type=int, default=500)
parser.add_argument("--valid_start", type=int, default=5000)
parser.add_argument("--valid_every", type=int, default=5000)
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--thresh", type=int, default=0.01)
parser.add_argument("--ahead", type=int, default=100)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--norm_case", type=str, default='postnorm') #default='prenorm') #postnorm, ours

# Model architecture
parser.add_argument("--dim_wemb", type=int, default=512)
parser.add_argument("--dim_model", type=int, default=512)
# RNN 
parser.add_argument("--rnn_name", type=str, default='lstm')
parser.add_argument("--dim_enc", type=int, default=0)
parser.add_argument("--dim_att", type=int, default=0)
parser.add_argument("--pos_enc", type=int, default=1)
# Transformer 
parser.add_argument("--tm_n_layers", type=int, default=3)
parser.add_argument("--tm_dim_ff", type=int, default=2048) 
parser.add_argument("--tm_n_head", type=int, default=8)
parser.add_argument("--tm_dk", type=int, default=64)
parser.add_argument("--tm_dv", type=int, default=64)
parser.add_argument("--eps", type=float, default=1)

# Translation
parser.add_argument("--trans_model_file", type=str, default='')
parser.add_argument("--trans_args_file", type=str, default='')
parser.add_argument("--trans", type=int, default=0)
parser.add_argument("--trans_file", type=str, default='')
parser.add_argument("--beam_width", type=int, default=1)

# ETC
parser.add_argument("--wandb", type=int, default=0)

args = parser.parse_args()

#wandb code
if args.wandb!=0:
    wandb.init(project="en2lu",entity="danastalyn")
    wandb.config = vars(args)

# training
if args.train:
    print(args)
    pkl.dump(args, open(args.save_dir+'/'+args.model_file+'_'+str(args.ahead)+'.args.pkl', 'wb'), -1)
    with open(args.save_dir+'/'+args.model_file+'_'+str(args.ahead)+'.args', 'w') as fp:
        for key in vars(args):
            fp.write(key + ': ' + str(getattr(args, key)) + '\n')
    
    src_dict = read_dict(args.src_dict)
    trg_dict = read_dict(args.trg_dict)
    args.src_words_n = len(src_dict) 
    args.trg_words_n = len(trg_dict) 
    
    print ('Training...')
    train_main(args)    

if args.trans:
    use_cuda = torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")

    src_dict = read_dict(args.src_dict)
    trg_dict = read_dict(args.trg_dict)
    old_args = pkl.load(open(args.trans_args_file, 'rb'))
    old_args.src_words_n = len(src_dict) 
    old_args.trg_words_n = len(trg_dict) 

    print('model load: ' + args.trans_model_file)
    model = Transformer(args=old_args).to(device)
    chk_point = torch.load(args.trans_model_file)
    model.load_state_dict(chk_point['state_dict'], strict=False)

    old_args.trans_file = args.trans_file
    old_args.valid_src_file= args.valid_src_file
    old_args.valid_trg_file= args.valid_trg_file
    print(old_args)

    print ('Translating...')
    ret = translate_file(model, old_args, valid=False)
