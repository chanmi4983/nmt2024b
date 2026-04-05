# H. Choi, hchoi@handong.edu

import argparse
import pickle as pkl

from nmt_train_main import train_main

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
# Files
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--model", type=str, default='tm')
parser.add_argument("--model_file", type=str, default='')
parser.add_argument("--train_src_file", type=str, default='')
parser.add_argument("--train_trg_file", type=str, default='')
parser.add_argument("--valid_src_file", type=str, default='')
parser.add_argument("--valid_trg_file", type=str, default='')
parser.add_argument("--valid_post", type=str, default='')
parser.add_argument("--src_dict", type=str, default='')
parser.add_argument("--trg_dict", type=str, default='')
parser.add_argument("--xy", type=str, default='xy')
parser.add_argument("--param_shared", type=int, default=0)
parser.add_argument("--max_length", type=int, default=125) 
parser.add_argument("--bleu_script", type=str, default='./tools/multi-bleu.perl')

# training
parser.add_argument("--resume", type=int, default=0)
parser.add_argument("--resume_line", type=int, default=0)
parser.add_argument("--resume_file", type=str, default='')
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--grad_clip", type=float, default=10.0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--opt_start", type=int, default=4000)
parser.add_argument("--opt_scheduled", type=int, default=0)
parser.add_argument("--drop_p", type=float, default=0.1)
parser.add_argument("--label_smooth", type=float, default=0.0)
parser.add_argument("--emb_noise", type=float, default=0.0)
parser.add_argument("--reload", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--valid_start", type=int, default=5000)
parser.add_argument("--valid_interval", type=int, default=5000)

# Transformer architecture
parser.add_argument("--dim_wemb", type=int, default=512)
parser.add_argument("--dim_model", type=int, default=512)
parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--dim_ff", type=int, default=2048) 
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--dk", type=int, default=64) # 64
parser.add_argument("--dv", type=int, default=64)


# ETC
parser.add_argument("--logging", type=int, default=0)

args = parser.parse_args()

# Save args
print(args)
pkl.dump(args, open(args.save_dir+'/'+args.model_file+'.args.pkl', 'wb'), -1)
with open(args.save_dir+'/'+args.model_file+'.args', 'w') as fp:
    for key in vars(args):
        fp.write(key + ': ' + str(getattr(args, key)) + '\n')

# Training
print ('Training...')
train_main(args)
