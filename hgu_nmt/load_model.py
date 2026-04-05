import argparse
import pickle as pkl

import torch

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--model_file", type=str, default='')
args = parser.parse_args()
print(args)

file_name = args.save_dir + '/' + args.model_file + '.best.pth'
print(file_name)
model = torch.load(file_name)
#print(model)
src = model['state_dict']['encoder.src_emb.weight'])
trg = model['state_dict']['decoder.trg_emb.weight'])

old_args = pkl.load(open(args.save_dir+'/'+args.model_file+'.args.pkl', 'rb'))
print(old_args)
