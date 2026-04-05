# H. Choi, hchoi@handong.edu

import argparse
import pickle as pkl

from text_data import read_dict

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
# Files
parser.add_argument("--src_dict", type=str, default='')
parser.add_argument("--trg_dict", type=str, default='')

args = parser.parse_args()

# Read dictionaries
src_dict = read_dict(args.src_dict)
trg_dict = read_dict(args.trg_dict)
print('args', args)
print('src', len(src_dict), 'trg', len(trg_dict))
