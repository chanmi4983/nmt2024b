import argparse
import numpy as np

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--file", type=str, default='')
args = parser.parse_args()

print('results/'+args.file)
with open('results/' +args.file) as fp:
    idx_bleu = fp.readlines()

bleus = [float(line.split('\t')[1]) for line in idx_bleu]

print('max bleu', np.max(bleus))
