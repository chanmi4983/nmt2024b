LANG1='en'
LANG2='kr'

SAVE_DIR='./results'

ARGS_FILE=$SAVE_DIR'/'$MODEL_PREFIX'.args.pkl'

DATA_DIR=$HOME'/hgu_data/processed' # should be 'processed3' for NMT-44 or before
VOC_NUM=10000
DICT1=$DATA_DIR'/aihub.vocab.'$LANG1'.tok.sym.'$VOC_NUM'sub.safe.P10.pkl'
DICT2=$DATA_DIR'/aihub.vocab.'$LANG2'.tok.sym.'$VOC_NUM'sub.safe.P10.pkl'

CUDA_VISIBLE_DEVICES=$1 python3 check_dict.py --src_dict=$DICT1 --trg_dict=$DICT2 

DICT1=$DATA_DIR'/aihub.vocab.'$LANG1'.tok.sym.'$VOC_NUM'sub.safe.fix.P10.pkl'
DICT2=$DATA_DIR'/aihub.vocab.'$LANG2'.tok.sym.'$VOC_NUM'sub.safe.fix.P10.pkl'

CUDA_VISIBLE_DEVICES=$1 python3 check_dict.py --src_dict=$DICT1 --trg_dict=$DICT2 

