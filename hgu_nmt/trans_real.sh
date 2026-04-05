LANG1='en'
LANG2='kr'

SAVE_DIR='./results'

VOC_NUM=10000

# AI Hub
#MODEL_PREFIX='en2kr.l6.d512.ff2048.aihub.gpu0s1.voc' # NMT-67
#MODEL_PREFIX='kr2en.l6.d512.ff2048.aihub.gpu1s2.voc' # NMT-68
# AI hub + HGU data
MODEL_PREFIX='en2kr.l6.d512.ff2048.aihub.gpu6s3.resume67' # 84
#MODEL_PREFIX='kr2en.l6.d512.ff2048.aihub.gpu7s4.resume68' # 85

MODEL_FILE=$SAVE_DIR'/'$MODEL_PREFIX'.best.pth'
ARGS_FILE=$SAVE_DIR'/'$MODEL_PREFIX'.args.pkl'

DATA_DIR=$HOME'/hgu_data/processed'
TEST_NAME='hgu_clean.test'

DICT1=$DATA_DIR'/aihub.vocab.'$LANG1'.tok.sym.'$VOC_NUM'sub.safe.P10.pkl'
DICT2=$DATA_DIR'/aihub.vocab.'$LANG2'.tok.sym.'$VOC_NUM'sub.safe.P10.pkl'

VALID_FILE1=$DATA_DIR'/'$TEST_NAME'.'$LANG1
VALID_FILE2=$DATA_DIR'/'$TEST_NAME'.'$LANG2
TRANS_FILE='./trans/'$MODEL_PREFIX'.trans' # output

# Preprocessing for the source language
TOOLS_DIR='./tools'
BPE_DIR='./subword_nmt'
PN_DICT=$DATA_DIR'/pn_dict_test.txt'
BPE_CODE=$DATA_DIR'/aihub.bpe.'$LANG1'.tok.sym.'$VOC_NUM'.code'

SRC_FILE=$DATA_DIR'/'$TEST_NAME'.'$LANG1
perl $TOOLS_DIR'/tokenizer.perl' -l 'en' < $SRC_FILE > $SRC_FILE'.tok'
SRC_FILE=$SRC_FILE'.tok'

python symb_test.py --pn_dict=$PN_DICT --input=$SRC_FILE --output=$SRC_FILE'.sym' --lang1=$LANG1
SRC_FILE=$SRC_FILE'.sym'

$BPE_DIR/apply_bpe.py -c $BPE_CODE < $SRC_FILE > $SRC_FILE'.'$VOC_NUM'sub'
SRC_FILE=$SRC_FILE'.'$VOC_NUM'sub'

python unbpe_symbols.py --input=$SRC_FILE --output=$SRC_FILE'.safe'

CUDA_VISIBLE_DEVICES=$1 python3 nmt_run.py --trans=1 \
        --trans_model_file=$MODEL_FILE --trans_args_file=$ARGS_FILE \
        --src_dict=$DICT1 --trg_dict=$DICT2 \
        --valid_src_file=$SRC_FILE \
        --trans_file=$TRANS_FILE --max_length=10000 

python nmt_post.py --input=$TRANS_FILE --output=$TRANS_FILE'.post' --lang1=$LANG1

perl $TOOLS_DIR'/multi-bleu.perl' $VALID_FILE2 <  $TRANS_FILE'.post'
