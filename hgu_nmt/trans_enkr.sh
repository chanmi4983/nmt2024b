LANG1='en'
LANG2='kr'

SAVE_DIR='./results'

# AI Hub
#MODEL_PREFIX='en2kr.l6.d512.ff2048.aihub.gpu0s1.voc' # NMT-67
#MODEL_PREFIX='kr2en.l6.d512.ff2048.aihub.gpu1s2.voc' # NMT-68
# AI hub + HGU data
#MODEL_PREFIX='en2kr.l6.d512.ff2048.aihub.gpu6s3.resume67' # 84
#MODEL_PREFIX='kr2en.l6.d512.ff2048.aihub.gpu7s4.resume68' # 85

MODEL_PREFIX='en2krENKR1124_mid_Best.gpu2.demo'
#en2krPreLNdkEnt0316.gpu7.demo
MODEL_FILE=$SAVE_DIR'/'$MODEL_PREFIX'.best.pth' # should be '.pth.best' for NMT-44 or before.
ARGS_FILE=$SAVE_DIR'/'$MODEL_PREFIX'.args.pkl'
TRANS_FILE='./trans/'$MODEL_PREFIX'.trans' # output

DATA_DIR=$HOME'/hgu_data/processed' # should be 'processed3' for NMT-44 or before
SYMBOL='.sym' # '.sym'
VOC_NUM=10000
DICT1=$DATA_DIR'/aihub.vocab.'$LANG1'.tok'$SYMBOL'.'$VOC_NUM'sub.safe.P10.pkl'
DICT2=$DATA_DIR'/aihub.vocab.'$LANG2'.tok'$SYMBOL'.'$VOC_NUM'sub.safe.P10.pkl'

DATA_NAME='aihub'
# en2kr 19.08 (NMT-67), 20.33 (NMT-84-cur)
# kr2en 37.3 (NMT-68), 38.6 (NMT-85-cur)
#DATA_NAME='hgu_clean' # version 2
# en2kr: 3.89 (NMT-67), 10.52 (NMT-84-cur)
# kr2en: 17.03 (NMT-68), 26.32 (NMT-85-cur)

VALID_FILE1=$DATA_DIR'/'$DATA_NAME'.valid.'$LANG1'.tok'$SYMBOL
VALID_FILE2=$DATA_DIR'/'$DATA_NAME'.valid.'$LANG2'.tok'$SYMBOL
#VALID_FILE1=$DATA_DIR'/'$DATA_NAME'.test.'$LANG1'.tok'$SYMBOL
#VALID_FILE2=$DATA_DIR'/'$DATA_NAME'.test.'$LANG2'.tok'$SYMBOL
VALID_POST='.'$VOC_NUM'sub.safe' 

CUDA_VISIBLE_DEVICES=$1 python3 nmt_trans.py \
        --trans_model_file=$MODEL_FILE --trans_args_file=$ARGS_FILE \
        --src_dict=$DICT1 --trg_dict=$DICT2 \
        --valid_src_file=$VALID_FILE1 --valid_trg_file=$VALID_FILE2 --valid_post=$VALID_POST \
        --trans_file=$TRANS_FILE 

perl ./tools/multi-bleu.perl $VALID_FILE2 <  $TRANS_FILE
