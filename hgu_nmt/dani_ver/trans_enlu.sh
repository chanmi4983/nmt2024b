LANG1='en'
LANG2='lu'

SAVE_DIR='./results'

# AI Hub
#MODEL_PREFIX='en2kr.l6.d512.ff2048.aihub.gpu0s1.voc' # NMT-67
#MODEL_PREFIX='kr2en.l6.d512.ff2048.aihub.gpu1s2.voc' # NMT-68
# AI hub + HGU data
#MODEL_PREFIX='en2kr.l6.d512.ff2048.aihub.gpu6s3.resume67' # 84
#MODEL_PREFIX='kr2en.l6.d512.ff2048.aihub.gpu7s4.resume68' # 85
#AI hub + EngLug data
MODEL_PREFIX='en2lu.l3.d256.ff2048.gpu.modpostnorm_750' #'en2lu.l3.d256.ff2048.gpu.prenorm_750' #'en2lu.l6_500' #'en2lu.l6.d256.ff2048.gpu.baseline'  #'en2lu.l6.d256.ff2048.gpu'  ##'en2lu.l3.d256.ff1024.gpu' # for Wandb run
#MODEL_PREFIX='en2lu.l6.d512.ff2048.gpu' # for normal runs, d and ff can change

MODEL_FILE=$SAVE_DIR'/'$MODEL_PREFIX'.best.pth' # should be '.pth.best' for NMT-44 or before.
ARGS_FILE=$SAVE_DIR'/'$MODEL_PREFIX'.args.pkl'
TRANS_FILE='./trans/'$MODEL_PREFIX'.trans_ins_postnorm' # output

DATA_DIR='../dataset' # should be 'processed3' for NMT-44 or before
VOC_NUM=10000
DICT1=$DATA_DIR'/raw.vocab.'$LANG1'.tok.'$VOC_NUM'sub.safe.P10.pkl'
DICT2=$DATA_DIR'/raw.vocab.'$LANG2'.tok.'$VOC_NUM'sub.safe.P10.pkl'

DATA_NAME='raw'
# en2kr 19.08 (NMT-67), 20.33 (NMT-84-cur)
# kr2en 37.3 (NMT-68), 38.6 (NMT-85-cur)
#DATA_NAME='hgu_clean' # version 2
# en2kr: 3.89 (NMT-67), 10.52 (NMT-84-cur)
# kr2en: 17.03 (NMT-68), 26.32 (NMT-85-cur)

#These are test files
VALID_FILE1=$DATA_DIR'/'$DATA_NAME'.test.'$LANG1'.tok.'$VOC_NUM'sub.safe.insert'
VALID_FILE2=$DATA_DIR'/'$DATA_NAME'.test.'$LANG2'.tok'

#COMMENT BY DANI: don't change CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=1 python3.8 nmt_run.py --trans=1 \
        --trans_model_file=$MODEL_FILE --trans_args_file=$ARGS_FILE \
        --src_dict=$DICT1 --trg_dict=$DICT2 \
        --valid_src_file=$VALID_FILE1 --valid_trg_file=$VALID_FILE2 \
        --trans_file=$TRANS_FILE --max_length=10000

perl ./tools/multi-bleu.perl $VALID_FILE2 <  $TRANS_FILE

