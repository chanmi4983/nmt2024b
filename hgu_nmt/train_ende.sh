LANG1='en'
LANG2='de'

PARAM_SHARED=0

DATA_DIR=$HOME'/wmt18_en_de'

LR=0.0005

LABEL_SMOOTH=0.1 # 0.01에서 0.1로 수정 (0416)
EMB_NOISE=0.00 # 0.01

DIM_MODEL=512 #512
DIM_FF=2048 #2048
N_LAYERS=6
N_HEAD=8
DK=64 # dk
DV=64

BATCH_SIZE=32 #64
UPDATE_STEP=2 # 1

DIM_WEMB=$DIM_MODEL # should be the same

#SYMBOL='.sym' # '.sym'
VOC_NUM=32000

# New data
# DICT1=$DATA_DIR'/aihub.vocab.'$LANG1'.tok'$SYMBOL'.'$VOC_NUM'sub.safe.P10.pkl'
# DICT2=$DATA_DIR'/aihub.vocab.'$LANG2'.tok'$SYMBOL'.'$VOC_NUM'sub.safe.P10.pkl'
DICT1=$DATA_DIR'/vocab.32000.joined.pkl'
DICT2=$DATA_DIR'/vocab.32000.joined.pkl'

# AI Hub
TRAIN_FILE1=$DATA_DIR'/bpe.train.'$LANG1
TRAIN_FILE2=$DATA_DIR'/bpe.train.'$LANG2
# AI Hub + HGU with AI Hub code and voca
#TRAIN_FILE1=$DATA_DIR'/aihub_hgu.train.'$LANG1'.tok'$SYMBOL'.'$VOC_NUM'sub.safe.shuf'
#TRAIN_FILE2=$DATA_DIR'/aihub_hgu.train.'$LANG2'.tok'$SYMBOL'.'$VOC_NUM'sub.safe.shuf'
# Dict
#TRAIN_FILE1=$DATA_DIR'/dict.'$LANG1'.tok.'$VOC_NUM'sub'
#TRAIN_FILE2=$DATA_DIR'/dict.'$LANG2'.tok.'$VOC_NUM'sub'

VALID_FILE1=$DATA_DIR'/bpe.test.2016.'$LANG1
VALID_FILE2=$DATA_DIR'/bpe.test.2016.'$LANG2
VALID_POST=''

SAVE_DIR='./results'
if [ ! -d $SAVE_DIR ]; then
    mkdir $SAVE_DIR
fi
if [ ! -d './trans' ]; then
    mkdir 'trans'
fi

MODEL_FILE=$LANG1'2'$LANG2'ENDE1101_PreDkOParaX.gpu'$1'.demo'
# 모델이름 바꾸기

RESUME=0
OPT_SCHEDULED=0
RESUME_LINE=0 # The last iteration * BATCH_SIZE when RESUME=1
#RESUME_FILE='en2kr.l6.d512.ff2048.aihub.gpu1s4.pth.best.pth' # NMT-28
#RESUME_FILE='kr2en.l6.d512.ff2048.aihub.gpu4s9.pth.best.pth' # NMT-29
#RESUME_FILE='en2kr.l6.d512.ff2048.aihub.gpu0s1.voc.pth' # NMT-67
#RESUME_FILE='kr2en.l6.d512.ff2048.aihub.gpu1s2.voc.pth' # NMT-68

# for resume, opt_schedule=0, lr=0.0001, valid_start=100000 valid_every=5000
CUDA_VISIBLE_DEVICES=$1,CUDA_LAUNCH_BLOCKING=1 python3 -W ignore nmt_train.py \
        --save_dir=$SAVE_DIR --model_file=$MODEL_FILE \
        --train_src_file=$TRAIN_FILE1 --train_trg_file=$TRAIN_FILE2 \
        --valid_src_file=$VALID_FILE1 --valid_trg_file=$VALID_FILE2 --valid_post=$VALID_POST \
        --src_dict=$DICT1 --trg_dict=$DICT2 --xy=$XY --param_shared=$PARAM_SHARED \
        --dim_wemb=$DIM_WEMB --dim_model=$DIM_MODEL --n_head=$N_HEAD --dk=$DK --dv=$DV \
        --n_layers=$N_LAYERS --dim_ff=$DIM_FF --batch_size=$BATCH_SIZE --update_step=$UPDATE_STEP \
        --label_smooth=$LABEL_SMOOTH \
        --emb_noise=$EMB_NOISE --opt_scheduled=$OPT_SCHEDULED \
        --log_interval=5000 --valid_start=500000 --valid_interval=50000 \
        --resume=$RESUME --resume_line=$RESUME_LINE --resume_file=$RESUME_FILE --lr=$LR \
        --logging=1
        #--log_interval=5000 --valid_start=500000 --valid_interval=50000 \
 #5000, 500000, 50000
# kr2en (in valid)
# NMT2022-11: original (sorted) version : 38.35, kr2en.l6.d256.ff1024.gpu5
# NMT2022-10: unsorted version: 38.53, kr2en.l6.d256.ff1024.gpu6.nosort

# NMT2022-40: label smoothing on NMT2022-11: 38.71, kr2en.l6.d256.ff1024.gpu4.lr0.0001.label0.01
# NMT2022-50: dec emb shared on NMT2022-40: 36.82 (yet)
