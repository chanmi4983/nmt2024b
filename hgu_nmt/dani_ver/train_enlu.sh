LANG1='en'
LANG2='lu'
DATA_DIR='../dataset'

EMB_NOISE=0.00 # 0.01

DIM_MODEL=256  #512
DIM_FF=2048
N_LAYERS=3

BATCH_SIZE=64  #124 
DIM_WEMB=$DIM_MODEL # should be the same

VOC_NUM=10000

# New data
DICT1=$DATA_DIR'/raw.vocab.'$LANG1'.tok.'$VOC_NUM'sub.safe.P10.pkl'
DICT2=$DATA_DIR'/raw.vocab.'$LANG2'.tok.'$VOC_NUM'sub.safe.P10.pkl'

# AI Hub
#TRAIN_FILE1=$DATA_DIR'/aihub.train.'$LANG1'.tok'$SYMBOL'.'$VOC_NUM'sub.safe'
#TRAIN_FILE2=$DATA_DIR'/aihub.train.'$LANG2'.tok'$SYMBOL'.'$VOC_NUM'sub.safe'
# AI Hub + HGU with AI Hub code and voca
TRAIN_FILE1=$DATA_DIR'/raw.train.'$LANG1'.tok.'$VOC_NUM'sub.safe'
TRAIN_FILE2=$DATA_DIR'/raw.train.'$LANG2'.tok.'$VOC_NUM'sub.safe'

VALID_FILE1=$DATA_DIR'/raw.valid.'$LANG1'.tok.'$VOC_NUM'sub.safe'
VALID_FILE2=$DATA_DIR'/raw.valid.'$LANG2'.tok'

SAVE_DIR='./results'
if [ ! -d $SAVE_DIR ]; then
    mkdir $SAVE_DIR
fi
if [ ! -d './trans' ]; then
    mkdir 'trans'
fi
MODEL_FILE=$LANG1'2'$LANG2'.l'$N_LAYERS'.d'$DIM_MODEL'.ff'$DIM_FF'.gpu'$1'.mod'
RESUME=0
OPT_SCHEDULED=0
RESUME_LINE=0 # The last iteration * BATCH_SIZE when RESUME=1
#RESUME_FILE='en2kr.l6.d512.ff2048.aihub.gpu1s4.pth.best.pth' # NMT-28
#RESUME_FILE='kr2en.l6.d512.ff2048.aihub.gpu4s9.pth.best.pth' # NMT-29
RESUME_FILE='en2lu.l3.d256.ff1024.gpu.best.pth' # NMT-67
#RESUME_FILE='en2lu.l6.d512.ff2048.raw.gpu1s2.voc.pth' # NMT-68

# for resume, opt_schedule=0, lr=0.0001, valid_start=100000 valid_every=5000

#COMMENT BY DANI: don't change CUDA_VISIBLE_DEVICES=0.

CUDA_LAUNCH_BLOCKING=1 python3.8 -W ignore nmt_run.py \
        --train=1 --rnn_name=$RNN \
        --save_dir=$SAVE_DIR --model_file=$MODEL_FILE \
        --train_src_file=$TRAIN_FILE1 --train_trg_file=$TRAIN_FILE2 \
        --valid_src_file=$VALID_FILE1 --valid_trg_file=$VALID_FILE2 \
        --src_dict=$DICT1 --trg_dict=$DICT2 \
        --dim_wemb=$DIM_WEMB --dim_model=$DIM_MODEL \
        --tm_n_layers=$N_LAYERS --tm_dim_ff=$DIM_FF --pos_enc=1 --batch_size=$BATCH_SIZE \
        --emb_noise=$EMB_NOISE --opt_scheduled=$OPT_SCHEDULED \
        --print_every=1000 --valid_start=5000 --valid_every=10000 \
        --resume=$RESUME --resume_line=$RESUME_LINE --resume_file=$RESUME_FILE --lr=0.0001 \
        --wandb=1 --ahead=750 --norm_case='ours' 
