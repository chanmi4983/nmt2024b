# train.sh 0 tm - train.sh GPU_ID MODEL 
TASK='wmt14'
MODE='train'
WORLD_SIZE=$2
PORT=24388

# Datasets
SRC_LANG='en' 
TRG_LANG='de'
AHEAD=1000 #18000
SEED=0

JOINED_DICT=1
if [ $JOINED_DICT == 1 ]
then
    DICT1='vocab.32000.joined.pkl'
    DICT2=$DICT1
else
    DICT1=''
    DICT2=''
fi
SAVE_DIR='./results/'
DATA_DIR='./data/wmt14_en_de/'

TRAIN_FILE1='bpe.train.'$SRC_LANG
TRAIN_FILE2='bpe.train.'$TRG_LANG
VALID_FILE1='bpe.test.2013.'$SRC_LANG
VALID_FILE2='bpe.test.2013.'$TRG_LANG
#BATCH_SIZE=64 #60 # when N_LAYERS= 6 (BATCH_SIZE could be 64, when N_LAYERS<6)
TEST_BATCH_SIZE=128
TOKEN_SIZE=8000
TEST_TOKEN_SIZE=2000
SORTING=1

# Training
LR=0.001 #0.0005
OPT_SCHEDULED=1
LABEL_SMOOTHING=0.1
UPDATE_STEP=1
GRAD_CLIP=1.0
PATIENCE=50
N_CHECKPOINT=8 # 8 -> (8+2) checkpoint ensemble

# Loading
RESUME=0
LOAD_NEPTUNE_EXP_ID=''

# Model Architecture
MODEL='normal_nmt'  # 'normal_nmt', 'la_nmt', 'ladder_nmt', 'bt', 'ladder_bt'
DIM_MODEL=512
DIM_WEMB=$DIM_MODEL # should be the same
DROPOUT_P=0.1
DIM_FF=2048
N_LAYERS=6
TM_N_HEAD=8

NEPTUNE=0

echo "No neptune"
#Test dataloader, Test model in nmt_main.py, Special token in subdir name"


export NEPTUNE_API_TOKEN=""

CUDA_VISIBLE_DEVICES=$1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1\
        --nproc_per_node=$WORLD_SIZE nmt_run.py \
        --translation_task=$TASK --mode=$MODE --world_size=$WORLD_SIZE --port=$PORT \
        --src_lang=$SRC_LANG --trg_lang=$TRG_LANG --ahead=$AHEAD \
        --dataset_seed=$SEED --src_dict=$DICT1 --trg_dict=$DICT2 --save_dir=$SAVE_DIR \
        --data_dir=$DATA_DIR --train_src_file=$TRAIN_FILE1 --train_trg_file=$TRAIN_FILE2 \
        --valid_src_file=$VALID_FILE1 --valid_trg_file=$VALID_FILE2 \
        --token_size=$TOKEN_SIZE \
        --test_batch_size=$TEST_BATCH_SIZE --test_token_size=$TEST_TOKEN_SIZE \
        --lr=$LR --opt_scheduled=$OPT_SCHEDULED \
        --label_smoothing=$LABEL_SMOOTHING --update_step=$UPDATE_STEP --grad_clip=$GRAD_CLIP \
        --patience=$PATIENCE --n_checkpoint=$N_CHECKPOINT --model=$MODEL \
        --dim_model=$DIM_MODEL --dim_wemb=$DIM_WEMB --dropout_p=$DROPOUT_P --tm_dim_ff=$DIM_FF \
        --tm_n_layers=$N_LAYERS --tm_n_head=$TM_N_HEAD --neptune=$NEPTUNE \
        --resume=$RESUME --load_neptune_exp_id=$LOAD_NEPTUNE_EXP_ID \
        --print_every=300 --valid_start=3000 --valid_every=3000 # default
