# train.sh 0 tm - train.sh GPU_ID MODEL 
TASK='wmt14'
MODE='test'
WORLD_SIZE=$2

SRC_LANG=$3   # en
TRG_LANG=$4   # de

SAVE_DIR='./results/'
DATA_DIR='./data/wmt14_en_de/'

YEAR=$5
if [ $YEAR == 2014 ]
then
    LANG_PAIR=$SRC_LANG'-'$TRG_LANG
    TEST_SRC_FILE='bpe.test.'$YEAR'.'$LANG_PAIR'.en'
    TEST_TRG_FILE='bpe.test.'$YEAR'.'$LANG_PAIR'.de'
    TEST_TOKEN_REF_SRC_FILE='test.'$YEAR'.'$LANG_PAIR'.en'
    TEST_TOKEN_REF_TRG_FILE='test.'$YEAR'.'$LANG_PAIR'.de'
    TEST_REF_SRC_FILE='ref.'$YEAR'.'$LANG_PAIR'.en'
    TEST_REF_TRG_FILE='ref.'$YEAR'.'$LANG_PAIR'.de'
else
    TEST_SRC_FILE='bpe.test.'$YEAR'.en'
    TEST_TRG_FILE='bpe.test.'$YEAR'.de'
    TEST_TOKEN_REF_SRC_FILE='test.'$YEAR'.en'
    TEST_TOKEN_REF_TRG_FILE='test.'$YEAR'.de'
    TEST_REF_SRC_FILE='ref.'$YEAR'.en'
    TEST_REF_TRG_FILE='ref.'$YEAR'.de'
fi
TEST_BATCH_SIZE=20 #64 #100 for Ti GPU, 60 for non-Ti GPU | 60 is standard | 1 is for measure time
TEST_TOKEN_SIZE=9000

MODEL='normal_nmt'

echo "WMT14 ENDE Test"

# Testing configuration
BEAM_WIDTH=1
TEST_SUBDIR='wmt14_en2de_normal_nmt_0.001lr_8000TK_1US_0sort_50pat_0.1drop_6layer_512D_8H_'
TEST_MODEL_FILE=$SRC_LANG'2'$TRG_LANG'.ensemble_model.best.pth'
TEST_MODEL_ARGS='model.args.pkl'

SACREBLEU_TOKENIZER='13a' # '13a' is default
SACREBLEU_LOWERCASE=0 # 0 is default

CUDA_VISIBLE_DEVICES=$1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1\
        --nproc_per_node=$WORLD_SIZE nmt_run.py \
        --world_size=$WORLD_SIZE \
        --translation_task=$TASK --mode=$MODE --src_lang=$SRC_LANG --trg_lang=$TRG_LANG\
        --save_dir=$SAVE_DIR --data_dir=$DATA_DIR --model=$MODEL \
        --test_batch_size=$TEST_BATCH_SIZE --test_token_size=$TEST_TOKEN_SIZE \
        --test_subdir=$TEST_SUBDIR --test_src_file=$TEST_SRC_FILE --test_trg_file=$TEST_TRG_FILE\
        --test_ref_src_file=$TEST_REF_SRC_FILE --test_ref_trg_file=$TEST_REF_TRG_FILE \
        --test_token_ref_src_file=$TEST_TOKEN_REF_SRC_FILE \
        --test_token_ref_trg_file=$TEST_TOKEN_REF_TRG_FILE \
        --beam_width=$BEAM_WIDTH --test_model_file=$TEST_MODEL_FILE \
        --test_model_args=$TEST_MODEL_ARGS \
        --sacrebleu_tokenizer=$SACREBLEU_TOKENIZER --sacrebleu_lowercase=$SACREBLEU_LOWERCASE
