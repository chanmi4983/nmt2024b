# train.sh 0 tm - train.sh GPU_ID MODEL 
TASK='iwslt14'
MODE='test'
WORLD_SIZE=$2

SRC_LANG='en'   # en
TRG_LANG='de'   # de

SAVE_DIR='./results/'
DATA_DIR='./data/iwslt14_ende/'

TEST_SRC_FILE='bpe.test.en'
TEST_TRG_FILE='bpe.test.de'
TEST_TOKEN_REF_SRC_FILE='test.en'
TEST_TOKEN_REF_TRG_FILE='test.de'
TEST_REF_SRC_FILE='ref.en'
TEST_REF_TRG_FILE='ref.de'
TEST_BATCH_SIZE=256 #100 for Ti GPU, 60 for non-Ti GPU | 60 is standard | 1 is for measure time
TEST_TOKEN_SIZE=9000

MODEL='normal_nmt'

echo "IWSLT14 Test"

# Testing configuration
BEAM_WIDTH=3
TEST_SUBDIR='iwslt14_en2de_normal_nmt_0.0005lr_3600TK_1US_1sort_30pat_0.3drop_6layer_512D_4H_'
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
        --beam_width=$BEAM_WIDTH --test_model_file=$TEST_MODEL_FILE\
        --test_model_args=$TEST_MODEL_ARGS \
        --sacrebleu_tokenizer=$SACREBLEU_TOKENIZER --sacrebleu_lowercase=$SACREBLEU_LOWERCASE
