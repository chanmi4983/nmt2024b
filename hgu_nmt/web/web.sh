BPE=1
BEAM_WIDTH=1

# pip3 install flask
CODE_HOME='..'
TOKENIZER=$CODE_HOME'/tools/tokenizer.perl'
DETOKENIZER=$CODE_HOME'/tools/detokenizer.perl'
MODEL_DIR='./models'
STOP_WORDS=$CODE_HOME'/share/stop_words.txt'

TEMP_DIR='./temp_dir'

DATA_DIR=$HOME'/hgu_data/processed'
KR_DICT=$DATA_DIR'/aihub.vocab.kr.tok.sym.10000sub.safe.P10.pkl'
EN_DICT=$DATA_DIR'/aihub.vocab.en.tok.sym.10000sub.safe.P10.pkl'
PN_DICT=$DATA_DIR'/pn_dict_test.txt'

KR_BPE_CODE=$DATA_DIR'/aihub.bpe.kr.tok.sym.10000.code'
EN_BPE_CODE=$DATA_DIR'/aihub.bpe.en.tok.sym.10000.code'

#E2K='en2kr.l6.d512.ff2048.aihub.gpu6s3.resume67' # 84
#K2E='kr2en.l6.d512.ff2048.aihub.gpu7s4.resume68' # 85
E2K='en2kr.gpu4'
K2E='kr2en.gpu5'
#E2K='en2kr.gpu2_ahead1'
#K2E='kr2en.gpu3_ahead1'
E2K_MODEL=$MODEL_DIR'/'$E2K'.best.pth' 
K2E_MODEL=$MODEL_DIR'/'$K2E'.best.pth' 
E2K_ARGS=$MODEL_DIR'/'$E2K'.args.pkl'
K2E_ARGS=$MODEL_DIR'/'$K2E'.args.pkl'

HOST_ADDR='203.252.112.19'
#HOST_ADDR='203.252.106.67'
PORT_NUM=8888
CUDA_VISIBLE_DEVICES=$1 python3 web_run.py \
        --tokenizer=$TOKENIZER --detokenizer=$DETOKENIZER --temp_dir=$TEMP_DIR \
        --e2k_args=$E2K_ARGS --k2e_args=$K2E_ARGS \
        --e2k_model=$E2K_MODEL --k2e_model=$K2E_MODEL \
        --kr_dict=$KR_DICT --en_dict=$EN_DICT --pn_dict=$PN_DICT --stop_words=$STOP_WORDS \
        --kr_bpe_code=$KR_BPE_CODE --en_bpe_code=$EN_BPE_CODE \
        --max_length=500 --beam_width=$BEAM_WIDTH \
        --host_addr=$HOST_ADDR --port_num=$PORT_NUM
