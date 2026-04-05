#!/bin/bash

LANG1='kr'
LANG2='en'

SAVE_DIR='./results'

# 모델 파일 지정
MODEL_PREFIX='kr2enPreLNdkEnt0203.gpu0.demo'
MODEL_FILE=$SAVE_DIR'/'$MODEL_PREFIX'.best.pth' 
ARGS_FILE=$SAVE_DIR'/'$MODEL_PREFIX'.args.pkl'
TRANS_FILE='./trans/'$MODEL_PREFIX'.no_bos_Hael' # 출력 파일

DATA_DIR=$HOME'/hgu_data/processed'
SYMBOL='.sym'
VOC_NUM=10000
DICT1=$DATA_DIR'/aihub.vocab.'$LANG1'.tok'$SYMBOL'.'$VOC_NUM'sub.safe.P10.pkl'
DICT2=$DATA_DIR'/aihub.vocab.'$LANG2'.tok'$SYMBOL'.'$VOC_NUM'sub.safe.P10.pkl'

# 정렬된 파일 경로를 지정
VALID_FILE1=$DATA_DIR'/aihub.valid.sorted.2.aihub.valid.'$LANG1'.tok'$SYMBOL
VALID_FILE2=$DATA_DIR'/aihub.valid.sorted.2.aihub.valid.'$LANG2'.tok'$SYMBOL
VALID_POST='.'$VOC_NUM'sub.safe' 

# BOS를 제거한 어텐션 맵 생성 (--save_attention --remove_bos 옵션 추가)
CUDA_VISIBLE_DEVICES=$1 python3 nmt_trans.py \
        --trans_model_file=$MODEL_FILE --trans_args_file=$ARGS_FILE \
        --src_dict=$DICT1 --trg_dict=$DICT2 \
        --valid_src_file=$VALID_FILE1 --valid_trg_file=$VALID_FILE2 --valid_post=$VALID_POST \
        --trans_file=$TRANS_FILE --save_attention --remove_bos

echo "BOS를 제거한 어텐션 맵이 ./trans/attention_maps_no_bos_Hael/ 디렉토리에 저장되었습니다."