#!/bin/bash

DATA_INFO=/path/to/data_info.json
IMAGE_ROOT=/path/to/images
SAVE_ROOT=/path/to/scores

# --- EVA-CLIP ViT-L/14 ---
CUDA_VISIBLE_DEVICES=0 python ../compute_evaclip_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --model_name EVA02-L-14 --pretrained merged2b_s4b_b131k \
    --save_root_path ${SAVE_ROOT}/evaclip-vit-l14 \
    --save_filename data_info_evaclip-vit-l14.json \
    --save_key evaclip-vit-l14_similarity \
    --batchsize 32

# --- EVA-CLIP ViT-g/14 ---
CUDA_VISIBLE_DEVICES=0 python ../compute_evaclip_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --model_name EVA01-g-14 --pretrained laion400m_s11b_b41k \
    --save_root_path ${SAVE_ROOT}/evaclip-vit-g14 \
    --save_filename data_info_evaclip-vit-g14.json \
    --save_key evaclip-vit-g14_similarity \
    --batchsize 16
