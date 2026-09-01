#!/bin/bash

DATA_INFO=/path/to/data_info.json
IMAGE_ROOT=/path/to/images
SAVE_ROOT=/path/to/scores

# --- CLIP RN50 ---
CUDA_VISIBLE_DEVICES=0 python ../compute_clip_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --clip_model_name RN50 \
    --save_root_path ${SAVE_ROOT}/clip-r50 \
    --save_filename data_info_clip-r50.json \
    --save_key clip-r50_similarity \
    --batchsize 100

# --- CLIP ViT-B/32 ---
CUDA_VISIBLE_DEVICES=0 python ../compute_clip_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --clip_model_name ViT-B/32 \
    --save_root_path ${SAVE_ROOT}/clip-vit-b32 \
    --save_filename data_info_clip-vit-b32.json \
    --save_key clip-vit-b32_similarity \
    --batchsize 100

# --- CLIP ViT-L/14 ---
CUDA_VISIBLE_DEVICES=0 python ../compute_clip_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --clip_model_name ViT-L/14 \
    --save_root_path ${SAVE_ROOT}/clip-vit-l14 \
    --save_filename data_info_clip-vit-l14.json \
    --save_key clip-vit-l14_similarity \
    --batchsize 32
