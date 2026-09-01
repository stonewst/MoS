#!/bin/bash

DATA_INFO=/path/to/data_info.json
IMAGE_ROOT=/path/to/images
SAVE_ROOT=/path/to/scores
BLIP_CODE=/path/to/BLIP
BLIP_CKPT_DIR=/path/to/BLIP/checkpoints

# --- BLIP base ---
CUDA_VISIBLE_DEVICES=0 python ../compute_blip_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --blip_code_path ${BLIP_CODE} \
    --blip_model_ckpt ${BLIP_CKPT_DIR}/model_base.pth \
    --vit_arch base --image_size 224 \
    --save_root_path ${SAVE_ROOT}/blip-base \
    --save_filename data_info_blip-base.json \
    --save_key blip-base

# --- BLIP large ---
CUDA_VISIBLE_DEVICES=0 python ../compute_blip_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --blip_code_path ${BLIP_CODE} \
    --blip_model_ckpt ${BLIP_CKPT_DIR}/model_large.pth \
    --vit_arch large --image_size 224 \
    --save_root_path ${SAVE_ROOT}/blip-large \
    --save_filename data_info_blip-large.json \
    --save_key blip-large

# --- BLIP base fine-tuned on COCO retrieval ---
CUDA_VISIBLE_DEVICES=0 python ../compute_blip_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --blip_code_path ${BLIP_CODE} \
    --blip_model_ckpt ${BLIP_CKPT_DIR}/model_base_retrieval_coco.pth \
    --vit_arch base --image_size 384 \
    --save_root_path ${SAVE_ROOT}/blip-base-retrieval-coco \
    --save_filename data_info_blip-base-retrieval-coco.json \
    --save_key blip-base-retrieval-coco
