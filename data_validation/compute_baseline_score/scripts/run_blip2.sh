#!/bin/bash

DATA_INFO=/path/to/data_info.json
IMAGE_ROOT=/path/to/images
SAVE_ROOT=/path/to/scores

# --- BLIP2 ViT-g/14 (pretrain) ---
CUDA_VISIBLE_DEVICES=0 python ../compute_blip2_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --name blip2_image_text_matching --model_type pretrain \
    --save_root_path ${SAVE_ROOT}/blip2-vit-g14 \
    --save_filename data_info_blip2-vit-g14.json \
    --save_key blip2-vit-g14

# --- BLIP2 fine-tuned on COCO ---
CUDA_VISIBLE_DEVICES=0 python ../compute_blip2_score.py \
    --data_info_path ${DATA_INFO} \
    --image_root_path ${IMAGE_ROOT} \
    --name blip2_image_text_matching --model_type coco \
    --save_root_path ${SAVE_ROOT}/blip2-coco \
    --save_filename data_info_blip2-coco.json \
    --save_key blip2-coco
