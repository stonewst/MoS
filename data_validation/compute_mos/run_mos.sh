#!/bin/bash

SCORES_DIR=/path/to/scores

python compute_mos.py \
    --data_info_path ${SCORES_DIR}/data_info_all_scores.json \
    --output_path ${SCORES_DIR}/data_info_mos.json \
    --score_keys \
        clip-r50_similarity \
        clip-vit-b32_similarity \
        clip-vit-l14_similarity \
        blip-base_itm-score \
        blip-large_itm-score \
    --normalize minmax --temp_min 0.5 --temp_max 1.5 \
    --save_key mos_score

