#!/bin/bash

SCORES_DIR=/path/to/scores

python filter_data.py \
    --data_info_path ${SCORES_DIR}/data_info_mos.json \
    --output_path    ${SCORES_DIR}/data_info_filtered.json \
    --score_key mos_score \
    --keep_ratio 0.8 \
    --removed_path ${SCORES_DIR}/data_info_removed.json
