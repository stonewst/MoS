#!/bin/bash
# Download CC3M with img2dataset.
# Produces, per shard, a `{shard}/{key}.jpg` image folder and a `{shard}.parquet`
# metadata file next to it.
#
#   pip install img2dataset
#
# The raw CC3M list (Train_GCC-training.tsv, from the Conceptual Captions
# website) has two tab-separated columns (caption, url) and NO header, so we
# prepend a header line first.

RAW_TSV=Train_GCC-training.tsv          # downloaded from the Conceptual Captions website
URL_LIST=cc3m.tsv
OUT_DIR=/path/to/cc3m_images

# add a header so img2dataset knows the column names
if [ ! -f ${URL_LIST} ]; then
    printf 'caption\turl\n' > ${URL_LIST}
    cat ${RAW_TSV} >> ${URL_LIST}
fi

img2dataset \
    --url_list ${URL_LIST} \
    --input_format tsv \
    --url_col url \
    --caption_col caption \
    --output_format files \
    --output_folder ${OUT_DIR} \
    --processes_count 16 \
    --thread_count 64 \
    --image_size 384 \
    --resize_mode keep_ratio \
    --encode_quality 95 \
    --enable_wandb False
