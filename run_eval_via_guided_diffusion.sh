#!/bin/bash
set -ex

OUTPUT_IMAGES_DIR=$1

python shuffle_gen_images.py --src_dir "${OUTPUT_IMAGES_DIR}"
python tools/save_npz.py --sample_dir "${OUTPUT_IMAGES_DIR}-shuffle"
python guided-diffusion/evaluations/evaluator.py \
    outputs/model_weights/ADM_npz/VIRTUAL_imagenet256_labeled.npz \
    "${OUTPUT_IMAGES_DIR}-shuffle.npz"