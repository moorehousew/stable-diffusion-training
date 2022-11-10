#!/bin/bash

export MODEL_NAME="./sd-finetuned-model"
export CHECKPOINT_NAME="./sd-finetuned-model.ckpt"

python convert_diffusers_to_sd.py \
  --model_path=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_NAME \
  --half
