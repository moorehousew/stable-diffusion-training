#!/bin/bash

export MODEL_NAME="./sd-finetuned-model"
export PROMPT="A photo of a dog outside by a tree."

accelerate launch inference.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="./sd-finetuned-model-samples" \
  --prompt=#PROMPT \
  --lowmem=True
