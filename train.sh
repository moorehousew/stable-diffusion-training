#!bin/sh

export TRAIN_DATA_DIR="./dataset"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export MODEL_NAME="./sd-finetuned-model"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --use_ema \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --num_train_epochs=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --output_dir="./sd-finetuned-model" 
