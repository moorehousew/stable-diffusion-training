#!bin/sh

export MODEL_NAME="./sd-finetuned-model"
export DATASET_NAME="./dataset"

# HACK: Passing any arguments to this script resumes training from the existing model.
if (( $# != 1))
then
	export MODEL_NAME="CompVis/stable-diffusion-v1-4"
fi

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --num_train_epochs=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --output_dir="./sd-finetuned-model" 
