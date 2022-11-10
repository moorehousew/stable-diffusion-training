import argparse
import time

from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Simple inference script'
    )
    
    parser.add_argument(
        '--pretrained_model_name_or_path',
        type = str,
        default = None,
        required = True,
        help = 'Path to pretrained model or model identifier from huggingface.co/models.',
    )
    
    parser.add_argument(
        '--output_dir',
        type = str,
        default = 'sd-model-finetuned-samples',
        help = 'The output directory where the model predictions and checkpoints will be written.',
    )
    
    parser.add_argument(
        '--prompt',
        type = str,
        default = None,
        required = True,
        help = 'The prompt for the inference.'
    )
    
    parser.add_argument(
        '--lowmem',
        type = bool,
        default = False,
        help = 'Reduces VRAM use during inference.'
    )
    
    args = parser.parse_args()
        
    return args


def main():
    args = parse_args()
    
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16
    )
    
    pipe.to('cuda')
    
    if args.lowmem:
        pipe.enable_attention_slicing()
    
    image = pipe(prompt = args.prompt).images[0]
    image_filename = time.strftime('%Y%m%d-%H%M%S') + '.png'
    
    image.save(
        os.path.join(
            args.output_dir,
            image_filename
        )
    )


if __name__ == "__main__":
    main()
