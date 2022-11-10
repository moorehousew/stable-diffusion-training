# Converts a WaifuDiffusion structured dataset to an ImageFolder-compatible dataset (use ./path-to-dataset/img as data_dir)

import argparse
import glob
import os.path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', default=None, type=str, required=True, help='Path to the dataset to convert.')

    args = parser.parse_args()

    assert args.dataset_path is not None, 'Must provide a dataset path!'
    
    images_path = os.path.join(args.dataset_path, 'img')
    captions_path = os.path.join(args.dataset_path, 'txt')
    
    # Generate the metadata file.
    metadata = []
    for image_filename in glob.iglob(os.path.join(images_path, '*.*'), recursive = False):
        image_name = os.path.basename(image_filename)
        name = image_name.split('.')[0]
        caption_filename = os.path.join(captions_path, name + '.txt')
        
        caption = None
        with open(caption_filename, 'r') as caption_file:
            caption = caption_file.read().replace('\n', '')
        
        metadata.append(
            '{ "file_name": "' +
            image_name +
            '", "text": "' +
            caption +
            '" }'
        )
    
    metadata_filename = os.path.join(args.dataset_path, 'metadata.jsonl')
    with open(metadata_filename, 'w') as metadata_file:
        for m in metadata:
            metadata_file.write('%s\n' % m)
    
    print('Done')
    