import argparse
import os

from torchvision.io import decode_image

from src.data.DOTADataset import DOTADataset
from src.utils.config_loader import load_config


def find_single_channel_images(dataset):
    '''
    Find images in a DOTADataset that have only one channel (grayscale).
    Reads each image directly from disk to inspect its channel count without
    applying the dataset transforms.
    :param dataset: DOTADataset object
    :return: list of image filenames that have a single channel
    '''
    single_channel = []
    for image_name in dataset.image_files:
        image_path = os.path.join(dataset.images_dir, image_name)
        image = decode_image(image_path)   # shape: (C, H, W)
        if image.size(0) == 1:
            single_channel.append(image_name)

    return single_channel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_ijepa.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    split = config['data'][args.split]
    dataset = DOTADataset(split['annotation_path'], split['image_path'],
                          config['data']['new_image_size'])

    bad = find_single_channel_images(dataset)
    print(f"{len(bad)} single-channel images in '{args.split}':")
    for name in bad:
        print(f"  {name}")