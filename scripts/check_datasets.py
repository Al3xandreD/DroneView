import argparse

from src.data.DOTADataset import DOTADataset
from src.data.data import find_single_channel_images
from src.utils.config_loader import load_config



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
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