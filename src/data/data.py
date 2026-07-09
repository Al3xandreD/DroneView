import os

import torch
from torch.utils.data import DataLoader
from torchvision.io import decode_image, ImageReadMode

from src.data.DOTADataset import DOTADataset
from src.data.preprocess import get_im_transforms


def load_image(image_path, new_size):
    '''
    Load and preprocess a single image the same way DOTADataset does, so it can
    be fed straight to the model for single-image inference.
    :param image_path: path to an image file
    :param new_size: (H, W) the image is resized to
    :return: (C, H, W) float tensor
    '''
    image = decode_image(image_path, mode=ImageReadMode.RGB)
    return get_im_transforms(new_size)(image)

def collate_fn(batch):
    '''
    Custom collate function for pytorch dataloader
    :param batch:
    :return:
    '''
    images      = torch.stack([b[0] for b in batch])
    boxes       = [b[1] for b in batch]        # list of tensors (variable size)
    labels      = [b[2] for b in batch]        # list of lists of strings
    difficulties = [b[3] for b in batch]

    return images, boxes, labels, difficulties

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

def get_dataloaders(args, config):
    loaders = {} # dict of loaders
    if args.train:
        train_dataset = DOTADataset(config['data']['train']['annotation_path'],
                                    config['data']['train']['image_path'],
                                    config['data']['new_image_size'])

        val_dataset = DOTADataset(config['data']['val']['annotation_path'],
                                  config['data']['val']['image_path'],
                                  config['data']['new_image_size'])
        train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=config['training']['num_workers'], persistent_workers=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=config['training']['num_workers'], persistent_workers=True)
        loaders['train'] = train_dataloader
        loaders['val'] = val_dataloader


    if args.test or args.predict:
        # DOTA's test split ships no public ground-truth boxes, evaluate on
        # the val split (which has boxes) instead. Reported detection metrics are
        # therefore validation-set numbers, not held-out test numbers.
        test_dataset = DOTADataset(config['data']['val']['annotation_path'],
                                   config['data']['val']['image_path'],
                                   config['data']['new_image_size'])
        test_dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=config['training']['num_workers'], persistent_workers=True)
        loaders['test'] = test_dataloader

    return loaders
