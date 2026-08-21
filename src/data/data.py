import os

import torch
from torch.utils.data import DataLoader
from torchvision.io import decode_image, ImageReadMode

from src.data.DOTADataset import DOTADataset
from src.data.FeatureDataset import FeatureDataset, feature_collate_fn
from src.data.preprocess import get_im_transforms
from src.utils.paths import DATA


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

def feature_cache_dir(config, split):
    '''
    Directory of cached feature records for a split, matching the layout written
    by scripts/cache_features.py: <root>/<split>. The root comes from
    config['data']['feature_cache'] if set, else data.nosync/feature_cache.
    '''
    root = config['data'].get('feature_cache') or (DATA / 'feature_cache')
    return os.path.join(str(root), split)

def get_feature_dataloaders(args, config):
    '''
    Build dataloaders over precomputed backbone features (FeatureDataset) instead
    of raw images, for training the detection heads without the frozen encoder in
    the loop. Mirrors get_dataloaders' splits; the test/predict loaders reuse the
    val cache, following the same val-as-test convention as the image path.
    '''
    loaders = {}
    bs = config['training']['batch_size']
    nw = config['training']['num_workers']

    def make(split, shuffle):
        dataset = FeatureDataset(feature_cache_dir(config, split))
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle,
                          collate_fn=feature_collate_fn, num_workers=nw,
                          persistent_workers=True)

    if args.train:
        loaders['train'] = make('Train', shuffle=True)
        loaders['val'] = make('Val', shuffle=False)
    if args.test:
        # DOTA's test split has no public boxes; evaluate on the val cache
        loaders['test'] = make('Val', shuffle=False)
    return loaders

def get_dataloaders(args, config):
    # cached-feature training/eval path skips image loading entirely
    if getattr(args, 'use_features', False) and not args.predict:
        return get_feature_dataloaders(args, config)

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
