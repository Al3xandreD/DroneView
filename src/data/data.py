import torch
from torch.utils.data import DataLoader

from src.data.DOTADataset import DOTADataset

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

def get_dataloaders(args, config):
    loaders = {} # dict of loaders
    if args.train:
        train_dataset = DOTADataset(config['data']['train']['annotation_path'],
                                    config['data']['train']['image_path'],
                                    config['data']['new_image_size'])

        val_dataset = DOTADataset(config['data']['val']['annotation_path'],
                                  config['data']['val']['image_path'],
                                  config['data']['new_image_size'])
        train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=config['training']['num_workers'])
        val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=config['training']['num_workers'])
        loaders['train'] = train_dataloader
        loaders['val'] = val_dataloader


    if args.test:
        test_dataset = DOTADataset(config['data']['test']['annotation_path'],
                                   config['data']['test']['image_path'],
                                   config['data']['new_image_size'])
        test_dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=config['training']['num_workers'])
        loaders['test'] = test_dataloader

    return loaders

