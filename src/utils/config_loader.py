import yaml
from src.utils.paths import ROOT

def load_config(path="configs/config.yml"):
    '''
    Loading the YAML config
    :param path: path to config file
    :return:
    '''
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # making the paths absolute
    for split in ['train', 'val', 'test']:
        config['data'][split]['image_path'] = ROOT / config['data'][split]['image_path']
        config['data'][split]['annotation_path'] = ROOT / config['data'][split]['annotation_path']

    return config

def merge_configs(config, args):
    '''
    Merging the configs in case that CLI arguments are used, the CLI arguments are used over the YAML config file.
    :param config:
    :param args:
    :return:
    '''

    # training args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    if getattr(args, "warmup_epochs", None):
        config['training']['warmup_epochs'] = args.warmup_epochs
    # image args
    if args.img_height:
        config['data']['new_image_size'][0] = args.img_height
    if args.img_width:
        config['data']['new_image_size'][1] = args.img_width

    return config