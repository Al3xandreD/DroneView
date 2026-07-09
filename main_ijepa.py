import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor

from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes

from src.models.Detector import Detector
from src.training.runner import build_trainer
from src.utils.config_loader import load_config, merge_configs
from src.data.data import get_dataloaders
from src.models.I_JEPA import IJEPA
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils.plots import plot_boxes, plot_representations

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_ijepa.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size",  type=int)
    parser.add_argument("--lr", type=float, help="peak learning rate reached at the end of warmup")
    parser.add_argument("--warmup_epochs", type=int, help="epochs to ramp lr up to the peak lr")
    parser.add_argument('--img_height', type=int)
    parser.add_argument('--img_width', type=int)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--predict", action="store_true", default=False)
    parser.add_argument("--ckpt", default=None, help="checkpoint path to load for testing")
    parser.add_argument("--show_sample", action="store_true", default=False, help="display a sample image with its bounding boxes before running")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_configs(config, args)

    # creating dataloaders
    loaders = get_dataloaders(args, config)
    train_loader = loaders.get('train', None)
    val_loader = loaders.get('val', None)
    test_loader = loaders.get('test', None)

    if args.show_sample and train_loader is not None:
        train_images, train_boxes, train_labels, train_difficulties = next(iter(train_loader))
        plot_boxes(train_images[0], train_boxes[0], train_labels[0])

    if args.train:
        # tensorboard logger
        trainer = build_trainer(log_dir="outputs.nosync/ijepa_outputs", log_name="ijepa_train_log",
                                ckpt_dir="models.nosync/ijepa_checkpoints",
                                ckpt_name="ijepa-{epoch:02d}-{val_loss:.4f}", training=True)

        # model
        model = IJEPA(
            config["ijepa"]["M"],
            lr=config["training"]["lr"],
            warmup_epochs=config["training"]["warmup_epochs"],
        )

        trainer.fit(model, train_loader, val_loader)

    if args.test:
        if args.ckpt is None:
            raise ValueError("--test requires --ckpt pointing to a trained checkpoint")

        trainer = build_trainer(log_dir="outputs.nosync/ijepa_outputs", log_name="ijepa_test_log",
                                log_version="version_" + args.ckpt,
                                training=False)

        model = IJEPA.load_from_checkpoint(args.ckpt)
        model.eval()

        trainer.test(model, test_loader)

    if args.predict:
        if args.ckpt is None:
            raise ValueError("--predict requires --ckpt pointing to a trained checkpoint")

        trainer = build_trainer(log_dir="outputs.nosync/ijepa_outputs", log_name="ijepa_pred_log",
                                log_version="version_" + args.ckpt,
                                training=False)

        model = IJEPA.load_from_checkpoint(args.ckpt)
        model.eval()

        predictions = trainer.predict(model, test_loader)
        plot_representations(predictions[0], 0)


