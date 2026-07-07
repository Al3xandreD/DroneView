import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor

from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes
from src.utils.config_loader import load_config, merge_configs
from src.data.data import get_dataloaders
from src.models.I_JEPA import IJEPA
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
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
        image = draw_bounding_boxes(train_images[0], train_boxes[0], labels=train_labels[0])
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.show()

    if args.train:
        # tensorboard logger
        logger = TensorBoardLogger(
            save_dir="outputs.nosync/ijepa_outputs",
            name="ijepa_train_log",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath="models.nosync/ijepa_checkpoints",
            filename="ijepa-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            every_n_epochs=1,
        )
        device_stats_callback = DeviceStatsMonitor()
        lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")


        # model
        model = IJEPA(
            config["model"]["M"],
            lr=config["training"]["lr"],
            warmup_epochs=config["training"]["warmup_epochs"],
        )
        # trainer
        trainer = L.Trainer(
            accelerator='auto',
            default_root_dir='models.nosync/ijepa_checkpoints',
            logger=logger,
            max_epochs=config["training"]["epochs"],
            log_every_n_steps=10,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, device_stats_callback, lr_monitor_callback],
        )
        trainer.fit(model, train_loader, val_loader)

    if args.test:
        # tensorboard logger
        logger = TensorBoardLogger(
            save_dir="outputs.nosync/ijepa_outputs",
            name="ijepa_test_log",
        )
        if args.ckpt is None:
            raise ValueError("--test requires --ckpt pointing to a trained checkpoint")
        model = IJEPA.load_from_checkpoint(args.ckpt)
        model.eval()
        trainer = L.Trainer(
            accelerator='auto',
            logger=logger,
        )
        trainer.test(model, test_loader)

    if args.predict:
        logger = TensorBoardLogger(
            save_dir="outputs.nosync/ijepa_outputs",
            name="ijepa_predict_log",
        )
        if args.ckpt is None:
            raise ValueError("--predict requires --ckpt pointing to a trained checkpoint")
        model = IJEPA.load_from_checkpoint(args.ckpt)
        model.eval()
        trainer = L.Trainer(
            accelerator='auto',
            logger=logger,
        )
        prediction = trainer.predict(model, test_loader)


