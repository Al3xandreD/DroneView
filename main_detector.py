import argparse

from src.data.data import get_dataloaders
from src.models.Detector import Detector
from src.training.runner import build_trainer
from src.utils.config_loader import load_config, merge_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_detector.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float, help="peak learning rate reached at the end of warmup")
    parser.add_argument('--img_height', type=int)
    parser.add_argument('--img_width', type=int)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--predict", action="store_true", default=False)
    parser.add_argument("--ckpt", default=None, help="checkpoint path to load for testing")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_configs(config, args)

    # creating dataloaders
    loaders = get_dataloaders(args, config)
    train_loader = loaders.get('train', None)
    val_loader = loaders.get('val', None)
    test_loader = loaders.get('test', None)

    if args.train:
        # trainer
        trainer = build_trainer(log_dir="outputs.nosync/detector_outputs", log_name="detector_train_log",
                                ckpt_dir="models.nosync/detector_checkpoints",
                                ckpt_name="detector-{epoch:02d}-{val_loss:.4f}", training=True, max_epochs=config["training"]["epochs"])

        # model
        model = Detector(
            path2ijepa=config["model"]["path2ijepa"],
            lr=config["training"]["lr"],
        )

        trainer.fit(model, train_loader, val_loader)

    if args.test:
        if args.ckpt is None:
            raise ValueError("--test requires --ckpt pointing to a trained checkpoint")

        trainer = build_trainer(log_dir="outputs.nosync/detector_outputs", log_name="detector_test_log",
                                log_version="version_" + args.ckpt,
                                training=False)

        model = Detector.load_from_checkpoint(args.ckpt)
        model.eval()

        trainer.test(model, test_loader)

    if args.predict:
        if args.ckpt is None:
            raise ValueError("--predict requires --ckpt pointing to a trained checkpoint")

        trainer = build_trainer(log_dir="outputs.nosync/detector_outputs", log_name="detector_test_log",
                                log_version="version_" + args.ckpt,
                                training=False)

        model = Detector.load_from_checkpoint(args.ckpt)
        model.eval()

        prediction = trainer.predict(model, test_loader)



