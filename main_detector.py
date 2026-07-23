import argparse

import torch

from src.data.data import get_dataloaders, load_image
from src.models.Detector import DOTA_CLASSES, Detector
from src.training.runner import build_trainer
from src.utils.config_loader import load_config, merge_configs
from src.utils.plots import plot_detections

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
    parser.add_argument("--image", default=None, help="path to a single image to run --predict on")
    parser.add_argument("--score_thr", type=float, default=0.5,
                        help="minimum class score for a detection to be printed/drawn")
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

        model = Detector.load_from_checkpoint(args.ckpt)
        model.eval()

        if args.image is not None:
            image = load_image(args.image, config['data']['new_image_size'])
        else:
            # fall back to the first image of the (val) test loader
            image = next(iter(test_loader))[0][0]

        with torch.no_grad():
            # list (len B) of {'boxes': (K,4), 'scores': (K,), 'labels': (K,)}
            detections = model(image.unsqueeze(0).to(model.device))

        detection = {k: v.cpu() for k, v in detections[0].items()}
        keep = detection['scores'] >= args.score_thr
        print(f"{int(keep.sum())} detections above {args.score_thr} "
              f"(of {detection['scores'].numel()} returned)")
        for box, score, label in zip(detection['boxes'][keep].tolist(),
                                     detection['scores'][keep].tolist(),
                                     detection['labels'][keep].tolist()):
            x1, y1, x2, y2 = (round(c, 1) for c in box)
            print(f"  {DOTA_CLASSES[label]:<20} {score:.3f}  [{x1}, {y1}, {x2}, {y2}]")

        plot_detections(image, detection, DOTA_CLASSES, score_thr=args.score_thr)
