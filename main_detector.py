import argparse
from collections import Counter

import torch

from src.data.data import get_dataloaders, load_image
from src.models.Detector import DOTA_CLASSES, Detector
from src.training.runner import build_trainer
from src.utils.config_loader import load_config, merge_configs
from src.utils.plots import plot_detections
from src.utils.utils_training import select_detections, selection_desc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_detector.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument('--img_height', type=int)
    parser.add_argument('--img_width', type=int)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--predict", action="store_true", default=False)
    parser.add_argument("--use_features", action="store_true", default=False,
                        help="train/test on precomputed backbone features "
                             "(scripts/cache_features.py) instead of raw images")
    parser.add_argument("--accelerator", default=None,
                        help="Lightning accelerator ('cpu', 'mps', 'auto'). Defaults to "
                             "training.accelerator in the config. 'cpu' is the fast "
                             "choice for the heads-only path: roi_align's MPS kernel is "
                             "pathologically slow (see build_trainer)")
    parser.add_argument("--image", default=None, help="path to a single image to run --predict on")
    parser.add_argument("--score_thr", type=float, default=None,
                        help="display score floor for printing/drawing. The ROI head "
                             "already applies its own (permissive) threshold and NMS; "
                             "omit this to see every detection it returns")
    parser.add_argument("--top_k", type=int, default=None,
                        help="print/draw only the top_k highest-scoring detections. "
                             "A rank cap, since the number of detections per image is "
                             "variable rather than fixed")
    parser.add_argument("--ckpt", default=None, help="checkpoint path to load for testing")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_configs(config, args)

    accelerator = args.accelerator or config["training"].get("accelerator", "auto")

    # creating dataloaders
    loaders = get_dataloaders(args, config)
    train_loader = loaders.get('train', None)
    val_loader = loaders.get('val', None)
    test_loader = loaders.get('test', None)

    if args.train:
        # trainer
        trainer = build_trainer(log_dir="outputs.nosync/detector_outputs", log_name="detector_train_log",
                                ckpt_dir="models.nosync/detector_checkpoints",
                                ckpt_name="detector-{epoch:02d}-{val_loss:.4f}", training=True, max_epochs=config["training"]["epochs"],
                                accelerator=accelerator)

        # model
        model = Detector(
            path2ijepa=config["model"]["path2ijepa"],
            lr=config["training"]["lr"],
            precomputed_features=args.use_features,
        )

        trainer.fit(model, train_loader, val_loader)

    if args.test:
        if args.ckpt is None:
            raise ValueError("--test requires --ckpt pointing to a trained checkpoint")

        trainer = build_trainer(log_dir="outputs.nosync/detector_outputs", log_name="detector_test_log",
                                log_version="version_" + args.ckpt,
                                training=False, accelerator=accelerator)

        model = Detector.load_from_checkpoint(args.ckpt)
        # test batches come from a feature or image loader independently of how
        # the checkpoint was trained, so set the mode from the CLI flag here
        model.precomputed_features = args.use_features
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

        detection = detections[0]
        # same display rule the plot uses, so the printout and the figure agree
        boxes, scores, labels, n_returned = select_detections(
            detection, args.score_thr, args.top_k)
        desc = selection_desc(args.score_thr, args.top_k)

        print(f"{scores.numel()} of {n_returned} returned detections ({desc})")
        if n_returned:
            all_scores = detection['scores'].detach().cpu()
            print(f"  returned score range: {all_scores.min():.3f} - {all_scores.max():.3f}")

        # per-class tally first: a DOTA frame can hold hundreds of one class, and
        # the count matters more than the individual boxes
        counts = Counter(DOTA_CLASSES[l] for l in labels.tolist())
        for name, n in counts.most_common():
            print(f"  {name:<20} x{n}")

        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            x1, y1, x2, y2 = (round(c, 1) for c in box)
            print(f"  {DOTA_CLASSES[label]:<20} {score:.3f}  [{x1}, {y1}, {x2}, {y2}]")

        plot_detections(image, detection, DOTA_CLASSES,
                        score_thr=args.score_thr, top_k=args.top_k)
