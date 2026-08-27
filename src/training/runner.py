import lightning as L

from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor

def build_trainer(log_dir: str, log_name: str, log_version: str=None, ckpt_dir: str = None, ckpt_name: str = None,
                  max_epochs=None, training: bool = False, accelerator: str = "auto", mlf_exp="ijepa_droneview",
                  tracking_uri: str = None):
    '''
    Shared runner function
    :param ckpt_name:
    :param log_dir: path to the tensorboard logger directory
    :param ckpt_dir: path to the checkpoint directory
    :param training: train boolean
    :param accelerator: Lightning accelerator. "auto" picks MPS on Apple Silicon,
        which is *not* always the fast choice here: torchvision's roi_align has a
        pathological MPS kernel (its cost barely varies with ROI count, and its
        backward measured ~10x slower than CPU on an M1 Pro), and the detection
        heads are small enough that the CPU wins outright. Prefer "cpu" for the
        heads-only / precomputed-feature path; "auto" or "mps" still makes sense
        for the backbone-bound path (scripts/cache_features.py), where the ViT
        forward dominates and runs well on the GPU.
    :return:
    '''

    mlf_logger = MLFlowLogger(
        experiment_name=mlf_exp,
        tracking_uri=tracking_uri,
    )

    tsb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=log_name,
        version=log_version,
    )

    if not training:
        return L.Trainer(accelerator=accelerator, logger=[tsb_logger, mlf_logger]), mlf_logger

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_name,
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        every_n_epochs=1,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        accelerator=accelerator,
        logger=[tsb_logger, mlf_logger],
        max_epochs=max_epochs,
        log_every_n_steps=10,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, lr_monitor_callback],
    )

    return trainer, mlf_logger

