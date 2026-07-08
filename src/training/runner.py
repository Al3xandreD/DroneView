import lightning as L

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor

def build_trainer(log_dir: str, log_name: str, log_version: str=None, ckpt_dir: str = None, ckpt_name: str = None,
                  max_epochs=None, training: bool = False):
    '''
    Shared runner function
    :param ckpt_name:
    :param log_dir: path to the tensorboard logger directory
    :param ckpt_dir: path to the checkpoint directory
    :param training: train boolean
    :return:
    '''
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=log_name,
        version=log_version,
    )

    if not training:
        return L.Trainer(accelerator="auto",logger=logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_name,
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        every_n_epochs=1,
    )
    device_stats_callback = DeviceStatsMonitor()
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        accelerator='auto',
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=10,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, device_stats_callback, lr_monitor_callback],
    )

    return trainer

