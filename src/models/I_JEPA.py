import math
import random

import lightning as L
import torch
from torch import nn

from torchvision.models import vit_b_16

class IJEPA(L.LightningModule):
    def __init__(self, M, lr=1e-3, warmup_start_lr=1e-4, final_lr=1e-6, warmup_epochs=5,
                 ema_momentum=0.996, target_scale=(0.15, 0.2), target_ratio=(0.75, 1.5), context_scale=(0.85, 1.0),
                 tiler = None):
        super(IJEPA, self).__init__()
        # saving the constructor args as hyperparameters so the model can be
        # rebuilt with IJEPA.load_from_checkpoint(...)
        self.save_hyperparameters(ignore=['tiler'])
        # attributes
        self.M = M # number of sampled targets
        self.lr = lr # peak learning rate reached at the end of warmup
        self.warmup_start_lr = warmup_start_lr # learning rate at the start of warmup
        self.final_lr = final_lr # learning rate the cosine schedule decays to
        self.warmup_epochs = warmup_epochs # number of epochs to ramp lr from warmup_start_lr to lr
        self.ema_momentum = ema_momentum # EMA decay for the target encoder
        self.target_scale = target_scale
        self.target_ratio = target_ratio
        self.context_scale = context_scale
        self.context_ratio = (1.0, 1.0)

        # network
        self.context_encoder = vit_b_16()
        self.target_encoder = vit_b_16()
        self.predictor = vit_b_16()

        dim = self.context_encoder.hidden_dim

        # image tiler
        self.tiler = tiler

        # learnable mask token used by the predictor for the patches to predict
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # freezing target encoder weights for EMA
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _embed(vit, imgs: torch.Tensor) -> torch.Tensor:
        '''
        Embedding the images to get the patches with positions tokens
        :param vit:
        :param imgs: input images (B, C, H, W), batch, number of channels, image height, image width
        :return: patches with positions tokens (B, N, h, w), batch, number of patches, patch height, patch width
        '''

        x = vit.conv_proj(imgs) # (B, D, h, w)
        x = torch.flatten(x, 2) # (B, D, N)
        x = torch.transpose(x, 1, 2)    # (B, N, D)
        x = x + vit.encoder.pos_embedding[:, 1:, :] # adding position embedding from the vit encoder layer
        return x

    @staticmethod
    def _transformer(vit, x:torch.Tensor) -> torch.Tensor:
        '''
        Passes tensor x through the encoder layers of a vit.
        Bypassing the conv layer and head present in the pytorch implementation
        :param vit: pytorch vit model
        :param x:
        :return:
        '''
        x = vit.encoder.dropout(x)
        x = vit.encoder.layers(x)
        x = vit.encoder.ln(x)
        return x
    def _get1block_indices(self,  gh, gw, scale:tuple[float, float], ratio:tuple[float, float]):
        '''
        Randomly sample a block inside the image with given scale and ratio, and returns the indices of the patches within the block
        :param gh: number of patches on height axis
        :param gw: number of patches on width axis
        :param scale:
        :param ratio:
        :return:
        '''
        s = random.uniform(scale[0], scale[1])  # random scale
        r = random.uniform(ratio[0], ratio[1])  # random ratio
        block_area = s * gh * gw
        block_height = int(round((block_area * r) ** 0.5))  # computing block height and width
        block_width = int(round((block_area / r) ** 0.5))

        bh = max(1, min(block_height, gh))  # keeping the block inside the frame
        bw = max(1, min(block_width, gw))
        top = random.randint(0, gh - block_height)  # random location
        left = random.randint(0, gw - block_width)

        rows = torch.arange(top, top + bh)  # indices
        cols = torch.arange(left, left + bw)
        # flattened index of patch (row, col) in the length-N sequence is row * gw + col
        idx = (rows.unsqueeze(1) * gw + cols.unsqueeze(0)).flatten()
        return idx

    def _block_indices(self, gh, gw, scale:tuple[float, float], ratio:tuple[float, float]):
        '''
        Randomly sampling M blocks in the image, and returns the indices of the patches within the blocks
        :param gh: number of patches on height axis
        :param gw: number of patches on width axis
        :param scale:
        :param ratio:
        :return:
        '''

        return [self._get1block_indices(gh, gw, scale, ratio) for _ in range(self.M)]


    def _shared_step(self, batch):
        '''
        Computes the I-JEPA prediction loss for a batch. Shared by the
        training, validation and test steps.
        :param batch:
        :return: mean L2 loss between predicted and frozen target representations
        '''
        y, _, _, _ = batch
        device = y.device
        B = y.shape[0]

        def _step(y):
            # dividing input in patches
            gh = gw = self.target_encoder.image_size//self.target_encoder.patch_size
            N = gh*gw # number of patches

            with torch.no_grad():
                # target encoder side
                target_patches = self._embed(self.target_encoder, y) # patches out of the encoder
                s_y = self._transformer(self.target_encoder, target_patches) # representation out of vit
                idx_target = self._block_indices(gh, gw, self.target_scale, self.target_ratio) # indices of sampled M target patches in the blocks
                idx_target = torch.cat(idx_target, dim=0).unique(dim=0).to(device) # union of all target patch indices
                sampled_y = s_y[:, idx_target] # (B, n_target, D) sampled block representations to predict

                # sampling one context block and removing the target patches from it
                idx_context = self._get1block_indices(gh, gw, self.context_scale, self.context_ratio)
                keep = torch.zeros(N, dtype=torch.bool) # mask of patches kept in the context
                keep[idx_context] = True
                keep[idx_target.cpu()] = False
                idx_context = torch.nonzero(keep).squeeze(1).to(device)

            # context encoder side
            context_patches = self._embed(self.context_encoder, y)  # (B, N, D)
            context_patches = context_patches[:, idx_context]       # (B, n_context, D)
            s_x = self._transformer(self.context_encoder, context_patches)

            # predict the target representations from the context
            pos = self.predictor.encoder.pos_embedding[:, 1:, :] # (1, N, D) predictor position embeddings
            n_target = idx_target.shape[0]

            ctx = s_x + pos[:, idx_context, :]                                     # context tokens + their positions
            mask = self.mask_token.expand(B, n_target, -1) + pos[:, idx_target, :] # mask tokens at the target positions

            seq = torch.cat([ctx, mask], dim=1)          # (B, n_context + n_target, D)
            seq = self._transformer(self.predictor, seq)
            pred_y = seq[:, -n_target:]                   # (B, n_target, D) predicted target representations

            # mean L2 distance between predictions and the frozen target representations
            loss = nn.functional.mse_loss(pred_y, sampled_y)
            return loss

        # if tiling
        if self.tiler:

            tiles = self.tiler(y) # (B, C, L, P, P)
            B, C, L, P, _ = tiles.shape
            assert tiles.shape[-1] == self.target_encoder.image_size, (
                f"tile size {tiles.shape[-1]} != encoder image_size {self.target_encoder.image_size}"
            ) # tiles shape must be equal to target encoder image size
            tiles = tiles.permute(0, 2, 1, 3, 4) # batching the tiles
            tiles = tiles.reshape(B*L, C, P, P)

            loss = _step(tiles)
            return loss

        else:
            loss = _step(y)
            return loss

    def training_step(self, batch, batch_idx):
        '''
        Training for building an internal representation
        :param batch:
        :param batch_idx:
        :return:
        '''
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def _update_target_encoder(self):
        '''
        EMA update of the (frozen) target encoder from the context encoder:
        theta_target = m * theta_target + (1 - m) * theta_context.
        :return:
        '''
        m = self.ema_momentum
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(m).add_(p_c.data, alpha=1.0 - m)
        # keep non-trainable buffers (if any) in sync
        for b_t, b_c in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            b_t.data.copy_(b_c.data)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        '''
        Update the target encoder after each optimizer step.
        '''
        self._update_target_encoder()

    def validation_step(self, batch, batch_idx):
        '''
        Same prediction objective as training, evaluated on held-out data
        (Lightning runs this under eval mode and torch.no_grad()).
        :param batch:
        :param batch_idx:
        :return:
        '''
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        '''
        Same prediction objective as training, evaluated on the test set.
        :param batch:
        :param batch_idx:
        :return:
        '''
        loss = self._shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Encode full images with the (frozen) target encoder, no masking applied.
        This is the representation used for inference / downstream detection.
        Returns per-patch tokens so a detection head can consume the spatial grid;
        mean-pool over dim=1 for a single (B, D) image embedding instead.
        :param images: (B, C, H, W)
        :return: (B, N, D) per-patch representations
        '''
        patches = self._embed(self.target_encoder, images)          # (B, N, D)
        return self._transformer(self.target_encoder, patches)      # (B, N, D)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Extract the learned image representation for inference / downstream use.
        :param batch: from collate_fn -> (images, boxes, labels, difficulties)
        :return: (B, N, D) per-patch representations
        '''
        y = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(y)

    def _lr_lambda(self, epoch):
        '''
        Multiplicative factor applied to the peak lr (self.lr) for a given epoch.
        Linear warmup from warmup_start_lr up to lr over warmup_epochs, then a
        cosine decay from lr down to final_lr over the remaining epochs.
        Returns a factor in [0, 1] so that factor * self.lr yields the target lr.
        :param epoch: current epoch index (0-based)
        :return: multiplicative factor for the peak learning rate
        '''
        if epoch < self.warmup_epochs:
            # linear warmup: warmup_start_lr -> lr
            warmup = self.warmup_start_lr + (self.lr - self.warmup_start_lr) * (epoch / max(1, self.warmup_epochs))
            return warmup / self.lr
        # cosine decay: lr -> final_lr over the epochs after warmup
        max_epochs = self.trainer.max_epochs
        if max_epochs is None or max_epochs < 0:
            # no finite horizon set: hold at peak lr rather than decaying blindly
            return 1.0
        total_decay_epochs = max(1, max_epochs - self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / total_decay_epochs
        progress = min(1.0, max(0.0, progress))
        decayed = self.final_lr + 0.5 * (self.lr - self.final_lr) * (1.0 + math.cos(math.pi * progress))
        return decayed / self.lr

    def configure_optimizers(self):
        '''
        Optimizer over the trainable parameters only (the target encoder is
        frozen and updated by EMA, so it is excluded), together with a per-epoch
        learning rate schedule: linear warmup (warmup_start_lr -> lr) followed by
        cosine decay (lr -> final_lr). Returned in Lightning's optimizer/scheduler
        format so the schedule is stepped and checkpointed automatically.
        :return:
        '''
        params = [p for p in self.parameters() if p.requires_grad]
        # AdamW is created at the peak lr; the scheduler's multiplicative factor
        # scales it each epoch (starting below 1.0 during warmup).
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # step the schedule once per epoch
                "frequency": 1,
            },
        }





