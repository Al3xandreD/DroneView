import random

import lightning as L
import torch
from torch import nn

from torchvision.models import vit_b_16
from src.data.preprocess import makePatches

class IJEPA(L.LightningModule):
    def __init__(self, M, lr=1e-3, ema_momentum=0.996, target_scale=(0.15, 0.2), target_ratio=(0.75, 1.5), context_scale=(0.85, 1.0)):
        super(IJEPA, self).__init__()
        # saving the constructor args as hyperparameters so the model can be
        # rebuilt with IJEPA.load_from_checkpoint(...)
        self.save_hyperparameters()
        # attributes
        self.M = M # number of sampled targets
        self.lr = lr
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

        # learnable mask token used by the predictor for the patches to predict
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # freezing target encoder weights for EMA
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _embed(vit, imgs):
        '''
        Embedding the images to get the patches with positions tokens
        :param vit:
        :param imgs: input images (B, C, H, W)
        :return: patches with positions tokens (B, N, h, w)
        '''

        x = vit.conv_proj(imgs) # (B, D, h, w)
        x = torch.flatten(x, 2) # (B, D, N)
        x = torch.transpose(x, 1, 2)    # (B, N, D)
        x = x + vit.encoder.pos_embedding[:, 1:, :] # adding position embedding from the vit encoder layer
        return x

    @staticmethod
    def _transformer(vit, x):
        '''
        Passes tensor x through the encoder layers of a vit. Bypassing the conv layer and head present in the pytorch implementation
        :param vit:
        :param x:
        :return:
        '''
        # position embeddings are already added in _embed and the sequence length
        # here is variable (masked context + mask tokens), so we run the encoder
        # layers directly instead of vit.encoder(x), which would re-add a fixed
        # length-(N+1) positional embedding and crash on these shapes.
        x = vit.encoder.dropout(x)
        x = vit.encoder.layers(x)
        x = vit.encoder.ln(x)
        return x
    def _get1block_indices(self,  gh, gw, scale:tuple[float, float], ratio:tuple[float, float]):
        '''
        Randomly sample a block inside the image with given scale and ratio, and returns the indices of the patches within the block
        :param gh:
        :param gw:
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

        # dividing input in patches
        gh = gw = self.target_encoder.image_size//self.target_encoder.patch_size
        N = gh*gw # number of patches

        with torch.no_grad():
            # target encoder side
            target_patches = self._embed(self.target_encoder, y)
            s_y = self._transformer(self.target_encoder, target_patches)
            idx_target = self._block_indices(gh, gw, self.target_scale, self.target_ratio)
            idx_target = torch.cat(idx_target, dim=0).unique(dim=0).to(device) # union of all target patch indices
            sampled_y = s_y[:, idx_target] # (B, n_target, D) representations to predict

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

    def configure_optimizers(self):
        '''
        Optimizer over the trainable parameters only (the target encoder is
        frozen and updated by EMA, so it is excluded).
        :return:
        '''
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr)





