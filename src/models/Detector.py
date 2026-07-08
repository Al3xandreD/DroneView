from src.models.I_JEPA import IJEPA

import lightning as L
import torch
from torch import nn


class Detector(L.LightningModule):
    '''
    Downstream oriented-box regressor on top of a frozen I-JEPA backbone.

    The I-JEPA target encoder is reused as a frozen feature extractor: an image
    is turned into N per-patch representations, mean-pooled into a single (B, D)
    image embedding, and an MLP head regresses a fixed number of oriented boxes
    (num_boxes) each described by 8 coordinates (4 corner points).

    Because the head output is fixed-size but DOTA has a variable number of boxes
    per image, targets are padded/truncated to num_boxes slots and a validity
    mask keeps padded slots out of the loss. This uses no box<->slot matching
    (slots are compared to targets in dataset order), which is the known
    limitation of the pooled formulation; a per-patch grid / DETR-style head is
    the upgrade path if you need permutation-invariant, variable-count detection.
    '''

    def __init__(self, path2ijepa, num_boxes=100, head_hidden_dim=512, lr=1e-3):
        super(Detector, self).__init__()
        # path2ijepa is stored as a hyperparameter so the detector can be rebuilt
        # from a checkpoint; the backbone weights themselves are saved with it.
        self.save_hyperparameters()

        self.num_boxes = num_boxes      # K: number of fixed box slots the head predicts
        self.num_coords = 8                 # 4 corner points (x1,y1,...,x4,y4) per oriented box
        self.lr = lr

        # backbone: frozen I-JEPA target encoder (the stable EMA representation)
        ijepa = IJEPA.load_from_checkpoint(path2ijepa)
        self.target_encoder = ijepa.target_encoder.eval()
        self.target_encoder.freeze()    # no grads / stays in eval mode

        D = self.target_encoder.hidden_dim              # per-token feature dimension (768)
        self.coord_scale = float(self.target_encoder.image_size)

        # head: pooled image embedding (B, D) -> (B, K * 8)
        self.head = nn.Sequential(
            nn.Linear(D, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, self.num_boxes * self.num_coords),
        )

    def _extract_features(self, images):
        '''
        Run the frozen backbone the same way I-JEPA produced its representations
        and mean-pool the patch tokens into one embedding per image.
        :param images: (B, C, H, W)
        :return: (B, D) pooled image embedding
        '''
        # using I_JEPA functions
        with torch.no_grad():
            x = IJEPA._embed(self.target_encoder, images)        # (B, N, D)
            x = IJEPA._transformer(self.target_encoder, x)       # (B, N, D)
        return x.mean(dim=1)                                     # (B, D)

    def forward(self, images):
        '''
        :param images: (B, C, H, W)
        :return: (B, num_boxes, 8) predicted oriented boxes, in normalised coords
        '''
        features = self._extract_features(images)               # (B, D)
        out = self.head(features)                               # (B, K * 8)
        return out.view(-1, self.num_boxes, self.num_coords)        # (B, K, 8)

    def _prepare_targets(self, boxes, device):
        '''
        Pad/truncate the variable-length per-image boxes to num_boxes slots and
        build a validity mask. Coordinates are normalized by coord_scale to match
        the head's output range.
        :param boxes: list (len B) of (n_i, 8) tensors from collate_fn
        :param device: device to place the target tensors on
        :return: (targets (B, K, 8), mask (B, K)) where mask is 1.0 for real boxes
        '''
        B = len(boxes)
        targets = torch.zeros(B, self.num_boxes, self.num_coords, device=device)
        mask = torch.zeros(B, self.num_boxes, device=device)
        for i, b in enumerate(boxes):
            n = min(b.shape[0], self.num_boxes)     # truncate images with > K boxes
            if n > 0:
                targets[i, :n] = b[:n].to(device) / self.coord_scale # coordinates are normalized
                mask[i, :n] = 1.0   # mask is True
        return targets, mask

    def _shared_step(self, batch):
        '''
        Masked coordinate regression loss shared by train/val/test.
        :param batch: (images, boxes, labels, difficulties) from collate_fn
        :return: mean squared error over valid box slots only
        '''
        images, boxes, _, _ = batch
        preds = self.forward(images) # predicted boxes  # (B, K, 8)
        targets, mask = self._prepare_targets(boxes, images.device)

        se = ((preds - targets) ** 2).mean(dim=-1)              # (B, K) mean over the 8 coords
        # average over valid slots only; clamp avoids div-by-zero on empty batches
        loss = (se * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        '''
        Only the head is trained; the backbone is frozen, so its parameters are
        excluded from the optimizer.
        :return:
        '''
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr)