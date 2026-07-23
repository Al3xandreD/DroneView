from src.models.I_JEPA import IJEPA
from src.models.RPN import RPN, poly8_to_aabb

import lightning as L
import torch


class Detector(L.LightningModule):
    '''
    Two-stage-style detector head on top of a frozen I-JEPA backbone.

    The I-JEPA target encoder is reused as a frozen feature extractor: an image
    is turned into N = H*W per-patch tokens which, being produced in row-major
    order, reshape back into a (B, D, H, W) spatial feature map. A Region
    Proposal Network (RPN) slides over that map and, at every location x anchor,
    scores objectness and regresses a box refinement.

    This is how a fixed-size head copes with a variable object count: rather than
    emitting a fixed number of oriented boxes (the previous pooled-MLP head, which
    forced padding/truncation to num_boxes slots), the RPN scores a fixed *dense*
    set of anchors and, at inference, selects a variable-length shortlist
    (top-k + NMS). The number of detected objects is the length of that list, not
    a tensor dimension.

    Scope: this wires up the first (proposal) stage only. DOTA's oriented boxes
    are reduced to axis-aligned enclosing boxes for the RPN targets; a second
    (ROI) stage that classifies each proposal and recovers orientation is the
    next step and is not built here.
    '''

    def __init__(self, path2ijepa, lr=1e-3):
        super(Detector, self).__init__()
        # path2ijepa is stored as a hyperparameter so the detector can be rebuilt
        # from a checkpoint; the backbone weights themselves are saved with it.
        self.save_hyperparameters()

        self.lr = lr

        # backbone: frozen I-JEPA target encoder (the stable EMA representation)
        ijepa = IJEPA.load_from_checkpoint(path2ijepa)
        self.target_encoder = ijepa.target_encoder.eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False    # no grads

        D = self.target_encoder.hidden_dim              # per-token feature dimension (768)
        self.image_size = int(self.target_encoder.image_size)   # 224
        self.patch_size = int(self.target_encoder.patch_size)   # 16
        self.grid = self.image_size // self.patch_size          # 14 tokens per side

        # detection stage: RPN over the ViT feature grid. Its stride (in image
        # pixels) is the patch size, since each token summarizes one patch.
        self.rpn = RPN(in_channels=D, stride=self.patch_size)

    def _extract_feature_map(self, images):
        '''
        Run the frozen backbone the same way I-JEPA produced its representations,
        then fold the flat token sequence back into a spatial (B, D, H, W) map.
        Tokens come out in row-major (row*grid + col) order, so a plain
        reshape + permute recovers the grid the RPN convolves over.
        :param images: (B, C, H, W)
        :return: (B, D, grid, grid) feature map
        '''
        with torch.no_grad():
            x = IJEPA._embed(self.target_encoder, images)        # (B, N, D)
            x = IJEPA._transformer(self.target_encoder, x)       # (B, N, D)
        B, N, D = x.shape
        return x.transpose(1, 2).reshape(B, D, self.grid, self.grid)

    def forward(self, images):
        '''
        :param images: (B, C, H, W)
        :return: list (len B) of proposal dicts {'boxes': (P,4), 'scores': (P,)}
                 in image pixel coordinates (axis-aligned).
        '''
        feat = self._extract_feature_map(images)                 # (B, D, H, W)
        objectness, deltas, anchors = self.rpn(feat)
        return self.rpn.proposals(objectness, deltas, anchors,
                                  (self.image_size, self.image_size))

    def _shared_step(self, batch):
        '''
        RPN objectness + box-regression loss shared by train/val/test.
        The oriented (8-coord) ground truth is reduced to axis-aligned enclosing
        boxes, which are what the horizontal-anchor RPN is trained against.
        :param batch: (images, boxes, labels, difficulties) from collate_fn
        :return: (total_loss, loss_dict) so callers can log the components
        '''
        images, boxes, _, _ = batch
        feat = self._extract_feature_map(images)                 # (B, D, H, W)
        objectness, deltas, anchors = self.rpn(feat)

        gt_boxes = [poly8_to_aabb(b.to(images.device)) for b in boxes]
        losses = self.rpn.loss(objectness, deltas, anchors, gt_boxes)
        return losses["rpn_loss"], losses

    def training_step(self, batch, batch_idx):
        loss, parts = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"train_{k}": v for k, v in parts.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, parts = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict({f"val_{k}": v for k, v in parts.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, parts = self._shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log_dict({f"test_{k}": v for k, v in parts.items()})
        return loss

    def configure_optimizers(self):
        '''
        Only the head is trained; the backbone is frozen, so its parameters are
        excluded from the optimizer.
        :return:
        '''
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr)