import lightning as L
import torch
from torch import nn
from torchvision.ops import box_iou, clip_boxes_to_image, nms, remove_small_boxes


def poly8_to_aabb(boxes):
    '''
    Convert oriented boxes given as 4 corner points (8 coords) into their
    axis-aligned enclosing box (x1, y1, x2, y2). The RPN reasons about
    horizontal proposals, so oriented ground truth is reduced to the tightest
    horizontal box that contains it; recovering the orientation is left to the
    (not-yet-built) second stage.
    :param boxes: (G, 8) tensor [x1,y1,x2,y2,x3,y3,x4,y4]
    :return: (G, 4) tensor [xmin, ymin, xmax, ymax]
    '''
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    xs = boxes[:, 0::2]                                  # (G, 4) the four corner x's
    ys = boxes[:, 1::2]                                  # (G, 4) the four corner y's
    return torch.stack([xs.min(1).values, ys.min(1).values,
                        xs.max(1).values, ys.max(1).values], dim=1)


class RPN(L.LightningModule):
    '''
    Region Proposal Network operating on a single feature map.

    The backbone, an I-JEPA model turns an image into a HxW grid
    of D-dim tokens, reshaped to a (B, D, H, W) feature map. The RPN slides a small
    conv over every location and, for each of A precomputed anchors at that
    location, predicts:
      - an objectness score
      - 4 box-regression deltas that refine the anchor into a proposal.

    The output tensor is fixed-size (H*W*A anchors).
    '''

    def __init__(self, in_channels, stride, lr=1e-3,
                 anchor_scales=(16, 32, 64, 128), anchor_ratios=(0.5, 1.0, 2.0),
                 mid_channels=256,
                 # training-time anchor assignment / sampling
                 pos_iou_thr=0.7, neg_iou_thr=0.3, num_samples=256, pos_fraction=0.5,
                 # proposal generation
                 pre_nms_topk=2000, post_nms_topk=1000, nms_thr=0.7, min_size=4):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.stride = stride                        # pixels between adjacent grid cells (image_size / grid)
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.num_samples = num_samples
        self.pos_fraction = pos_fraction
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thr = nms_thr
        self.min_size = min_size

        # base anchors (A, 4) centered at the origin; shifted across the grid on demand
        base_anchors = self._generate_base_anchors(anchor_scales, anchor_ratios)
        self.register_buffer("base_anchors", base_anchors, persistent=False)
        self.num_anchors = base_anchors.shape[0]            # A anchors per location

        # shared 3x3 conv then two sibling 1x1 conv heads (objectness / box deltas)
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_logits = nn.Conv2d(mid_channels, self.num_anchors, kernel_size=1)
        self.bbox_deltas = nn.Conv2d(mid_channels, self.num_anchors * 4, kernel_size=1)

        for layer in (self.conv, self.cls_logits, self.bbox_deltas):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0.0)

    # ------------------------------------------------------------------ anchors
    @staticmethod
    def _generate_base_anchors(scales, ratios):
        '''
        Build the A = len(scales) * len(ratios) reference boxes centered at (0, 0).
        A ratio r is interpreted as height/width at constant area = scale**2.
        :return: (A, 4) tensor of [x1, y1, x2, y2]
        '''
        anchors = []
        for s in scales:
            for r in ratios:
                w = s / (r ** 0.5)
                h = s * (r ** 0.5)
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32)

    def _grid_anchors(self, height, width, device):
        '''
        Place the base anchors at every cell center of the HxW grid.
        Ordering is (row y, col x, anchor a) so it lines up with the flattened
        prediction tensors below.
        :return: (H*W*A, 4) anchor boxes in image pixel coords
        '''
        shift_x = (torch.arange(width, device=device) + 0.5) * self.stride
        shift_y = (torch.arange(height, device=device) + 0.5) * self.stride
        cy, cx = torch.meshgrid(shift_y, shift_x, indexing="ij")        # (H, W) each
        centers = torch.stack([cx, cy, cx, cy], dim=-1).reshape(-1, 1, 4)  # (H*W, 1, 4)
        anchors = centers + self.base_anchors.to(device).unsqueeze(0)      # (H*W, A, 4)
        return anchors.reshape(-1, 4)

    # ------------------------------------------------------------------ deltas
    @staticmethod
    def _encode(anchors, gt):
        '''
        Faster R-CNN box parameterization: express gt relative to anchors.
        :param anchors: (M, 4), :param gt: (M, 4) both [x1,y1,x2,y2]
        :return: (M, 4) targets [tx, ty, tw, th]
        '''
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        ax = anchors[:, 0] + 0.5 * aw
        ay = anchors[:, 1] + 0.5 * ah
        gw = gt[:, 2] - gt[:, 0]
        gh = gt[:, 3] - gt[:, 1]
        gx = gt[:, 0] + 0.5 * gw
        gy = gt[:, 1] + 0.5 * gh
        tx = (gx - ax) / aw
        ty = (gy - ay) / ah
        tw = torch.log(gw / aw)
        th = torch.log(gh / ah)
        return torch.stack([tx, ty, tw, th], dim=1)

    @staticmethod
    def _decode(anchors, deltas):
        '''
        Inverse of _encode: apply predicted deltas to anchors to get boxes.
        tw/th are clamped to keep exp() from blowing up on early, noisy predictions.
        :return: (M, 4) boxes [x1,y1,x2,y2]
        '''
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        ax = anchors[:, 0] + 0.5 * aw
        ay = anchors[:, 1] + 0.5 * ah
        tx, ty, tw, th = deltas.unbind(1)
        tw = tw.clamp(max=4.135)        # log(1000/16), the torchvision default
        th = th.clamp(max=4.135)
        cx = tx * aw + ax
        cy = ty * ah + ay
        w = torch.exp(tw) * aw
        h = torch.exp(th) * ah
        return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                            cx + 0.5 * w, cy + 0.5 * h], dim=1)

    # ------------------------------------------------------------------ forward
    def forward(self, feat):
        '''
        :param feat: (B, C, H, W) feature map from the backbone
        :return: objectness (B, H*W*A), deltas (B, H*W*A, 4), anchors (H*W*A, 4)
        '''
        B, _, H, W = feat.shape
        t = self.relu(self.conv(feat))
        obj = self.cls_logits(t)                # (B, A, H, W)
        reg = self.bbox_deltas(t)               # (B, 4A, H, W)

        # flatten to (B, H*W*A, ...) in (y, x, a) order to match _grid_anchors
        obj = obj.permute(0, 2, 3, 1).reshape(B, -1)
        reg = reg.view(B, self.num_anchors, 4, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 4)
        anchors = self._grid_anchors(H, W, feat.device)
        return obj, reg, anchors

    # ------------------------------------------------------------------ training
    def _assign_single(self, anchors, gt):
        '''
        Label each anchor for one image: 1 = positive, 0 = negative, -1 = ignore.
        Positive if IoU with any gt > pos_iou_thr, or if it is the best-matching
        anchor for some gt (guarantees every gt gets at least one positive).
        :param anchors: (M, 4), :param gt: (G, 4)
        :return: labels (M,), matched_gt_idx (M,) (0 where no match, unused for neg)
        '''
        M = anchors.shape[0]
        if gt.numel() == 0:
            return anchors.new_full((M,), 0, dtype=torch.long), anchors.new_zeros(M, dtype=torch.long)

        iou = box_iou(anchors, gt)                          # (M, G)
        max_iou, matched = iou.max(dim=1)                   # best gt per anchor
        labels = anchors.new_full((M,), -1, dtype=torch.long)
        labels[max_iou < self.neg_iou_thr] = 0
        labels[max_iou >= self.pos_iou_thr] = 1

        # force the best anchor(s) for each gt to be positive
        gt_best = iou.max(dim=0).values                     # (G,)
        force = (iou == gt_best.unsqueeze(0)) & (iou > 0)   # (M, G)
        labels[force.any(dim=1)] = 1
        return labels, matched

    def _sample(self, labels):
        '''
        Subsample anchors to num_samples with at most pos_fraction positives, so
        the objectness loss is not swamped by the overwhelming negative majority.
        :param labels: (M,) from _assign_single
        :return: (pos_idx, neg_idx) index tensors into the anchor dimension
        '''
        pos = torch.nonzero(labels == 1).squeeze(1)
        neg = torch.nonzero(labels == 0).squeeze(1)
        num_pos = min(int(self.num_samples * self.pos_fraction), pos.numel())
        num_neg = min(self.num_samples - num_pos, neg.numel())
        pos = pos[torch.randperm(pos.numel(), device=pos.device)[:num_pos]]
        neg = neg[torch.randperm(neg.numel(), device=neg.device)[:num_neg]]
        return pos, neg

    def loss(self, objectness, deltas, anchors, gt_boxes):
        '''
        RPN multi-task loss averaged over the batch.
        :param objectness: (B, M) logits, :param deltas: (B, M, 4) predicted deltas
        :param anchors: (M, 4) shared across the batch
        :param gt_boxes: list (len B) of (G_i, 4) axis-aligned gt boxes
        :return: dict with 'rpn_cls', 'rpn_reg' and their sum 'rpn_loss'
        '''
        cls_losses, reg_losses = [], []
        for i, gt in enumerate(gt_boxes):
            labels, matched = self._assign_single(anchors, gt)
            pos, neg = self._sample(labels)
            samp = torch.cat([pos, neg])
            if samp.numel() == 0:
                continue

            # objectness: binary classification over the sampled anchors
            target = torch.zeros(samp.numel(), device=objectness.device)
            target[:pos.numel()] = 1.0
            cls_losses.append(nn.functional.binary_cross_entropy_with_logits(
                objectness[i, samp], target))

            # regression: only on positives, toward their matched gt
            if pos.numel() > 0:
                reg_t = self._encode(anchors[pos], gt[matched[pos]])
                reg_losses.append(nn.functional.smooth_l1_loss(
                    deltas[i, pos], reg_t, beta=1.0 / 9.0))

        device = objectness.device
        cls = torch.stack(cls_losses).mean() if cls_losses else torch.zeros((), device=device)
        reg = torch.stack(reg_losses).mean() if reg_losses else torch.zeros((), device=device)
        return {"rpn_cls": cls, "rpn_reg": reg, "rpn_loss": cls + reg}

    # ------------------------------------------------------- lightning training
    def _shared_step(self, batch):
        '''
        Loss over a batch of precomputed feature maps. Used only for standalone
        stage-1 training of the RPN; the Detector drives the end-to-end path.
        :param batch: (feature_map (B,C,H,W), gt_boxes list of (G_i,4) aabb)
        :return: (total_loss, loss_dict)
        '''
        feat, gt_boxes = batch
        objectness, deltas, anchors = self(feat)
        losses = self.loss(objectness, deltas, anchors, gt_boxes)
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
        '''All RPN parameters are trainable (the encoder lives in the Detector).'''
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def proposals(self, objectness, deltas, anchors, image_size):
        '''
        Turn dense predictions into a per-image shortlist of region proposals.
        This is the step where the fixed anchor set collapses to a variable
        number of boxes: score -> top-k -> decode -> clip -> drop tiny -> NMS.
        :param image_size: (H_img, W_img) in pixels, for clipping
        :return: list (len B) of dicts {'boxes': (P,4), 'scores': (P,)}
        '''
        B = objectness.shape[0]
        out = []
        for i in range(B):
            scores = torch.sigmoid(objectness[i])
            topk = min(self.pre_nms_topk, scores.numel())
            score_top, idx = scores.topk(topk)
            boxes = self._decode(anchors[idx], deltas[i, idx])
            boxes = clip_boxes_to_image(boxes, image_size)

            keep = remove_small_boxes(boxes, self.min_size)
            boxes, score_top = boxes[keep], score_top[keep]

            keep = nms(boxes, score_top, self.nms_thr)[:self.post_nms_topk]
            out.append({"boxes": boxes[keep], "scores": score_top[keep]})
        return out