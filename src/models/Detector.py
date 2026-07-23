from src.models.I_JEPA import IJEPA
from src.utils.utils_training import poly8_to_aabb

import lightning as L
import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms, box_iou, clip_boxes_to_image, roi_align, remove_small_boxes, nms

# DOTA-v1.5 category names, in the order used as class indices. Index 0 of the
# classifier is reserved for background, so class c here is logit c + 1.
DOTA_CLASSES = (
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool", "container-crane",
)


class ROIHead(nn.Module):
    '''
    Second stage: classify each RPN proposal and refine its box.

    Where the RPN answers "is there *something* here", this head answers "what
    is it, and where exactly". Each proposal is pooled from the shared feature
    map into a fixed pool_size x pool_size patch (so a variable number of
    variably-sized boxes becomes a fixed-width tensor an MLP can consume),
    flattened, and pushed through a two-layer MLP with two sibling linear heads:
      - cls_score: (C + 1) logits, background at index 0
      - bbox_pred: 4 deltas *per class*, so the refinement can be
        category-specific (a harbor and a small-vehicle are not adjusted alike)

    Box deltas use the same Faster R-CNN parameterization as the RPN, so
    RPN._encode / RPN._decode are reused directly.
    '''

    def __init__(self, in_channels, num_classes, spatial_scale,
                 pool_size=7, mid_channels=1024,
                 # training-time proposal assignment / sampling
                 fg_iou_thr=0.5, num_samples=512, pos_fraction=0.25,
                 # inference-time filtering
                 score_thr=0.05, nms_thr=0.5, detections_per_img=100):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_scale = spatial_scale      # feature-map pixels per image pixel (1 / stride)
        self.pool_size = pool_size
        self.fg_iou_thr = fg_iou_thr
        self.num_samples = num_samples
        self.pos_fraction = pos_fraction
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.detections_per_img = detections_per_img

        self.mlp = nn.Sequential(
            nn.Linear(in_channels * pool_size * pool_size, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(mid_channels, num_classes + 1)
        self.bbox_pred = nn.Linear(mid_channels, (num_classes + 1) * 4)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0.0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0.0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0.0)

        # Standard Faster R-CNN target rescaling. The second stage regresses much
        # smaller corrections than the RPN (proposals already overlap the object),
        # so targets are inflated to keep the regression loss on a useful scale.
        self.register_buffer("bbox_weights", torch.tensor([10.0, 10.0, 5.0, 5.0]),
                             persistent=False)

    def forward(self, feat, proposals):
        '''
        :param feat: (B, C, H, W) shared feature map
        :param proposals: list (len B) of (P_i, 4) boxes in image pixel coords
        :return: cls_logits (sum P_i, C+1), deltas (sum P_i, (C+1)*4)
                 rows concatenated in image order
        '''
        rois = roi_align(feat, proposals, output_size=self.pool_size,
                         spatial_scale=self.spatial_scale, sampling_ratio=2,
                         aligned=True)                       # (sum P_i, C, pool, pool)
        x = self.mlp(rois.flatten(1))
        return self.cls_score(x), self.bbox_pred(x)

    # ------------------------------------------------------------------ training
    def select_training_samples(self, proposals, gt_boxes, gt_labels):
        '''
        For one image: label proposals against the ground truth and subsample
        them so the batch is not almost entirely background.

        A proposal is foreground if its best IoU with any gt reaches fg_iou_thr,
        and inherits that gt's class (+1, background occupying index 0);
        everything else is background. Ground-truth boxes are expected to have
        been appended to the proposals by the caller, which guarantees a few
        high-quality positives even while the RPN is still poor.
        :return: (sampled_boxes (S,4), labels (S,), reg_targets (S,4))
        '''
        device = proposals.device
        if gt_boxes.numel() == 0:
            labels = torch.zeros(proposals.shape[0], dtype=torch.long, device=device)
            reg_targets = torch.zeros((proposals.shape[0], 4), device=device)
        else:
            iou = box_iou(proposals, gt_boxes)                  # (P, G)
            max_iou, matched = iou.max(dim=1)                   # best gt per proposal
            labels = gt_labels[matched] + 1                     # shift for background
            labels[max_iou < self.fg_iou_thr] = 0
            reg_targets = RPN._encode(proposals, gt_boxes[matched]) * self.bbox_weights

        pos = torch.nonzero(labels > 0).squeeze(1)
        neg = torch.nonzero(labels == 0).squeeze(1)
        num_pos = min(int(self.num_samples * self.pos_fraction), pos.numel())
        num_neg = min(self.num_samples - num_pos, neg.numel())
        pos = pos[torch.randperm(pos.numel(), device=device)[:num_pos]]
        neg = neg[torch.randperm(neg.numel(), device=device)[:num_neg]]
        keep = torch.cat([pos, neg])

        return proposals[keep], labels[keep], reg_targets[keep]

    def loss(self, cls_logits, deltas, labels, reg_targets):
        '''
        Fast R-CNN multi-task loss over the sampled proposals of the whole batch.
        Classification is (C+1)-way softmax over every sample; regression is
        smooth L1 on foreground samples only, taking the 4 deltas belonging to
        the sample's own class. Both are normalized by the total sample count,
        so an image with few positives does not get an inflated per-box weight.
        :return: dict with 'roi_cls' and 'roi_reg'
        '''
        device = cls_logits.device
        if labels.numel() == 0:
            zero = torch.zeros((), device=device)
            return {"roi_cls": zero, "roi_reg": zero}

        cls = nn.functional.cross_entropy(cls_logits, labels)

        pos = torch.nonzero(labels > 0).squeeze(1)
        if pos.numel() == 0:
            reg = torch.zeros((), device=device)
        else:
            # pick, for each positive sample, the 4 deltas of its ground-truth class
            pred = deltas.view(-1, self.num_classes + 1, 4)[pos, labels[pos]]
            reg = nn.functional.smooth_l1_loss(
                pred, reg_targets[pos], beta=1.0 / 9.0, reduction="sum") / labels.numel()

        return {"roi_cls": cls, "roi_reg": reg}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def postprocess(self, cls_logits, deltas, proposals, image_size):
        '''
        Turn per-proposal logits into final detections, per image:
        softmax -> per-class decode -> clip -> score threshold -> class-wise NMS.
        :param proposals: list (len B) of (P_i, 4), same ones fed to forward()
        :return: list (len B) of {'boxes': (K,4), 'scores': (K,), 'labels': (K,)}
                 with labels in [0, C) indexing DOTA_CLASSES
        '''
        counts = [p.shape[0] for p in proposals]
        scores_all = torch.softmax(cls_logits, dim=-1).split(counts)
        deltas_all = deltas.split(counts)

        out = []
        for boxes_i, scores_i, deltas_i in zip(proposals, scores_all, deltas_all):
            P = boxes_i.shape[0]
            if P == 0:
                out.append({"boxes": boxes_i.new_zeros((0, 4)),
                            "scores": boxes_i.new_zeros((0,)),
                            "labels": boxes_i.new_zeros((0,), dtype=torch.long)})
                continue

            # decode every class's deltas against the same proposal box
            rep = boxes_i.unsqueeze(1).expand(-1, self.num_classes + 1, 4).reshape(-1, 4)
            d = (deltas_i.view(-1, 4) / self.bbox_weights)
            boxes = RPN._decode(rep, d).view(P, self.num_classes + 1, 4)
            boxes = clip_boxes_to_image(boxes.reshape(-1, 4), image_size).view(
                P, self.num_classes + 1, 4)

            # drop the background column, then flatten to one row per (box, class)
            boxes = boxes[:, 1:].reshape(-1, 4)
            scores = scores_i[:, 1:].reshape(-1)
            labels = torch.arange(self.num_classes, device=boxes.device).repeat(P)

            keep = scores > self.score_thr
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = batched_nms(boxes, scores, labels, self.nms_thr)[:self.detections_per_img]
            out.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        return out


class Detector(L.LightningModule):
    '''
    Two-stage detector head on top of a frozen I-JEPA backbone.

    The I-JEPA target encoder is reused as a frozen feature extractor: an image
    is turned into N = H*W per-patch tokens which, being produced in row-major
    order, reshape back into a (B, D, H, W) spatial feature map. A Region
    Proposal Network (RPN) slides over that map and, at every location x anchor,
    scores objectness and regresses a box refinement. The surviving proposals are
    pooled from the same map and sent to an ROI head that assigns a DOTA class
    and a class-specific box refinement.

    Scope: both stages are axis-aligned. DOTA's oriented boxes are reduced to
    their enclosing horizontal boxes for training targets; recovering the
    orientation (an extra angle regression on the ROI head) is not built here.
    '''

    def __init__(self, path2ijepa, lr=1e-3, num_classes=len(DOTA_CLASSES),
                 roi_pool_size=7, roi_mid_channels=1024):
        super(Detector, self).__init__()
        # path2ijepa is stored as a hyperparameter so the detector can be rebuilt
        # from a checkpoint; the backbone weights themselves are saved with it.
        self.save_hyperparameters()

        self.lr = lr
        self.num_classes = num_classes
        self.class_to_idx = {name: i for i, name in enumerate(DOTA_CLASSES[:num_classes])}

        # backbone: frozen I-JEPA target encoder (the stable EMA representation)
        ijepa = IJEPA.load_from_checkpoint(path2ijepa)
        self.target_encoder = ijepa.target_encoder.eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False    # no grads

        D = self.target_encoder.hidden_dim              # per-token feature dimension (768)
        self.image_size = int(self.target_encoder.image_size)   # 224
        self.patch_size = int(self.target_encoder.patch_size)   # 16
        self.grid = self.image_size // self.patch_size          # 14 tokens per side

        # stage 1: RPN over the ViT feature grid. Its stride (in image pixels) is
        # the patch size, since each token summarizes one patch.
        self.rpn = RPN(in_channels=D, stride=self.patch_size)

        # stage 2: ROI classification + box refinement on the same feature map
        self.roi_head = ROIHead(in_channels=D, num_classes=num_classes,
                                spatial_scale=1.0 / self.patch_size,
                                pool_size=roi_pool_size, mid_channels=roi_mid_channels)

        # COCO-style mAP, accumulated across the whole test epoch (a per-batch
        # value would be meaningless: AP is an area under a precision/recall
        # curve that only exists once every prediction has been ranked together).
        self.test_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                             class_metrics=True)

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

    def _prepare_targets(self, boxes, labels, device):
        '''
        Convert one batch of raw annotations into ROI-head targets: oriented
        8-coord boxes become axis-aligned, class-name strings become indices.
        :param boxes: list (len B) of (G_i, 8), :param labels: list (len B) of
                      lists of DOTA class-name strings
        :return: (gt_boxes list of (G_i,4), gt_labels list of (G_i,) long)
        '''
        gt_boxes = [poly8_to_aabb(b.to(device)) for b in boxes]
        gt_labels = []
        for names in labels:
            unknown = {n for n in names if n not in self.class_to_idx}
            if unknown:
                raise ValueError(f"annotation contains classes outside DOTA_CLASSES: "
                                 f"{sorted(unknown)}")
            gt_labels.append(torch.tensor([self.class_to_idx[n] for n in names],
                                          dtype=torch.long, device=device))
        return gt_boxes, gt_labels

    def forward(self, images):
        '''
        :param images: (B, C, H, W)
        :return: list (len B) of detection dicts
                 {'boxes': (K,4), 'scores': (K,), 'labels': (K,)} in image pixel
                 coordinates (axis-aligned), labels indexing DOTA_CLASSES.
        '''
        image_size = (self.image_size, self.image_size)
        feat = self._extract_feature_map(images)                 # (B, D, H, W)
        objectness, deltas, anchors = self.rpn(feat)
        proposals = [p["boxes"] for p in
                     self.rpn.proposals(objectness, deltas, anchors, image_size)]

        cls_logits, roi_deltas = self.roi_head(feat, proposals)
        return self.roi_head.postprocess(cls_logits, roi_deltas, proposals, image_size)

    def _shared_step(self, batch, detect=False):
        '''
        Full two-stage loss shared by train/val/test: the RPN's objectness +
        box-regression terms, plus the ROI head's classification + box-refinement
        terms. The RPN proposals feeding stage 2 are generated under no_grad, so
        the two stages are trained jointly but the ROI gradients do not flow back
        into the proposal boxes (the usual Faster R-CNN approximation).
        :param batch: (images, boxes, labels, difficulties) from collate_fn
        :param detect: also run inference-time postprocessing, reusing the
                       feature map and proposals already computed here so
                       evaluation costs one extra ROI pass, not a second forward
        :return: (total_loss, loss_dict, detections|None, targets|None)
        '''
        images, boxes, labels, _ = batch
        image_size = (self.image_size, self.image_size)
        feat = self._extract_feature_map(images)                 # (B, D, H, W)
        gt_boxes, gt_labels = self._prepare_targets(boxes, labels, images.device)

        # ---- stage 1
        objectness, deltas, anchors = self.rpn(feat)
        losses = self.rpn.loss(objectness, deltas, anchors, gt_boxes)

        # ---- stage 2
        proposals = [p["boxes"] for p in
                     self.rpn.proposals(objectness, deltas, anchors, image_size)]
        # append the ground truth so the head sees clean positives from step one
        train_proposals = [torch.cat([p, g]) for p, g in zip(proposals, gt_boxes)]

        sampled, samp_labels, samp_targets = [], [], []
        for p, g, l in zip(train_proposals, gt_boxes, gt_labels):
            s, lab, tgt = self.roi_head.select_training_samples(p, g, l)
            sampled.append(s)
            samp_labels.append(lab)
            samp_targets.append(tgt)

        cls_logits, roi_deltas = self.roi_head(feat, sampled)
        losses.update(self.roi_head.loss(cls_logits, roi_deltas,
                                         torch.cat(samp_labels), torch.cat(samp_targets)))

        total = losses["rpn_loss"] + losses["roi_cls"] + losses["roi_reg"]

        detections, targets = None, None
        if detect:
            # the un-augmented proposals: the gt boxes appended above are a
            # training-only crutch and would leak perfect boxes into the metric
            with torch.no_grad():
                c, d = self.roi_head(feat, proposals)
                detections = self.roi_head.postprocess(c, d, proposals, image_size)
            targets = [{"boxes": b, "labels": l} for b, l in zip(gt_boxes, gt_labels)]

        return total, losses, detections, targets

    def training_step(self, batch, batch_idx):
        loss, parts, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"train_{k}": v for k, v in parts.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, parts, _, _ = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict({f"val_{k}": v for k, v in parts.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, parts, detections, targets = self._shared_step(batch, detect=True)
        self.log("test_loss", loss, prog_bar=True)
        self.log_dict({f"test_{k}": v for k, v in parts.items()})
        self.test_map.update(detections, targets)
        return loss

    def on_test_epoch_end(self):
        '''
        Compute the accumulated detection metrics once every test batch has been
        seen, log the scalars, and break out the per-class APs under readable
        names. torchmetrics reports -1 for a class with no ground truth in the
        split, so those are dropped rather than logged as a real score.
        '''
        res = self.test_map.compute()
        present = res.pop("classes", torch.zeros(0, dtype=torch.long))
        map_per_class = res.pop("map_per_class", None)
        mar_per_class = res.pop("mar_100_per_class", None)

        self.log_dict({f"test_{k}": v.float() for k, v in res.items()
                       if torch.is_tensor(v) and v.numel() == 1})

        for name, values in (("ap", map_per_class), ("ar100", mar_per_class)):
            if values is None or not torch.is_tensor(values) or values.numel() == 0:
                continue
            for cls_id, value in zip(present.tolist(), values.tolist()):
                if value >= 0:
                    self.log(f"test_{name}_{DOTA_CLASSES[cls_id]}", value)

        self.test_map.reset()

    def configure_optimizers(self):
        '''
        Only the heads are trained; the backbone is frozen, so its parameters are
        excluded from the optimizer.
        :return:
        '''
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr)


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
        :param anchors: (M, 4),
        :param gt: (M, 4) both [x1,y1,x2,y2]
        :return: (M, 4) targets [tx, ty, tw, th]
        '''
        aw = anchors[:, 2] - anchors[:, 0]  # anchor width
        ah = anchors[:, 3] - anchors[:, 1]  # --- height
        ax = anchors[:, 0] + 0.5 * aw   # anchor center coordinates
        ay = anchors[:, 1] + 0.5 * ah
        gw = gt[:, 2] - gt[:, 0]    # gt width
        gh = gt[:, 3] - gt[:, 1]    # --- height
        gx = gt[:, 0] + 0.5 * gw    # gt center coordinates
        gy = gt[:, 1] + 0.5 * gh
        tx = (gx - ax) / aw # center offsets expressed in units of anchor size
        ty = (gy - ay) / ah
        tw = torch.log(gw / aw) # log of size ratio
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
        obj = self.cls_logits(t)                # (B, A, H, W), objectness score
        reg = self.bbox_deltas(t)               # (B, 4A, H, W), bbox deltas regression

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

        iou = box_iou(anchors, gt)                          # (M, G), iou score between anchors and gt
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
            score_top, idx = scores.topk(topk) # keeping the topk scores
            boxes = self._decode(anchors[idx], deltas[i, idx])  # passing back to box fomat
            boxes = clip_boxes_to_image(boxes, image_size) # clipping images to fit in image

            # removing boxes to small
            keep = remove_small_boxes(boxes, self.min_size)
            boxes, score_top = boxes[keep], score_top[keep]

            keep = nms(boxes, score_top, self.nms_thr)[:self.post_nms_topk] # non-maximum suppression
            out.append({"boxes": boxes[keep], "scores": score_top[keep]})
        return out
