import torch

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


def selection_desc(score_thr=None, top_k=None):
    '''
    Human-readable description of a select_detections rule, for titles and logs.
    '''
    parts = []
    if score_thr is not None:
        parts.append(f"score >= {score_thr}")
    if top_k is not None:
        parts.append(f"top {top_k}")
    return ", ".join(parts) if parts else "all returned"


def select_detections(detection, score_thr=None, top_k=None):
    '''
    Narrow one image's detections down for *display*, not for detection.

    ROIHead.postprocess has already done the detection work: score floor, class-wise
    NMS, and the detections_per_img cap. Nothing is re-suppressed or re-ranked here
    — batched_nms returns its indices in descending score order, so the incoming
    detections are already sorted and this only ever slices a prefix or masks rows.

    What remains is a presentation problem. The head's score_thr is deliberately
    permissive (0.05) because mAP integrates over the whole precision/recall curve
    and needs the low-confidence tail to measure recall; plotting that same tail
    puts up to detections_per_img boxes on one frame, most of them noise. So a
    display cut is a separate knob from the model's threshold, serving a different
    purpose, and it is rank-based because the detection count is variable: the RPN
    proposes however many regions survive its own top-k and NMS, so K differs per
    image (a frame packed with small-vehicles yields far more than an empty sea
    tile) and there is no fixed slot count to assume.

    With neither argument set, every returned detection is kept — the honest
    "however many the detector found" view.
    :param detection: dict {'boxes': (K,4), 'scores': (K,), 'labels': (K,)}
    :param score_thr: display score floor, or None to keep the full tail
    :param top_k: keep at most this many highest-scoring detections, or None
    :return: (boxes (S,4), scores (S,), labels (S,), n_returned) on cpu
    '''
    boxes = detection['boxes'].detach().cpu()
    scores = detection['scores'].detach().cpu()
    labels = detection['labels'].detach().cpu()
    n_returned = int(scores.numel())

    if score_thr is not None:
        keep = scores >= score_thr
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if top_k is not None:
        # a prefix is the top-k by score, the order postprocess already produced
        boxes, scores, labels = boxes[:top_k], scores[:top_k], labels[:top_k]

    return boxes, scores, labels, n_returned
