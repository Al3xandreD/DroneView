import torch
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from torchvision.utils import draw_bounding_boxes

from src.utils.utils_training import select_detections, selection_desc


def plot_boxes(image, boxes, label):
    image = draw_bounding_boxes(image, boxes[0], labels=label)  # , labels=train_labels[0])
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(f"Train Image using ground truth bounding boxes")
    plt.show()

def plot_detections(image, detection, class_names, score_thr=None, top_k=None,
                    title=None, max_labels=25):
    '''
    Draw the Detector's output for a single image: axis-aligned boxes colored per
    class.

    The detector returns a variable number of boxes (the RPN proposes as many
    regions as survive its objectness top-k and NMS), so nothing here assumes a
    fixed count. Detections arrive already scored, suppressed and score-ordered by
    ROIHead.postprocess; select_detections only applies the optional *display*
    cut, and by default nothing is hidden.

    Two concessions to crowded frames: colors are per class so overlapping
    categories stay distinguishable, and once more than max_labels boxes are drawn
    the text tags are dropped (the boxes all stay) rather than silently dropping
    detections to keep the frame readable.
    :param image: (C, H, W) float tensor in [0, 1], as produced by the transforms
    :param detection: dict {'boxes': (K,4), 'scores': (K,), 'labels': (K,)}
    :param class_names: sequence indexed by the label ids
    :param score_thr: display score floor, or None to draw the full tail
    :param top_k: draw at most this many highest-scoring detections, or None
    :param max_labels: above this many boxes, draw them untagged
    '''
    image = (image.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    boxes, scores, labels, n_returned = select_detections(detection, score_thr, top_k)
    desc = selection_desc(score_thr, top_k)

    if boxes.numel() == 0:
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.title(title or f"no detections drawn of {n_returned} returned ({desc})")
    else:
        # stable per-class colors: same class -> same color across images
        cmap = plt.get_cmap('tab20')
        colors = [to_hex(cmap(int(l) % 20)) for l in labels.tolist()]

        tags = None
        if boxes.shape[0] <= max_labels:
            tags = [f"{class_names[l]} {s:.2f}"
                    for l, s in zip(labels.tolist(), scores.tolist())]

        drawn = draw_bounding_boxes(image, boxes, labels=tags, colors=colors, width=2)
        plt.imshow(drawn.permute(1, 2, 0).numpy())
        plt.title(title or f"{boxes.shape[0]} of {n_returned} detections "
                           f"({desc}, {scores.min():.2f}-{scores.max():.2f})")
    plt.axis('off')
    plt.show()


def plot_representations(representation):
    '''
    Visualize the per-patch target-encoder representation of a single image.
    Shows three panels:
      1. the raw (N, D) patch-token matrix as a heatmap,
      2. the per-patch feature norm laid back on the patch grid,
      3. a PCA(3)-of-patches map rendered as RGB on the patch grid, which exposes
         the coarse semantic structure the encoder captures.
    :param representation: (1, N, D) or (N, D) tensor of patch tokens
    '''
    representation = torch.as_tensor(representation).detach().cpu().float()
    if representation.dim() == 3:
        representation = representation[0]           # (N, D)
    N, D = representation.shape
    g = int(round(N ** 0.5))                         # patch-grid side (14 for 224/16)

    norms = representation.norm(dim=1)               # (N,) per-patch magnitude

    # PCA to 3 components -> normalized to [0, 1] for an RGB spatial map
    centered = representation - representation.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(centered, q=3)
    pcs = centered @ V[:, :3]                         # (N, 3)
    pcs = (pcs - pcs.min(0).values) / (pcs.max(0).values - pcs.min(0).values + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(representation.numpy(), aspect='auto', cmap='viridis')
    axes[0].set_title(f"Patch tokens (N={N}, D={D})")
    axes[0].set_xlabel("feature dim"); axes[0].set_ylabel("patch")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    if g * g == N:
        im1 = axes[1].imshow(norms.reshape(g, g).numpy(), cmap='magma')
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        axes[2].imshow(pcs.reshape(g, g, 3).numpy())
    else:
        # non-square patch count: fall back to 1-D views
        axes[1].plot(norms.numpy())
        axes[2].plot(pcs.numpy())
    axes[1].set_title("Per-patch feature norm")
    axes[2].set_title("PCA(3) of patches as RGB")
    for ax in (axes[1], axes[2]):
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.show()



