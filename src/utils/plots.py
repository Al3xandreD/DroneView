import torch
from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes


def plot_boxes(image, boxes, label):
    image = draw_bounding_boxes(image, boxes[0], labels=label)  # , labels=train_labels[0])
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(f"Train Image using ground truth bounding boxes")
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



