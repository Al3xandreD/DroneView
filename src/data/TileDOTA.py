import torch
from matplotlib import pyplot as plt
from torch import  nn
from torch.nn import Unfold
from torch.utils.data import DataLoader

from src.data.DOTADataset import DOTADataset
from src.utils.paths import DATA


class Tiler(nn.Module):
    '''
    Creates a tiler model to tile the dota dataset
    '''

    def __init__(self, tile_size=(224, 224)):
        super().__init__()
        self.patch_size = tile_size
        self.unfold = Unfold(kernel_size=tile_size, stride=tile_size)


    def forward(self, image):
        B, C, H, W = image.shape
        image = self.unfold(image)  # (B, CxPxP, L) # tiling the image
        image = image.view(B, C, self.patch_size[0], self.patch_size[1], -1)  # (B, C, P, P, L)
        image = image.permute(0, 1, 4, 2, 3) # (B, C, L, P, P)

        return image.contiguous()


if __name__ == '__main__':
    annotation_path = DATA / "Train/labelTxt-v1.5/DOTA-v1.5_train"
    images_dir = DATA / "Train/images"
    image_size = (1024, 1024)
    patch_size = (250, 250)

    tiler = Tiler(patch_size)
    dataset = DOTADataset(annotation_path, images_dir, (500, 500))

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_images, train_boxes, train_labels, train_difficulties = next(iter(dataloader))
    tiled_images = tiler(train_images)

    image1 = tiled_images[0, :,0, :, :]
    image1 = (image1.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    plt.imshow(image1.permute(1, 2, 0).numpy())
    plt.show()

    image2 = tiled_images[0, :, 1, :, :]
    image2 = (image2.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    plt.imshow(image2.permute(1, 2, 0).numpy())
    plt.show()

    image3 = tiled_images[0, :, 2, :, :]
    image3 = (image3.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    plt.imshow(image3.permute(1, 2, 0).numpy())
    plt.show()

    image4 = tiled_images[0, :, 3, :, :]
    image4 = (image4.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    plt.imshow(image4.permute(1, 2, 0).numpy())
    plt.show()
