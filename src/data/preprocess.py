import torch

from torch.nn.functional import unfold
from torchvision.transforms import v2

def makePatches(y:torch.Tensor, num_patches:int)-> torch.Tensor:
    '''
    Creates N non-overlapping patches from the input tensor
    :param y: BxCxHxW input tensor, image has to be squared
    :param num_patches: number of patches
    :return: BxCxNxhxw
    '''

    B, C, H, W = y.shape
    assert H == W # image is squared

    patch_size = H // num_patches
    patches = unfold(y, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)) # (B, C*patch_size*patch_size, N)
    patches = patches.transpose(1, 2) # (B, N, C*patch_size*patch_size)

    return patches

def scale_boxes(boxes: torch.Tensor, orig_size, new_size) -> torch.Tensor:
    '''
    boxes: (N, 8) in DOTA format (x1, y1, x2, y2, x3, y3, x4, y4)
    orig_size: (H, W)
    new_size:  (H, W)
    '''
    scale_x = new_size[1] / orig_size[1]
    scale_y = new_size[0] / orig_size[0]

    scaled = boxes.clone()
    scaled[:, 0::2] *= scale_x
    scaled[:, 1::2] *= scale_y
    return scaled

def get_im_transforms(new_size) -> v2.Compose:
    '''
    Building image transforms
    :param new_size:
    :return:
    '''
    image_transform = v2.Compose([
        v2.Resize(new_size),
        v2.ToDtype(torch.float32, scale=True),  # replaces ToTensor() for tensors
    ])

    return image_transform
