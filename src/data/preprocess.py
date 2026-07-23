import torch

from torchvision.transforms import v2

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
