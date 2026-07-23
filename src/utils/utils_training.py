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
