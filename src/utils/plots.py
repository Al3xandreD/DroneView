from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes


def plot_boxes(image, boxes):
    image = draw_bounding_boxes(image, boxes[0])  # , labels=train_labels[0])
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()