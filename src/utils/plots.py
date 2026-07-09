from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes


def plot_boxes(image, boxes, label):
    image = draw_bounding_boxes(image, boxes[0], labels=label)  # , labels=train_labels[0])
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(f"Train Image using ground truth bounding boxes")
    plt.show()

def plot_representations(predictions, index):
    '''
    Plotting the representation out of the target encoder
    :param predictions: (1, N, D)
    :param index: index of the patch representation to plot
    :return:
    '''
    predictions = predictions.cpu().detach().numpy() # (1, N, D)
    plt.plot(predictions[:, index], label="Prediction")
    plt.show()



