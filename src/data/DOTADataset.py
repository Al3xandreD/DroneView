import os
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch import tensor, float32
from torchvision.io import decode_image
from torchvision.io import ImageReadMode

from src.utils.paths import DATA
from src.data.preprocess import get_im_transforms, scale_boxes

class DOTADataset(Dataset):
    '''
    Creates a torch Dataset object based on DOTA dataset.
    '''
    def __init__(self, annotation_path, images_dir, new_size):

        self.annotation_path = annotation_path
        self.images_dir = images_dir
        self.new_size = new_size

        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith(('.png', '.jpg', '.tif'))
        ])

    def __getitem__(self, index):
        # image
        image_name = self.image_files[index]
        image_path = os.path.join(self.images_dir, image_name)
        image = decode_image(image_path, mode=ImageReadMode.RGB)

        # annotation
        annotation_name = os.path.splitext(image_name)[0] + '.txt'
        annotation_path = os.path.join(self.annotation_path, annotation_name)

        boxes, labels, difficulties = parse_annotation(annotation_path)
        boxes = tensor(boxes, dtype=float32).reshape(-1, 8) # reshape keeps images with no annotations as (0, 8) instead of (0,)


        # transforms
        orig_size = image.size()[1:]    # discarding channels in the size
        image_transform = get_im_transforms(self.new_size)

        image = image_transform(image)   # image transform
        boxes = scale_boxes(boxes, orig_size, self.new_size)  # boxe transform

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.image_files)


def parse_annotation(annotation_path):
    boxes, labels, difficulties = [], [], []

    with open(annotation_path, 'r') as f:   # opening file
        lines = f.readlines()

    for line in lines:  # discarding metadata
        if line.startswith('imagesource') or line.startswith('gsd'):
            continue

        parts = line.strip().split()
        if len(parts) < 10:
            continue

        # 8 coordinates + class + difficulty
        coords = list(map(float, parts[:8]))
        label = parts[8]
        difficulty = int(parts[9])

        boxes.append(coords)
        labels.append(label)
        difficulties.append(difficulty)

    return boxes, labels, difficulties

if __name__ == '__main__':
    annotation_path = DATA / "Train/labelTxt-v1.5/DOTA-v1.5_train"
    images_dir = DATA / "Train/images"
    image_size = (1024, 1024)

    dataset = DOTADataset(annotation_path, images_dir, (500, 500))

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_images, train_boxes, train_labels, train_difficulties = next(iter(dataloader))
    plt.imshow(train_images[0].permute(1, 2, 0).numpy())
    plt.show()

