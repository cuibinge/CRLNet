import torch
from torch.utils.data import Dataset
import imageio
import glob
import os


class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, Transform=None):
        self.images_path_list = glob.glob(os.path.join(images_path, '*.tif'))
        self.labels_path_list = glob.glob(os.path.join(labels_path, '*.tif'))

    def __getitem__(self, index):
        image_path = self.images_path_list[index]
        label_path = self.labels_path_list[index]

        image = imageio.imread(image_path)
        label = imageio.imread(label_path)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        image = image.permute(2,0,1)

        label = torch.squeeze(label, 0)

        return image, label

    def __len__(self):
        return len(self.images_path_list)