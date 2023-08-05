import torch

from config import Config

from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import os


class Transform:

    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Grayscale()
        ])

    def __call__(self, img):
        img = Image.open(img)
        img = self.data_transform(img)
        img = np.array(img)
        img = torch.from_numpy(img)

        return img


class Dataset(data.Dataset):

    def __init__(self):
        super(Dataset, self).__init__()

        self.dataset_path = Config.dataset_path

        self.transform = Transform()

        self.images = []
        self.labels = []

        self.get_data_from_dir()

        ex_img = Image.open(self.images[0])
        print(ex_img.size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.transform(self.images[idx])
        lb = self.labels[idx]

        return img, lb

    def get_data_from_dir(self):

        data_files = os.listdir(self.dataset_path)  # normal, dos, fuzzy, ...

        for data_file in data_files:

            path_to_data = os.path.join(self.dataset_path, data_file)  # ../dataset/CHD/id_image/fuzzy
            filenames = os.listdir(path_to_data)  # normal_0.png, abnormal_1.png, ...

            for file in filenames:  # normal_0.png

                full_path = os.path.join(path_to_data, file)  # # ../dataset/CHD/id_image/fuzzy/normal_0.png

                label = file.split('_')[0]  # 'normal'
                if label == 'normal':
                    label = 1
                elif label == 'abnormal':
                    label = 0

                self.labels.append(label)
                self.images.append(full_path)
