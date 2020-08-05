# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

import os

import numpy as np
from PIL import Image

from base import BaseDataSet, BaseDataLoader
from utils import palette


class CDDataset(BaseDataSet):

    def __init__(self, **kwargs):
        self.num_classes = 8
        self.palette = palette.CD_palette
        super(CDDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'data')
        self.image_dir = os.path.join(self.root, 'images')
        self.label_dir = os.path.join(self.root, 'labels')

        file_list = os.path.join(self.root, "segmentation", self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.uint8)

        return image, label, image_id


class CD(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False,
                 shuffle=True, flip=True, rotate=True, blur=True, augment=True, val_split=None, return_id=False):
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]  # probably Imagenet normalization, need if pretrained

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = CDDataset(**kwargs)
        super(CD, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
