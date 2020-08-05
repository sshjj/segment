import os
import random
from pathlib import Path
import numpy as np
import skimage.external.tifffile as tifffile
from PIL import Image
from base import BaseDataSet, BaseDataLoader
from utils import palette

global channels
import json

global num_classes
with open("config.json") as file:
    config = json.load(file)
channels = config['arch']['args']["in_channels"]
num_classes = config['num_classes']


class XADataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = num_classes
        self.palette_label2int = list([[0, 0, 0], [150, 250, 0], [0, 250, 0], [0, 100, 0],
                                       [200, 0, 0], [255, 255, 255], [0, 0, 200], [0, 150, 250]])
        self.palette = palette.CD_palette
        super(XADataset, self).__init__(**kwargs)

    def _set_files(self):
        # set path(numpy) set list
        self.root = os.path.join(self.root, 'data')
        # print(self.channels)
        # print(channels)
        if channels == 4:
            self.image_dir = os.path.join(self.root, 'images')
            self.savePath_lbl = Path(r'{}/labels_xa_2020_4'.format(self.root))
            self.savePath_img = Path(r'{}/images_xa_2020_4'.format(self.root))
        elif channels == 3:
            self.image_dir = os.path.join(self.root, 'images_jpg')
            self.savePath_lbl = Path(r'{}/labels_xa_2020_3'.format(self.root))
            self.savePath_img = Path(r'{}/images_xa_2020_3'.format(self.root))
        self.savePath_lbl.mkdir(exist_ok=True)
        self.savePath_img.mkdir(exist_ok=True)
        self.need_to_cut = self.need_to_cut
        self.label_dir = os.path.join(self.root, 'labels')

        self.savePath_img.mkdir(exist_ok=True)
        self.savePath_lbl.mkdir(exist_ok=True)

        if self.val:
            file_list = os.path.join(self.root, "segmentation", self.split + ".txt")
            self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
        else:
            if self.need_to_cut:
                print("generate train_dataset and val_dataset....")
                self.preprocess()
                self.gentxt()
            file_list = os.path.join(self.root, "segmentation", self.split + ".txt")
            self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        # read 小图(numpy)
        # print(os.path.splitext(self.files[index]))
        image_id = "".join(os.path.splitext(self.files[index]))
        # print(image_id)
        image_path = os.path.join(str(self.savePath_img), image_id + '.npy')
        label_path = os.path.join(str(self.savePath_lbl), image_id + '.npy')
        image = np.load(image_path)
        label = np.load(label_path)

        return image, label, image_id

    def preprocess(self):
        picPath_img = self.image_dir
        picPath_lbl = self.label_dir

        imagelist = os.listdir(picPath_img)
        labellist = os.listdir(picPath_lbl)
        for image_name in imagelist:
            im_name = os.path.basename(image_name)
            # ityp = im_name[-4:]
            print('image_name', im_name)
            self.cut_util('image', os.path.join(str(picPath_img), im_name), self.savePath_img)
        for lab_name in labellist:
            lable_name = os.path.basename(lab_name)
            # ltyp = lable_name[-4:]
            self.cut_util('label', os.path.join(str(picPath_lbl), lable_name), self.savePath_lbl)

    def label2intarray(self, label):
        """

        :param trans_palette:
        :param label: the ndarray needs to be transfer into int-element-array
        :return:
        """
        if len(label.shape) == 2:
            return label

        h, w, c = label.shape
        label_int = list(np.zeros(label.shape[:2], dtype=np.uint8))
        label.tolist()
        error = set()
        for i in range(h):
            for j in range(w):
                try:
                    idx = self.palette_label2int.index(label[i][j].tolist())
                    label_int[i][j] = idx
                except KeyError:
                    error.add(tuple(label[i][j]))
        return np.array(label_int)

    def cut_util(self, tag, base_name, save_dir):
        """

        :param tag: 标记label还是image
        :param base_name: 图片路径
        :param save_dir: 保存地址
        将label转成1通道灰度label图
        将3通道或者4通道大图切割成小图并保存成.npy形式
        """
        SubImgWidth = 512
        SubImgHeight = 512
        img_name = os.path.basename(base_name)
        im_name = os.path.splitext(img_name)[0]
        print(im_name)

        if "label" in tag:
            im = Image.open(base_name)
            im = np.array(im)
            im = self.label2intarray(im)
            im_height, im_width = im.shape

            for i in range(im_height // SubImgHeight):
                for j in range(im_width // SubImgWidth):
                    save_name = 'Hid{}_Wid{}_'.format(i, j) + im_name + '.npy'
                    crop = im[i * SubImgHeight:(i + 1) * SubImgHeight, j * SubImgWidth:(j + 1) * SubImgWidth]
                    np.save(os.path.join(str(save_dir), save_name), crop)

        elif "image" in tag:
            # print("base_name.{}".format(base_name))
            if base_name.endswith(".tiff") or base_name.endswith(".tif"):
                im = tifffile.imread(base_name)
            elif base_name.endswith(".jpg"):
                im = Image.open(base_name)
            else:
                raise FileNotFoundError("check your file path")
            im = np.array(im)
            im_height, im_width, channel = im.shape
            print(im.shape)
            for i in range(im_height // SubImgHeight):
                for j in range(im_width // SubImgWidth):
                    save_name = 'Hid{}_Wid{}_'.format(i, j) + im_name + '.npy'
                    crop = im[i * SubImgHeight:(i + 1) * SubImgHeight, j * SubImgWidth:(j + 1) * SubImgWidth, :]

                    np.save(os.path.join(str(save_dir), str(save_name)), crop)

    def gentxt(self):
        """
        制作训练集和测试集的.text文件，放在segmentation文件夹中
        """
        train_ratio = 0.9  # 随机抽取90%的样本作为训练集
        picPath = Path(r'{}'.format(self.savePath_img))
        filelist = os.listdir(str(picPath))
        nbr_samples = len(filelist)
        random.shuffle(filelist)
        txt_dir = Path(r'{}/segmentation'.format(self.root))
        txt_dir.mkdir(exist_ok=True)

        # write test/train/all

        for i, string in enumerate(filelist):
            if i / nbr_samples >= train_ratio:
                with open(os.path.join(str(txt_dir), 'val.txt'), 'a') as f:
                    f.write(string.split('.npy')[0] + '\n')
            else:
                with open(os.path.join(str(txt_dir), 'train.txt'), 'a') as f:
                    f.write(string.split('.npy')[0] + '\n')

            with open(os.path.join(str(txt_dir), 'trainval.txt'), 'a') as f:
                f.write(string.split('.npy')[0] + '\n')


class XA(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False, need_to_cut=True,
                 shuffle=True, flip=True, rotate=True, blur=True, augment=True, val_split=None, return_id=False):

        if channels == 4:
            self.MEAN = [0.45734706, 0.43338275, 0.40058118, 0.4000000]
            self.STD = [0.23965294, 0.23532275, 0.2398498, 0.230000000]
        else:
            self.MEAN = [0.45734706, 0.43338275, 0.40058118]
            self.STD = [0.23965294, 0.23532275, 0.2398498]

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
            'val': val,
            'need_to_cut': need_to_cut
        }

        self.dataset = XADataset(**kwargs)
        super(XA, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
