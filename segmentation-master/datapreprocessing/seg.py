import os
from pathlib import Path

import numpy as np
from PIL import Image

root = os.getcwd()
picPath_lbl = root + '/../data/color_label_xa_2020'
# picPath_lbl = root + '/../data/color_labels_xa_2020'
# picPath_lbl = root + '/../data/labels'
picPath_img = root + '/../data/image_xa_2020'
savePath_lbl = Path(root + '/../data/labels_xa_2020')
savePath_img = Path(root + '/../data/images_xa_2020')
savePath_img.mkdir(exist_ok=True)
savePath_lbl.mkdir(exist_ok=True)
imagelist = os.listdir(picPath_lbl)


SubImgWidth = 512
SubImgHeight = 512

palette = list([[0, 0, 0], [150, 250, 0], [0, 250, 0], [0, 100, 0],
                [200, 0, 0], [255, 255, 255], [0, 0, 200], [0, 150, 250]])


def label2intarray(label):
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
                idx = palette.index(label[i][j].tolist())
                label_int[i][j] = idx
            except KeyError:
                error.add(tuple(label[i][j]))
    # print('error', error)
    return np.array(label_int)


def cut_util(tag, base_name, save_dir):
    save_ext = ['.jpg', '.png']["label" in tag]
    img_name = base_name + save_ext
    im = Image.open(img_name)
    im = np.array(im)
    if 'label' in tag:
        im = label2intarray(im)
        im_height, im_width = im.shape
    else:
        im_height, im_width, channels = im.shape

    for i in range(im_height // SubImgHeight):
        for j in range(im_width // SubImgWidth):
            save_name = 'Hid{}_Wid{}_'.format(i, j) + im_name + save_ext
            print('start save {}'.format(save_name))
            crop = im[i * SubImgHeight:(i + 1) * SubImgHeight, j * SubImgWidth:(j + 1) * SubImgWidth]
            crop_im = Image.fromarray(crop)
            crop_im.save(os.path.join(save_dir, save_name))


for image_name in imagelist:
    im_name = str(image_name).split('\\')[-1][:-4]
    print('image_name', savePath_img.  joinpath(im_name))
    cut_util('label', os.path.join(picPath_lbl, im_name), savePath_lbl)
    # cut_util('image', os.path.join(picPath_img, im_name), savePath_img)
