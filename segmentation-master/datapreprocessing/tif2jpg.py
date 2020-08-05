import skimage.external.tifffile as tifffile
import os
import cv2
import numpy as np

picpath = r'D:\data\xa\0513'
pic_names = [picpath + '/' + i for i in os.listdir(picpath)]
for pic in pic_names:
    name = (pic.split('/')[1]).split('.tif')[0]
    img = cv2.imread(pic)
    img = np.array(img)
    cv2.imwrite(picpath+'/{}.jpg'.format(name),img)