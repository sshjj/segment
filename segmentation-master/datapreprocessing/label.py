
import numpy as np
import re
import os
import cv2
from random import randint
from PIL import Image
import os
from skimage import io,transform
import skimage.external.tifffile as tifffile

picPath = r'D:\data\xa_mod_data\20180131\vislabels'
savePath = r'D:\data\xa_mod_data\20180131\label'
pic_names = [picPath +'/'+i for i in os.listdir(picPath)]

for picname in pic_names:
    img = Image.open(picname)
    img = np.array(img)
    print(img)
    img = Image.fromarray(img.astype('uint8')).convert('L')
    # img = Image.open(picname).convert('L')
    print(np.array(img))
    name = (picname.split('/')[1]).split('.png')[0]
    # img.show()
    print(img.size)
    # print(type(img.getpixel((0, 0))))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if img.getpixel((x, y)) == 0:  # 给标注部分分配一个像素值，多分类分配多个像素值
                img.putpixel((x, y), 0)
            if img.getpixel((x, y)) == 191:
                img.putpixel((x, y), 1)
            if img.getpixel((x, y)) == 146:
                img.putpixel((x, y), 2)
            if img.getpixel((x, y)) == 58:
                img.putpixel((x, y), 3)
            if img.getpixel((x, y)) == 59:
                img.putpixel((x, y), 4)
            if img.getpixel((x, y)) == 255:
                img.putpixel((x, y), 5)
            if img.getpixel((x, y)) == 22:
                img.putpixel((x, y), 6)
            if img.getpixel((x, y)) == 116:
                img.putpixel((x, y), 7)
    # img.putpalette([0, 0, 0, 150, 250, 0, 0, 250, 0, 0, 100, 0, 200, 0, 0, 255, 255, 255, 0, 0, 200, 0, 150, 250])
    # img.show()
    img = np.array(img)
    img = img + 1
    # img.save(savePath+'/{}.png'.format(name))
    cv2.imwrite(savePath+'/{}_label.png'.format(name),img)
