
import numpy as np
import re
import os
import cv2
from random import randint
from PIL import Image
import os
from skimage import io,transform
import skimage.external.tifffile as tifffile

picPath = r'C:\Users\96251\Desktop\BS_data\n'
savePath = r'C:\Users\96251\Desktop\BS_data\labels'
pic_names = [picPath +'/'+i for i in os.listdir(picPath)]

for picname in pic_names:
    img = Image.open(picname)
    img = np.array(img)
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if (img[m][n] == [250,150,150]).all():
                img[m][n] = np.array([255, 255, 255])
    print(img)
    img = Image.fromarray(img.astype('uint8')).convert('L')
    # img = Image.open(picname).convert('L')
    print(np.array(img))
    name = (picname.split('/')[1]).split('_label.tif')[0]
    img.show()
    print(img.size)
    # print(type(img.getpixel((0, 0))))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if img.getpixel((x, y)) == 0:  # 给标注部分分配一个像素值，多分类分配多个像素值
                img.putpixel((x, y), 0)
            if img.getpixel((x, y)) == 191:
                img.putpixel((x, y), 1)
            if img.getpixel((x, y)) == 179:
                img.putpixel((x, y), 1)
            if img.getpixel((x, y)) == 117:
                img.putpixel((x, y), 1)
            if img.getpixel((x, y)) == 192:
                img.putpixel((x, y), 2)
            if img.getpixel((x, y)) == 177:
                img.putpixel((x, y), 2)
            if img.getpixel((x, y)) == 73:
                img.putpixel((x, y), 3)
            if img.getpixel((x, y)) == 161:
                img.putpixel((x, y), 3)
            if img.getpixel((x, y)) == 82:
                img.putpixel((x, y), 3)
            if img.getpixel((x, y)) == 59:
                img.putpixel((x, y), 4)
            if img.getpixel((x, y)) == 91:
                img.putpixel((x, y), 4)
            if img.getpixel((x, y)) == 164:
                img.putpixel((x, y), 4)
            if img.getpixel((x, y)) == 255:
                img.putpixel((x, y), 5)
            if img.getpixel((x, y)) == 22:
                img.putpixel((x, y), 6)
            if img.getpixel((x, y)) == 110:
                img.putpixel((x, y), 6)
            if img.getpixel((x, y)) == 145:
                img.putpixel((x, y), 6)
            # if img.getpixel((x, y)) == 116:
            #     img.putpixel((x, y), 7)
    img.putpalette([0, 0, 0, 150, 250, 0, 0, 250, 0, 0, 100, 0, 200, 0, 0, 255, 255, 255, 0, 0, 200])
    img.show()
    img.save(savePath+'/{}.png'.format(name))

