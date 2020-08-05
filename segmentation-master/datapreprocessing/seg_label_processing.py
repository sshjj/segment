from PIL import Image
import numpy as np
import re
import os
import cv2

picPath = r'D:\data\5'
savePath = r'D:\data\xa_mod_data\20180131'
SubImgWidth = 399
SubImgHeight = 425

segPath = '{}/vis_labels'.format(savePath)
labelPath = '{}/labels'.format(savePath)
if not os.path.exists(labelPath):
    os.mkdir(labelPath)
if not os.path.exists(segPath):
    os.mkdir(segPath)

pic_names = [picPath + '/' + i for i in os.listdir(picPath)]
for picname in pic_names:
    print(picname)
    name = (picname.split('/')[1]).split('_label.png')[0]
    img = cv2.imread(picname)
    img = np.array(img)
    for i in range(0, img.shape[0] // SubImgHeight, 1):
        for j in range(0, img.shape[1] // SubImgWidth, 1):
            SubImg = img[i * SubImgHeight:(i + 1) * SubImgHeight,
                     j * SubImgWidth:(j + 1) * SubImgWidth, :]
            cv2.imwrite(segPath + '/{}_{}_{}.png'.format(name, i, j), SubImg)

            # 标签处理部分
            # img2 = Image.fromarray(SubImg).convert('L')
            img2 = Image.open(segPath + '/{}_{}_{}.png'.format(name, i, j)).convert('L')
            # L = R * 299/1000 + G * 587/1000+ B * 114/1000
            for x in range(img2.size[0]):
                for y in range(img2.size[1]):
                    if img2.getpixel((x, y)) == 0:  # 给标注部分分配一个像素值，多分类分配多个像素值
                        img2.putpixel((x, y), 0)
                    if img2.getpixel((x, y)) == 191:
                        img2.putpixel((x, y), 1)
                    if img2.getpixel((x, y)) == 146:
                        img2.putpixel((x, y), 2)
                    if img2.getpixel((x, y)) == 58:
                        img2.putpixel((x, y), 3)
                    if img2.getpixel((x, y)) == 59:
                        img2.putpixel((x, y), 4)
                    if img2.getpixel((x, y)) == 255:
                        img2.putpixel((x, y), 5)
                    if img2.getpixel((x, y)) == 22:
                        img2.putpixel((x, y), 6)
                    if img2.getpixel((x, y)) == 116:
                        img2.putpixel((x, y), 7)
            img2.putpalette(
                [0, 0, 0, 150, 250, 0, 0, 250, 0, 0, 100, 0, 200, 0, 0, 255, 255, 255, 0, 0, 200, 0, 150, 250])
            # img.show()
            img2.save(labelPath + '/{}_{}_{}_label.png'.format(name, i, j))
