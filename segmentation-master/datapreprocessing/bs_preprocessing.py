from PIL import Image
import numpy as np
import re
import os
import cv2

picPath = r'C:\Users\96251\Desktop\BS_data\labels_16'
savePath = r'C:\Users\96251\Desktop\BS_data\bs200'
SubImgWidth = 200
SubImgHeight = 200

segPath = '{}/vis_labels'.format(savePath)
labelPath = '{}/labels'.format(savePath)
if not os.path.exists(labelPath):
    os.mkdir(labelPath)
if not os.path.exists(segPath):
    os.mkdir(segPath)

pic_names = [picPath + '/' + i for i in os.listdir(picPath)]
for picname in pic_names:
    print(picname)
    img_ori = Image.open(picname)
    img_ori = np.array(img_ori)
    for m in range(img_ori.shape[0]):
        for n in range(img_ori.shape[1]):
            if (img_ori[m][n] == [250,150,150]).all():
                img_ori[m][n] = np.array([255, 255, 255])
    name = (picname.split('/')[1]).split('_label.tif')[0]
    for i in range(0, img_ori.shape[0] // SubImgHeight, 1):
        for j in range(0, img_ori.shape[1] // SubImgWidth, 1):
            SubImg = img_ori[i * SubImgHeight:(i + 1) * SubImgHeight,
                     j * SubImgWidth:(j + 1) * SubImgWidth, :]
            SubImg = Image.fromarray(SubImg)
            SubImg.save(segPath + '/{}_{}_{}.png'.format(name, i, j))
            # cv2.imwrite(segPath + '/{}_{}_{}.png'.format(name, i, j), SubImg)

            # 标签处理部分
            img = Image.open(segPath + '/{}_{}_{}.png'.format(name, i, j)).convert('L')
            # L = R * 299/1000 + G * 587/1000+ B * 114/1000
            voidcount = 0
            for x in range(img.size[0]):
                for y in range(img.size[1]):
                    if img.getpixel((x, y)) == 0:  # 给标注部分分配一个像素值，多分类分配多个像素值
                        img.putpixel((x, y), 0)
                        voidcount += 1
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
            if voidcount <= 0.2 * SubImgWidth *SubImgHeight:
                img.save(labelPath + '/{}_{}_{}.png'.format(name, i, j))