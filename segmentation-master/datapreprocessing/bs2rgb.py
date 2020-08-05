import numpy as np
import cv2
import os
import skimage.external.tifffile as tifffile
import gdal

PicPath = r'C:\Users\96251\Desktop\BS_data\tif-image'
OutPath = r'C:\Users\96251\Desktop\BS_data\rgb'
imgs = [os.path.join(PicPath, img) for img in os.listdir(PicPath)]
for i in range(len(imgs)):
    print(imgs[i])
    name = (imgs[i].split('\\')[-1]).split('.tif')[0]
    # img = img.ReadAsarray(img.data)
    img = tifffile.imread(imgs[i])
    img = np.array(img)
    print(img.shape)
    NIR=img[:,:,0]
    R=img[:,:,1]
    G=img[:,:,2]
    B=img[:,:,3]
    img=cv2.merge([R, G, B])
    print(img)
    # cv2.imwrite(OutPath+'/test_{}.png'.format(i),img)
    # SubImg_1 = img[0:2453,:0:3594,:]
    tifffile.imsave(OutPath+'/{}.jpg'.format(name), img)