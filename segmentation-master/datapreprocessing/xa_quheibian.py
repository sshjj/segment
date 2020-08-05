import numpy as np
import cv2

SubImgWidth = 512
SubImgHeight = 512
img = cv2.imread(r'D:\data\5/GF2_PMS1_E116.2_N39.2_20180131_L1A0002971148-MSS1.png')
newimg = img[ :,18:7200,: ]
print(newimg.shape)
cv2.imwrite(r'D:\data\555/GF2_PMS1_E116.2_N39.2_20180131_L1A0002971148-MSS1.png',newimg)