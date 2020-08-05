import numpy as np
import os
import cv2

def lstsort(elem):
    return int(((elem.split('(')[1]).split(')')[0]))
picpath = r'C:\Users\96251\Desktop\20150912'


picname = [picpath + '/' + i for i in os.listdir(picpath)]
picname.sort(key=lstsort)
print(picname)
row = 6800
col = 7182
img_size_r = 425
img_size_c = 399

res=np.zeros((row, col, 3), dtype=np.uint8)
idx=-1
for i in range(row//img_size_r):
    for j in range(col//img_size_c):
        idx += 1
        print(idx)
        res[i*img_size_r:(i+1)*img_size_r, j*img_size_c:(j+1)*img_size_c, :]=cv2.imread(picname[idx]);
cv2.imwrite(r'C:\Users\96251\Desktop\GF2_PMS2_E115.8_N38.8_20181103_L1A0003570838-MSS2.png',res)
