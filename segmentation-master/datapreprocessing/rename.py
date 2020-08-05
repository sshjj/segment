import os
import cv2



picPath = r'C:\Users\96251\Desktop\CD_data\cd_new\18'
savePath = r'C:\Users\96251\Desktop\CD_data\cd_new\18'
pic_names = [picPath +'/'+i for i in os.listdir(picPath)]
for picname in pic_names:
    name = picname.split('/')[1].split('.jpg')[0].split('second')[0] + \
           'first'+ picname.split('/')[1].split('.jpg')[0].split('second')[1]
    newname = name + '.jpg'
    os.rename(picname, os.path.join(savePath, newname))

