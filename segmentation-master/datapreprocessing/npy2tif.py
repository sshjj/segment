import numpy as np
import skimage.external.tifffile as tifffile
import os

picpath = r'C:\Users\96251\Desktop\XA_data\4ChannelNpy'
pic_names = [picpath + '/' + i for i in os.listdir(picpath)]
for pic in pic_names:
    name = (pic.split('/')[1]).split('.npy')[0]
    print(name)
    npy = np.load(pic)
    tifffile.imsave(r'C:\Users\96251\Desktop\XA_data\4channeltif/{}.tif'.format(name),npy)