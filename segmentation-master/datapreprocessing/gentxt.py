# coding=utf-8
# 根据路径，生成对应比例的train.txt/val.txt/trainval.txt
import os
from pathlib import Path

import random

train_ratio = 0.9  # 随机抽取90%的样本作为训练集
root = os.getcwd()
picPath = Path(root+'/../data/images_xa_2020')
filelist = os.listdir(picPath)
nbr_samples = len(filelist)
random.shuffle(filelist)
txt_dir = Path.joinpath(picPath, Path('./../txt'))
txt_dir.mkdir(exist_ok=True)

# write test/train/all

for i, string in enumerate(filelist):
    if i / nbr_samples >= train_ratio:
        with open(txt_dir.joinpath(Path('./test.txt')), 'a') as f:
            f.write(string+'\n')
    else:
        with open(txt_dir.joinpath(Path('./train.txt')), 'a') as f:
            f.write(string+'\n')

    with open(txt_dir.joinpath(Path('./trainval.txt')), 'a') as f:
        f.write(string+'\n')
