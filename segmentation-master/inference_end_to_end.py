"""
文件说明:跳转到该文件最下面
使用方法:
python inference_end_to-end.py
在inference时候可以修改的参数在config.json中的inference下修改,一般只需要改传入的模型权重文件,输入图片文件夹路径和输出图片文件夹路径

使用前注意!!!!!!!!!!!!!!!!
运行之前一定要确认,加载的模型和config中的模型是一致的,比如使用的pspnet.pth,如果网络结构是deeplab,一定会报错的.
包括通道数,要预测的文件后缀名字 .extension 参数
"""

from tqdm import tqdm
from glob import glob
from PIL import Image
import PIL
import models as mymodels
from torchvision import transforms
from scipy import ndimage
import torch.nn.functional as F
import numpy as np
import math
import torch
import os
import torch.nn as nn
import json
import argparse
import tifffile
import gdal


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode="bilinear", align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions


def img_padding(img, patch_size):
    """
    :param img:numpy object
    :param padding_size: padding size
    :return: new img after padding
    """
    print("input image shape:{}".format(img.shape))  # CHW
    _, h, w = img.shape
    padding_size = ((math.ceil(w / patch_size)) * patch_size - w, (math.ceil(h / patch_size)) * patch_size - h)
    new_image = np.pad(img, ((0, 0), (0, padding_size[1]), (0, padding_size[0])), 'constant', constant_values=0)
    # print(new_image.shape)
    return new_image


def image2numpy(image_path):
    if image_path.endswith('.tiff') or image_path.endswith('.tif'):
        tiff = tifffile.imread(image_path).transpose((2, 0, 1))
        print(tiff.shape)
        c, im_height, im_width = tiff.shape
        if np.max(tiff) > 255:
            tiff = np.uint8(stretch_n(np.float32(tiff)) * 255)
        return im_width, im_height, tiff  ###numpy格式,shape(C*H*W)
    elif image_path.endswith(".jpg") or image_path.endswith(".png"):
        image_data = Image.open(image_path)
        image_data = np.array(image_data).transpose((2, 0, 1))
        print(image_data.shape)
        c, im_height, im_width = image_data.shape
        return im_width, im_height, image_data
    else:
        raise FileNotFoundError("filepath error:please check you input tiff_image")


def writeTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def tiff2numpy(tiff_path):
    if tiff_path.endswith('.tiff') or tiff_path.endswith('.tif'):
        tiff = gdal.Open(tiff_path, gdal.GA_ReadOnly)
        im_width = tiff.RasterXSize  # 栅格矩阵的列数
        im_height = tiff.RasterYSize  # 栅格矩阵的行数
        im_bands = tiff.RasterCount  # 波段数
        im_geotrans = tiff.GetGeoTransform()  # 获取仿射矩阵信息
        tiff_projection = tiff.GetProjectionRef()  # 获取空间参考信息
        tiff_data = tiff.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        if np.max(tiff_data) > 255:
            tiff_data = np.uint8(stretch_n(np.float32(tiff_data)) * 255)
        # im_blueBand = tiff_data[0, 0:im_height, 0:im_width]  # 获取蓝波段
        # im_greenBand = tiff_data[1, 0:im_height, 0:im_width]  # 获取绿波段
        # im_redBand = tiff_data[2, 0:im_height, 0:im_width]  # 获取红波段
        # im_nirBand = tiff_data[3, 0:im_height, 0:im_width]  # 获取近红外波段
        return im_width, im_height, tiff_data, im_bands, im_geotrans, tiff_projection  ###numpy格式,shape(C*H*W)
    else:
        raise FileNotFoundError("filepath error:please check you input tiff_image")


def stretch_n(bands, lower_percent=2, higher_percent=98):
    '''
    tiff 16位转8位
    :param bands:
    :param lower_percent:
    :param higher_percent:
    :return:
    '''
    bands = bands.transpose(1, 2, 0)
    # print(bands.dtype)
    # 一定要使用float32类型，原因有两个：1、Keras不支持float64运算；2、float32运算要好于uint16
    out = np.zeros_like(bands).astype(np.float32)
    # print(out.dtype)
    for i in range(bands.shape[2]):
        # 这里直接拉伸到[0,1]之间，不需要先拉伸到[0,255]后面再转
        a = 0
        b = 1
        # 计算百分位数（从小到大排序之后第 percent% 的数）
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    # import cv2
    # cv2.imshow("11", out[:, :, 0:3])
    # cv2.waitKey()
    return out.transpose(2, 0, 1)


def main():
    # Model
    args = parse_arguments()
    config = json.load(open(args.config))
    print("config:{}".format(config))
    inference_params = config["inference"]
    print("inference_params:{}".format(inference_params))
    num_classes = config["num_classes"]
    palette = config["palette"]  ##调色板
    model = getattr(mymodels, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
    print("GPU info :{}".format(device))
    checkpoint = torch.load(inference_params["model_weight_path"])
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Dataset used for training the model
    # scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    scales = [1.0]  ##单尺度预测,也可以多尺度预测,速度慢
    to_tensor = transforms.ToTensor()
    channel = config['arch']['args']["in_channels"]
    if channel == 3:
        MEAN = [0.45734706, 0.43338275, 0.40058118]
        STD = [0.23965294, 0.23532275, 0.2398498]
    elif channel == 4:
        MEAN = [0.45734706, 0.43338275, 0.40058118, 0.4]
        STD = [0.23965294, 0.23532275, 0.2398498, 0.23]
    else:
        raise Exception("Channel param Error")
    normalize = transforms.Normalize(MEAN, STD)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    extension = inference_params["extension"]
    image_files = sorted(glob(os.path.join(inference_params["images_dir_path"], '*{}'.format(extension))))
    patch_size = inference_params["patch_size"]
    if not image_files:
        raise FileNotFoundError("no file in dir ,check path and extension")
    with torch.no_grad():
        for img_file in image_files:
            save_path = os.path.join(inference_params["output_path"],
                                     os.path.basename(img_file).replace(extension, '.png'))
            print("input image path:{}".format(img_file))
            print("output image path:{}".format(save_path))
            if img_file.endswith(".tiff") or img_file.endswith(".tif"):
                width, height, im_data, im_bands, im_geotrans, tiff_projection = tiff2numpy(img_file)
            elif img_file.endswith(".jpg"):

                width, height, im_data=image2numpy(img_file)

            if channel == 3:
                im_data = img_padding(im_data, patch_size)[0:3, :, :]
            else:
                im_data = img_padding(im_data, patch_size)[:, :, :]
            _, h, w = im_data.shape
            im_data = np.pad(im_data, ((0, 0), (patch_size, patch_size), (patch_size, patch_size)), 'constant',
                             constant_values=0)
            x_len = int(w / patch_size)
            y_len = int(h / patch_size)
            predictions_numpy = np.zeros((h, w)).astype(np.uint8)
            if w < 3 * 600 or h < 3 * 600:
                prediction = multi_scale_predict(model, im_data, scales, num_classes, device)
                prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
                colorized_mask = colorize_mask(predictions_numpy, palette)
                colorized_mask.save(os.path.join('test.png'))
            else:
                for i in tqdm(range(x_len)):
                    for j in range(y_len):
                        input = im_data[:, j * patch_size:(j + 3) * patch_size,
                                i * patch_size:(i + 3) * patch_size].transpose(1, 2, 0)
                        # print(input.shape)
                        input = normalize(to_tensor(input)).unsqueeze(0)
                        prediction = multi_scale_predict(model, input, scales, num_classes, device)
                        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()[
                                     patch_size:2 * patch_size, patch_size:2 * patch_size]
                        predictions_numpy[j * patch_size:(j + 1) * patch_size,
                        i * patch_size:(i + 1) * patch_size] = prediction
                predictions_numpy = predictions_numpy[0:height, 0:width]
                if img_file.endswith(".tiff") or img_file.endswith(".tif"):
                    writeTiff(im_data=predictions_numpy, im_width=width, im_height=height, im_bands=1,
                            im_geotrans=im_geotrans, im_proj=tiff_projection, path=save_path.replace(".png", ".tiff"))
                colorized_mask = colorize_mask(predictions_numpy, palette)
                colorized_mask.save(os.path.join(save_path))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='The config used to train the model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

##
##python inference_end_to_end.py
