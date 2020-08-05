# Origian code and chechpoints by Hang Zhang
# https://github.com/zhanghang1989/PyTorch-Encoding


import math
import torch
import os
import sys
import zipfile
import shutil
import torch.utils.model_zoo as model_zoo
import torch.nn as nn

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'BasicBlock', 'Bottleneck']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://hangzh.s3.amazonaws.com/encoding/models/resnet50-25c4b509.zip',
    'resnet101': 'https://hangzh.s3.amazonaws.com/encoding/models/resnet101-2a57e44d.zip',
    'resnet152': 'https://hangzh.s3.amazonaws.com/encoding/models/resnet152-0d43d698.zip'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock, 跳接模块，仅在较浅层的 resnet 中使用
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck 有三层卷积的跳接模块，在更深的网络中使用，planes * 4， H, W 除以 stride 和 down sample
    """
    # pylint: disable=unused-argument
    expansion = 4  # 通道的扩张率

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        """
        :param inplanes: 输入通道
        :param planes: 基本输出通道，实际 = planes * expansion
        :param stride: 中间层的 stride 设置
        :param dilation: 中间层用的空洞卷积的 dialation rate
        :param downsample: 最后给卷积层分支的下采样
        :param previous_dilation:
        :param norm_layer:
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." CVPR. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes=1000, dilated=True, multi_grid=False,
                 deep_base=True, norm_layer=nn.BatchNorm2d):
        """

        :param block: 构建网络的基本模块，在浅层和深层中各不相同，浅层为 basic block，深层为 bottleneck
        :param layers: 分别构建多少个四层不同的网络
        :param num_classes: 最后需要分类多少个类别
        :param dilated: 给第三层和第四层是否使用 扩张
        :param multi_grid: 给第四层参数
        :param deep_base: 是否使用一个更深的基础层，多了更多卷积
        :param norm_layer: 用来归一化的层是哪个
        """
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            if multi_grid:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=True)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7, stride=1)  # 平均池化输出 7*7 方块，N，channels 不变
        # 自适应调整 kernel size = (input_size+target_size-1)//target_size，由此导致一定会覆盖所有元素，但是不是规律地平移
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化卷积层为正态分布，归一化层为 0，1 参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        """
        :param block: 用于构建层的基础 block
        :param planes: 输入通道数
        :param blocks: 这一层构建几个
        :param stride:
        :param dilation:
        :param norm_layer:
        :param multi_grid: 首个空洞卷积层 dialation rate = 4， 设置多个 dialation rate
        :return: 构建完成的 layer
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 给 deepbase 统一输入通道/预先做一个 layer 的 stride
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 64 * H/2 * W/2 | Deepbase: 128 * H/2 * W/2 更强表达，更少参数，更深网络（梯度问题）
        print('After conv1: ', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64 * H/4 * W/4 | Deepbase: 128 * H/4 * W/4
        print('After conv1 + maxpool: ', x.shape)

        x = self.layer1(x)  # 256 * H/4 * H/4
        print('After layer1: ', x.shape)

        x = self.layer2(x)  # 512 * H/16 * H/8
        print('After layer2: ', x.shape)

        x = self.layer3(x)  # 1024 * H/8 * H/8
        print('After layer3: ', x.shape)

        x = self.layer4(x)  # 2048 * H/4 * H/4
        print('After layer4: ', x.shape)

        x = self.avgpool(x)
        print('After avgpool: ', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, root='./pretrained', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50'], model_dir=root))
    return model


def resnet101(pretrained=False, root='./pretrained', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101'], model_dir=root))
    return model


def resnet152(pretrained=False, root='/root/data/segmentation/models/pretrained', **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(root + '/resnet152-0d43d698.pth'))
    return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1].split('.')[0]
    cached_file = os.path.join(model_dir, filename + '.pth')
    if not os.path.exists(cached_file):
        cached_file = os.path.join(model_dir, filename + '.zip')
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
        zip_ref = zipfile.ZipFile(cached_file, 'r')
        zip_ref.extractall(model_dir)
        zip_ref.close()
        os.remove(cached_file)
        cached_file = os.path.join(model_dir, filename + '.pth')
    return torch.load(cached_file, map_location=map_location)
