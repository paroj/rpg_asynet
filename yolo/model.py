'''
This script defines the network architecture.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from .utils_yolo import *


# model_weights = 'curr_model_weights.pth'
model_weights = None


## Config format
# M = Maxpool
# tuple = Conv(kernel_size, out_channels, stride)

cfg = [
        (7, 64, 2), 'M',  # 1
           (3, 192), 'M',   # 2
           (1, 128), (3, 256), (1, 256), (3, 512), 'M',  # 3
           [(1, 256), (3, 512), 4], (1, 512), (3, 1024), 'M',  # 4
           [(1, 512), (3, 1024), 2], (3, 1024), (3, 1024, 2),  # 5
           (3, 1024), (3, 1024)  # 6
    ]


class Conv2dSamePadding(nn.Conv2d):

    def conv2d_same_padding(self, input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
        input_rows = input.size(2)
        filter_rows = weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                             input_rows)
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        padding_cols = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)
        cols_odd = (padding_rows % 2 != 0)

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, weight, bias, stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=dilation, groups=groups)

    def forward(self, input):
        # return F.conv2d(input, self.weight, self.bias, self.stride,
        #                        self.padding, self.dilation, self.groups)
        return self.conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)  ## same padding like TensorFlow


class Darknet(nn.Module):

    def __init__(self, config=cfg, nr_classes=20, nr_box=2, cnn_spatial_output_size=(7, 7), nr_input_channels=2,
                 pretrained_weights_path=None):
        super(Darknet, self).__init__()
        self.nr_classes = nr_classes
        self.nr_box = nr_box
        self.cnn_spatial_output_size = cnn_spatial_output_size
        self.nr_input_channels = nr_input_channels
        self.features = self.make_layers(expand_cfg(config))
        self.cnn_spatial_size_product = cnn_spatial_output_size[0] * cnn_spatial_output_size[1]
        self.classifier = nn.Sequential(
            nn.Linear(1024 * self.cnn_spatial_size_product, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.cnn_spatial_size_product * (nr_box * 5 + nr_classes)),
        )

        if pretrained_weights_path is None:
            self._initialize_weights()
            print("Weights initialized.")
        else:
            # load checkpoints / pretrained weights
            self.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(pretrained_weights_path).items()})
            print('Weights loaded from "{}"'.format(pretrained_weights_path))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(
            x.size(0),
            self.cnn_spatial_output_size[0], self.cnn_spatial_output_size[1],
            self.nr_box * 5 + self.nr_classes
        )
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg):
        '''
        Make layers based on configuration.
        :param cfg: expanded cfg, that is, no list as element
        :return: nn sequential module
        '''
        print(cfg)
        layers = []
        in_channels = self.nr_input_channels
        for v in cfg:
            if v == 'M':  # Max pool
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(v, tuple):
                if len(v) == 3:
                    # Conv (kernel_size, out_channels, stride)
                    layers += [Conv2dSamePadding(in_channels, out_channels=v[1], kernel_size=v[0], stride=2)]
                    # layers += [nn.Conv2d(in_channels, out_channels=v[1], kernel_size=v[0], stride=2)]
                else:
                    # Conv (kernel_size, out_channels)
                    layers += [Conv2dSamePadding(in_channels, out_channels=v[1], kernel_size=v[0])]
                    # layers += [nn.Conv2d(in_channels, out_channels=v[1], kernel_size=v[0])]
                    layers += [nn.BatchNorm2d(num_features=v[1])]  # BN
                    # print('[new] BN is added.')

                layers += [nn.LeakyReLU(0.1)]  # Leaky rectified linear activation
                in_channels = v[1]
        print('Make layers done.')
        return nn.Sequential(*layers)


def expand_cfg(cfg):
    cfg_expanded = []
    for v in cfg:
        if isinstance(v, list):
            times = v[-1]
            for _ in range(times):
                cfg_expanded = cfg_expanded + v[:-1]
        else:
            cfg_expanded.append(v)
    return cfg_expanded


def build_darknet(path=None, **kwargs):
    # define architecture
    extract_features = make_layers(expand_cfg(cfg))
    model = Darknet(extract_features, path, **kwargs)
    '''
    # load weights if using pre-trained
    if path is not None:
        model.load_state_dict(path)
    '''
    return model


if __name__ == "__main__":

    # model
    yolo_model = build_darknet(path=model_weights)

    # input
    '''
    I = io.imread('000001.jpg')
    I = resize(I, (448, 448))
    Imgs = I[np.newaxis, :]
    Imgs = torch.Tensor(Imgs).permute(0, 3, 1, 2)
    print('Imgs.size = ', Imgs.size())
    '''

    Imgs = torch.randn(20, 3, 448, 448)  # test image batch
    print('Imgs.size = ', Imgs.size())

    # output
    output = yolo_model(Imgs)
    print('Done.')



