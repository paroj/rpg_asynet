"""
This script defines the sparse YOLOv1 network architecture.
"""

# import torch.nn.functional as F
# from skimage import io
# from skimage.transform import rescale, resize, downscale_local_mean
# import numpy as np
import torch.nn.functional

from .utils_yolo import *
import sparseconvnet as scn
import sys
# import os
from utils import test_util
# from sparseconvnet.sparseConvNetTensor import SparseConvNetTensor
from sparseconvnet.utils import *
# from sparseconvnet.sparseConvNetTensor import SparseConvNetTensor
# # from sparseconvnet.convolution import ConvolutionFunction
# from layers.conv_layer_2D import asynSparseConvolution2D, asynNonValidSparseConvolution2D
# from layers.max_pool import asynMaxPool
# import torch.nn.functional as F
# from layers.site_enum import Sites
# from dataloader.dataset import NCaltech101
if __name__ == "__main__":
    from training.trainer import AbstractTrainer
    from dataloader.dataset import NCaltech101_ObjectDetection

from layers.asyn_sparse_conv_model import AsynSparseConvModel, SparseConvModel

CONFIG = [
    (7, 64, 2), 'M',  # 1
    (3, 192), 'M',   # 2
    (1, 128), (3, 256), (1, 256), (3, 512), 'M',  # 3
    [(1, 256), (3, 512), 4], (1, 512), (3, 1024), 'M',  # 4
    [(1, 512), (3, 1024), 2], (3, 1024), (3, 1024, 2),  # 5
    (3, 1024), (3, 1024)  # 6
]
"""layer config of the convolutional block of YOLOv1"""


class SparseYolo(SparseConvModel):
    """
    synchronous sparse YOLOv1 model
    """

    def __init__(self, config=CONFIG, nr_classes=C, nr_box=B, cnn_spatial_output_size=(S, S), nr_input_channels=2,
                 linear_input_channels=1024, batch_size=1, use_bias=True,
                 device=torch.device("cpu")):

        self.nr_classes = nr_classes
        self.nr_box = nr_box

        spatial_size_product = cnn_spatial_output_size[0] * cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * linear_input_channels

        # create SparseConvModel with the appropriate values and properties to model YOLOv1
        super(SparseYolo, self).__init__(
            config=config,
            dense_layers=[
                {'layer': 'FC', 'in_features': self.linear_input_features, 'out_features': 4096},
                {'layer': 'LeakyRelu', 'negative_slope': 0.1},
                {
                    'layer': 'FC',
                    'in_features': 4096,
                    'out_features': spatial_size_product * (self.nr_box * 5 + self.nr_classes)
                }
            ],
            nr_input_channels=nr_input_channels,
            batch_size=batch_size, dense_input_channels=linear_input_channels,
            cnn_spatial_output_size=cnn_spatial_output_size,
            use_bias=use_bias, device=device
        )

        self.cnn_spatial_output_size = list(cnn_spatial_output_size)

    def forward(self, x):
        """
        compute forward pass, then reshape to YOLO output format
        :param x: input sample(s)
        :return: prediction(s)
        """
        x = super().forward(x)
        x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5 * self.nr_box)])
        return x


class AsynSparseYolo(AsynSparseConvModel):
    """
    asynchronous sparse YOLOv1 model
    """

    def __init__(self, config=CONFIG, nr_classes=C, nr_box=B, cnn_spatial_output_size=(S, S),
                 nr_input_channels=2, linear_input_channels=1024, use_same_padding=True, pretrained_weights_path=None,
                 device=torch.device("cpu"), cpp: bool = False):

        self.cnn_spatial_output_size = list(cnn_spatial_output_size)
        self.nr_classes = nr_classes
        self.nr_box = nr_box

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]

        self.linear_input_features = spatial_size_product * linear_input_channels

        # create AsynSparseConvModel with the appropriate values and properties to model YOLOv1
        super().__init__(
            config,
            dense_layers=[
                {'layer': 'FC', 'in_features': self.linear_input_features, 'out_features': 4096},
                {'layer': 'LeakyRelu', 'negative_slope': 0.1},
                {
                    'layer': 'FC',
                    'in_features': 4096,
                    'out_features': spatial_size_product * (self.nr_box * 5 + self.nr_classes)
                }
            ],
            nr_input_channels=nr_input_channels,
            use_bias=False,
            device=device,
            cpp=cpp
        )

    def forward(self, x_asyn):
        """
        compute forward pass, then reshape to YOLO output format
        :param x_asyn: input sample(s)
        :return: prediction(s)
        """
        x_asyn = super().forward(x_asyn)

        x_asyn[1] = x_asyn[1].view(
            (
                self.cnn_spatial_output_size[0],
                self.cnn_spatial_output_size[1],
                (self.nr_classes + 5*self.nr_box)
            )
        ).permute(1, 2, 0)

        return x_asyn


if __name__ == "__main__":

    sys.path.append(os.path.abspath('..'))

    # model
    nr_input_channels = 2
    yolo_model = SparseYolo(nr_input_channels=nr_input_channels)
    input_spacial_size = yolo_model.get_input_spacial_size()

    # ---- Create Input ----
    sequence_length = 10
    out = test_util.createInput(nIn=nr_input_channels, spatial_dimensions=input_spacial_size,
                                asynchronous_input=True, sequence_length=sequence_length, simplified=False)
    batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

    # ---- Sparse Yolo ----
    select_indices = tuple(batch_update_locations.T)
    features = batch_input[select_indices]
    batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
    features = torch.tensor(features, dtype=torch.float)

    # output
    output = yolo_model([batch_locations, features])

    print(f"output: {output}")

    print('Done.')
    print('Testing real data...')

    nr_classes = 101
    nr_last_events = 1000
    sequence_length = 10

    # ---- Create Input ----
    dataset_path = '../data/NCaltech101_ObjectDetection'
    small_out_map = dataset_path.endswith('NCaltech101_ObjectDetection')
    height = 180
    width = 240
    train_dataset = NCaltech101_ObjectDetection(dataset_path, ['Motorbikes'], height, width, augmentation=False,
                                                mode='validation', nr_events_window=nr_last_events)

    # ---- Sparse Yolo ----
    model = SparseYolo(nr_input_channels=nr_input_channels).eval()
    spatial_dimensions = model.spatial_size

    events, bbs, histogram = train_dataset.__getitem__(0)

    histogram = torch.from_numpy(histogram[np.newaxis, :, :])
    histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(spatial_dimensions))
    histogram = histogram.permute(0, 2, 3, 1)
    locations, features = AbstractTrainer.denseToSparse(histogram)

    output = model([locations, features])

    print(f"output: {output}")
    print('Done.')
