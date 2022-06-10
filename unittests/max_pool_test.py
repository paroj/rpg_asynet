"""
Example command: python -m unittests.sparse_VGG_test
"""
import random

import numpy as np
import torch
import tqdm
import unittest

# TODO
import sparseconvnet as scn
from layers.conv_layer_2D import asynSparseConvolution2D, asynNonValidSparseConvolution2D
from layers.max_pool import asynMaxPool
from layers.asyn_sparse_conv_model import SparseConvolutionSamePadding

from training.trainer import AbstractTrainer
import utils.test_util as test_util

# DEVICE = torch.device("cuda:0")
DEVICE = torch.device("cpu")

spatial_dimensions = [10, 10]
nIn = 1
nOut = 1
stride = 2
filter_size = 2

output_spatial_size = [(dim + stride - 1) // stride for dim in spatial_dimensions]

class TestSparseVGG(unittest.TestCase):
    def test_max_pool_artificial_input(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        # generate/set seed
        seed = np.random.randint(2**32-1)
        # seed = 294013692
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print('Asynchronous N-Layer Test')
        for i_test in tqdm.tqdm(range(10)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')
            sequence_length = 10

            # ---- Facebook sparse convolution with same padding as ground truth ----
            # main layer
            fb_model = scn.MaxPooling(dimension=2, pool_size=filter_size, pool_stride=stride)
            # compute padding and required spacial size
            spatial_size = fb_model.input_spatial_size(torch.LongTensor(output_spatial_size))
            # print(f"self.same_padding_offset: {same_padding_offset}")
            # input layer to convert data to correct format
            input_layer = scn.InputLayer(
                dimension=2, spatial_size=spatial_size, mode=2
            ).to(torch.device("cpu"))
            # sparse to dense layer to retrieve output
            sparse_to_dense = scn.SparseToDense(dimension=2, nPlanes=nOut)

            # ---- Create Input ----
            out = test_util.createInput(
                nIn=nIn, spatial_dimensions=list(spatial_size), asynchronous_input=True,
                sequence_length=sequence_length, simplified=False
            )
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            # async sparse convolution (SC, not VSC)
            asyn_model = asynMaxPool(
                dimension=2, filter_size=filter_size, filter_stride=stride,
                padding_mode='valid',
                device=DEVICE
            )

            # prepare data for fb layer
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
            features = torch.tensor(features, dtype=torch.float)
            # compute ground truth
            # apply padding
            # TODO
            # batch_locations = batch_locations + same_padding_offset[:batch_locations.shape[-1]]
            # compute convolution
            fb_output = sparse_to_dense(fb_model(input_layer([batch_locations, features])))
            # debug
            # print("\n\n")
            # print(sparse_to_dense(input_layer([batch_locations, features])))
            print("\n\n")
            print(fb_output)
            print("\n\n")

            # ---- Asynchronous VGG ----
            with torch.no_grad():
                for i_sequence in range(sequence_length):
                    x_asyn = [None] * 5
                    x_asyn[0] = torch.tensor(asyn_update_locations[i_sequence], dtype=torch.long).to(DEVICE)
                    x_asyn[1] = torch.tensor(asyn_input[i_sequence], dtype=torch.float).to(DEVICE)
                    # debug
                    # print("\n\n")
                    # print(x_asyn[1].float().data.cpu().numpy())
                    # forward
                    asyn_output = asyn_model.forward(
                        update_location=x_asyn[0].to(DEVICE),
                        feature_map=x_asyn[1].double().to(DEVICE)
                    )
                    asyn_output = list(asyn_output)
                    print('--------Sequence %s----------' % i_sequence)
                    # debug
                    print("\n\n")
                    print(x_asyn[1])
                    print("\n\n")
                    print(asyn_output[1].float().data.cpu().numpy())
                    print("\n\n")
                    print(x_asyn[0])
                    print("\n\n")
                    print(asyn_output[0].float().data.cpu().numpy())

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy(), decimal=5)

    def test_max_pool_two_layers(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        # generate/set seed
        seed = np.random.randint(2**32-1)
        # seed = 294013692
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print('Asynchronous N-Layer Test')
        for i_test in tqdm.tqdm(range(10)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')
            sequence_length = 10

            # ---- Facebook sparse convolution with same padding as ground truth ----
            # main layer
            fb_layer_1 = scn.MaxPooling(dimension=2, pool_size=filter_size, pool_stride=stride)
            fb_layer_2 = scn.MaxPooling(dimension=2, pool_size=filter_size, pool_stride=stride)
            # compute padding and required spacial size
            spatial_size = fb_layer_1.input_spatial_size(
                fb_layer_2.input_spatial_size(
                    torch.LongTensor(output_spatial_size)
                )
            )
            # print(f"self.same_padding_offset: {same_padding_offset}")
            # input layer to convert data to correct format
            input_layer = scn.InputLayer(
                dimension=2, spatial_size=spatial_size, mode=2
            ).to(torch.device("cpu"))
            # sparse to dense layer to retrieve output
            sparse_to_dense = scn.SparseToDense(dimension=2, nPlanes=nOut)

            # ---- Create Input ----
            out = test_util.createInput(
                nIn=nIn, spatial_dimensions=list(spatial_size), asynchronous_input=True,
                sequence_length=sequence_length, simplified=False
            )
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            # async sparse convolution (SC, not VSC)
            asyn_layer_1 = asynMaxPool(
                dimension=2, filter_size=filter_size, filter_stride=stride,
                padding_mode='valid',
                device=DEVICE
            )
            asyn_layer_2 = asynMaxPool(
                dimension=2, filter_size=filter_size, filter_stride=stride,
                padding_mode='valid',
                device=DEVICE
            )

            # prepare data for fb layer
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
            features = torch.tensor(features, dtype=torch.float)
            # compute ground truth
            # apply padding
            # TODO
            # batch_locations = batch_locations + same_padding_offset[:batch_locations.shape[-1]]
            # compute convolution
            fb_output = sparse_to_dense(fb_layer_1(fb_layer_2(input_layer([batch_locations, features]))))
            # debug
            # print("\n\n")
            # print(sparse_to_dense(input_layer([batch_locations, features])))
            print("\n\n")
            print(fb_output)
            print("\n\n")

            # ---- Asynchronous VGG ----
            with torch.no_grad():
                for i_sequence in range(sequence_length):
                    x_asyn = [None] * 5
                    x_asyn[0] = torch.tensor(asyn_update_locations[i_sequence], dtype=torch.long).to(DEVICE)
                    x_asyn[1] = torch.tensor(asyn_input[i_sequence], dtype=torch.float).to(DEVICE)
                    # debug
                    # print("\n\n")
                    # print(x_asyn[1].float().data.cpu().numpy())
                    # forward
                    tmp = asyn_layer_1.forward(
                        update_location=x_asyn[0].to(DEVICE),
                        feature_map=x_asyn[1].double().to(DEVICE)
                    )
                    asyn_output = asyn_layer_2.forward(
                        update_location=tmp[0],
                        feature_map=tmp[1].double()
                    )
                    asyn_output = list(asyn_output)
                    print('--------Sequence %s----------' % i_sequence)
                    # debug
                    print("\n\n")
                    print(x_asyn[1])
                    print("\n\n")
                    print(asyn_output[1].float().data.cpu().numpy())
                    print("\n\n")
                    print(x_asyn[0])
                    print("\n\n")
                    print(asyn_output[0].float().data.cpu().numpy())

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy(), decimal=5)


if __name__ == '__main__':
    unittest.main()
