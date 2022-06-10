"""
Example command: python -m unittests.sparse_VGG_test
"""
import random

import numpy as np
import torch
import tqdm
import unittest

import sparseconvnet as scn
from layers.conv_layer_2D import asynSparseConvolution2D, asynNonValidSparseConvolution2D
from layers.asyn_sparse_conv_model import SparseConvolutionSamePadding

import utils.test_util as test_util

# DEVICE = torch.device("cuda:0")
DEVICE = torch.device("cpu")

spatial_dimensions = [199, 254]
# spatial_dimensions = [3, 3]  # [199, 254]
nIn = 1
nOut = 1
stride = 3
filter_size = 3


class TestSingleConvOp(unittest.TestCase):
    def test_single_conv_op_against_torch_conv2d_artificial_input(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        print('Asynchronous N-Layer Test')
        output_spatial_size = [(dim + stride - 1) // stride for dim in spatial_dimensions]
        for i_test in tqdm.tqdm(range(10)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')

            # ---- Create Input ----
            out = test_util.createInput(nIn=nIn, spatial_dimensions=spatial_dimensions,
                                        asynchronous_input=True, sequence_length=1, simplified=False)
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            asyn_model = asynNonValidSparseConvolution2D(
                dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, first_layer=True, filter_stride=stride,
                use_bias=False, device=DEVICE
            )

            dense_model = torch.nn.Conv2d(nIn, nOut, filter_size, stride=stride, padding=filter_size // 2, bias=False)
            # set weights equal
            dense_model.weight = torch.nn.Parameter(asyn_model.weight.data.reshape(1, nIn, filter_size, filter_size))

            with torch.no_grad():
                x_asyn = [None] * 5
                x_asyn[0] = torch.tensor(asyn_update_locations[0], dtype=torch.float).to(DEVICE)
                x_asyn[1] = torch.tensor(asyn_input[0], dtype=torch.float).to(DEVICE)

                # forward sparse
                asyn_output = asyn_model.forward(
                    update_location=x_asyn[0].to(DEVICE),
                    feature_map=x_asyn[1].to(DEVICE),
                    active_sites_map=None,
                    rule_book_input=None,
                    rule_book_output=None
                )
                asyn_output = list(asyn_output)
                # forward dense
                dense_output = dense_model(x_asyn[1].reshape([1, 1] + spatial_dimensions))

                # debug
                # print("\n\n")
                # print(x_asyn[1].float().data.cpu().numpy())
                # print("\n\n")
                # print(asyn_model.weight.data)
                # print("\n\n")
                # print(asyn_output[1].float().data.cpu().numpy())
                # print("\n\n")
                # print(dense_output)
                # print("\n\n")

                np.testing.assert_almost_equal(
                    asyn_output[1].float().data.cpu().numpy().reshape((nOut, ) + tuple(output_spatial_size)),
                    dense_output.squeeze(0).detach().numpy(), decimal=5)


class TestSparseVGG(unittest.TestCase):
    def test_single_conv_op_against_SCN_conv2d_artificial_input(self):
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
        output_spatial_size = [(dim + stride - 1) // stride for dim in spatial_dimensions]
        for i_test in tqdm.tqdm(range(10)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')
            sequence_length = 10

            # ---- Facebook sparse convolution with same padding as ground truth ----
            # main layer
            fb_model = SparseConvolutionSamePadding(
                dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, bias=False, filter_stride=stride, batch_size=1
            )
            # compute padding and required spacial size
            # fb_model.set_return_padded_size(False)
            spatial_size = fb_model.input_spatial_size(torch.LongTensor(output_spatial_size))
            # print(f"self.spatial_size: {spatial_size}")
            # fb_model.set_return_padded_size(True)
            spatial_size_with_padding = fb_model.input_spatial_size(torch.LongTensor(output_spatial_size))
            # print(f"spatial_size_with_padding: {spatial_size_with_padding}")
            same_padding_offset = (spatial_size_with_padding - spatial_size) // 2
            same_padding_offset = torch.cat(
                (same_padding_offset, torch.zeros(1, dtype=torch.long))
            ).to(DEVICE)
            print(f"self.same_padding_offset: {same_padding_offset}")
            # input layer to convert data to correct format
            input_layer = scn.InputLayer(
                dimension=2, spatial_size=spatial_size_with_padding, mode=2
            ).to(torch.device("cpu"))
            # sparse to dense layer to retrieve output
            sparse_to_dense = scn.SparseToDense(dimension=2, nPlanes=nOut)

            # ---- Create Input ----
            out = test_util.createInput(
                nIn=nIn, spatial_dimensions=spatial_dimensions, asynchronous_input=True,
                sequence_length=sequence_length, simplified=False
            )
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            # async sparse convolution (SC, not VSC)
            asyn_model = asynNonValidSparseConvolution2D(
                dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, first_layer=True, filter_stride=stride,
                use_bias=False, device=DEVICE
            )
            # set weights equal to facebook sparseconv layer
            asyn_model.weight.data = fb_model.weight.squeeze(1).to(torch.device("cpu"))

            # prepare data for fb layer
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
            features = torch.tensor(features, dtype=torch.float)
            # compute ground truth
            # apply padding
            batch_locations = batch_locations + same_padding_offset[:batch_locations.shape[-1]]
            # compute convolution
            fb_output = sparse_to_dense(fb_model(input_layer([batch_locations, features])))
            # debug
            # print("\n\n")
            # print(sparse_to_dense(input_layer([batch_locations, features])))
            # print("\n\n")
            # print(fb_model.weight.squeeze(1))
            # print("\n\n")
            # print(fb_output)
            # print("\n\n")

            # ---- Asynchronous VGG ----
            with torch.no_grad():
                for i_sequence in range(sequence_length):
                    x_asyn = [None] * 5
                    x_asyn[0] = torch.tensor(asyn_update_locations[i_sequence], dtype=torch.float).to(DEVICE)
                    x_asyn[1] = torch.tensor(asyn_input[i_sequence], dtype=torch.float).to(DEVICE)
                    # debug
                    # print("\n\n")
                    # print(x_asyn[1].float().data.cpu().numpy())
                    # forward
                    asyn_output = asyn_model.forward(
                        update_location=x_asyn[0].to(torch.device("cpu")),
                        feature_map=x_asyn[1].to(torch.device("cpu")),
                        active_sites_map=None,
                        rule_book_input=None,
                        rule_book_output=None
                    )
                    asyn_output = list(asyn_output)
                    print('--------Sequence %s----------' % i_sequence)
                    # debug
                    # print("\n\n")
                    # print(asyn_model.weight.data)
                    # print("\n\n")
                    # print(asyn_output[1].float().data.cpu().numpy())
                    # print("\n\n")

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy(), decimal=5)

    def test_two_stacked_conv_ops_against_SCN_conv2d_artificial_input(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        # generate/set seed
        seed = np.random.randint(2**32-1)
        # seed = 4190198142
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print('Asynchronous N-Layer Test')
        # override
        output_spatial_size = [(dim + stride - 1) // stride for dim in spatial_dimensions]
        output_spatial_size = [(dim + stride - 1) // stride for dim in output_spatial_size]
        for i_test in tqdm.tqdm(range(10)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')
            sequence_length = 10

            # ---- Facebook sparse convolution with same padding as ground truth ----
            # main layer
            fb_layer_1 = SparseConvolutionSamePadding(
                dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, bias=False, filter_stride=stride, batch_size=1
            )
            fb_layer_2 = SparseConvolutionSamePadding(
                dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, bias=False, filter_stride=stride,
                batch_size=1
            )
            # compute required spacial size
            spatial_size = fb_layer_1.input_spatial_size(
                fb_layer_2.input_spatial_size(
                    torch.LongTensor(output_spatial_size)
                )
            )
            # print(f"self.spatial_size: {spatial_size}")
            # input layer to convert data to correct format
            input_layer = scn.InputLayer(
                dimension=2, spatial_size=spatial_size, mode=2
            ).to(torch.device("cpu"))
            # sparse to dense layer to retrieve output
            sparse_to_dense = scn.SparseToDense(dimension=2, nPlanes=nOut)

            # ---- Create Input ----
            out = test_util.createInput(
                nIn=nIn, spatial_dimensions=spatial_dimensions, asynchronous_input=True,
                sequence_length=sequence_length, simplified=False
            )
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            # async sparse convolution (SC, not VSC)
            asyn_layer_1 = asynNonValidSparseConvolution2D(
                dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, first_layer=True, filter_stride=stride,
                use_bias=False, device=DEVICE
            )
            asyn_layer_2 = asynNonValidSparseConvolution2D(
                dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, first_layer=False, filter_stride=stride,
                use_bias=False, device=DEVICE
            )
            # set weights equal to facebook sparseconv layer
            asyn_layer_1.weight.data = fb_layer_1.weight.squeeze(1).to(torch.device("cpu"))
            asyn_layer_2.weight.data = fb_layer_2.weight.squeeze(1).to(torch.device("cpu"))

            # prepare data for fb layer
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
            features = torch.tensor(features, dtype=torch.float)
            # compute ground truth
            # compute convolution
            fb_input = input_layer([batch_locations, features])
            fb_tmp_1 = fb_layer_1(fb_input)
            fb_output_tmp_1 = sparse_to_dense(fb_tmp_1)
            fb_tmp_2 = fb_layer_2(fb_tmp_1)
            fb_output = sparse_to_dense(fb_tmp_2)
            # debug
            # print("\n\n")
            # print(sparse_to_dense(input_layer([batch_locations, features])))
            # print("\n\n")
            # print(fb_model.weight.squeeze(1))
            # print("\n\n")
            # print(fb_output)
            # print("\n\n")

            # ---- Asynchronous VGG ----
            with torch.no_grad():
                for i_sequence in range(sequence_length):
                    x_asyn = [None] * 5
                    x_asyn[0] = torch.tensor(asyn_update_locations[i_sequence], dtype=torch.float).to(DEVICE)
                    x_asyn[1] = torch.tensor(asyn_input[i_sequence], dtype=torch.float).to(DEVICE)
                    # debug
                    # print("\n\n")
                    # print(x_asyn[1].float().data.cpu().numpy())
                    # forward
                    tmp = asyn_layer_1.forward(
                        update_location=x_asyn[0].to(torch.device("cpu")),
                        feature_map=x_asyn[1].to(torch.device("cpu")),
                        active_sites_map=None,
                        rule_book_input=None,
                        rule_book_output=None
                    )
                    asyn_output = asyn_layer_2.forward(
                        update_location=tmp[0],
                        feature_map=tmp[1],
                        active_sites_map=tmp[2],
                        rule_book_input=tmp[3],
                        rule_book_output=tmp[4]
                    )
                    asyn_output = list(asyn_output)
                    print('--------Sequence %s----------' % i_sequence)
                    # debug
                    # print("\n\n")
                    # print(asyn_model.weight.data)
                    # print("\n\n")
                    # print(asyn_output[1].float().data.cpu().numpy())
                    # print("\n\n")

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(tmp[1].float().data.cpu().numpy(),
                                               fb_output_tmp_1.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(tmp[1].float().data.cpu().numpy(),
                                               fb_output_tmp_1.squeeze(0).detach().numpy(), decimal=5)

            print("flag")

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy(), decimal=5)

    def test_reset_two_stacked_conv_ops_against_SCN_conv2d_artificial_input(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        # generate/set seed
        seed = np.random.randint(2**32-1)
        # seed = 3954672100
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print('Asynchronous N-Layer Test')
        # override
        output_spatial_size = [(dim + stride - 1) // stride for dim in spatial_dimensions]
        output_spatial_size = [(dim + stride - 1) // stride for dim in output_spatial_size]
        # ---- Facebook sparse convolution with same padding as ground truth ----
        # main layer
        fb_layer_1 = SparseConvolutionSamePadding(
            dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, bias=False, filter_stride=stride, batch_size=1
        )
        fb_layer_2 = SparseConvolutionSamePadding(
            dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, bias=False, filter_stride=stride, batch_size=1
        )
        # compute required spacial size
        spatial_size = fb_layer_1.input_spatial_size(
            fb_layer_2.input_spatial_size(
                torch.LongTensor(output_spatial_size)
            )
        )
        # print(f"self.spatial_size: {spatial_size}")
        # input layer to convert data to correct format
        input_layer = scn.InputLayer(
            dimension=2, spatial_size=spatial_size, mode=2
        ).to(torch.device("cpu"))
        # sparse to dense layer to retrieve output
        sparse_to_dense = scn.SparseToDense(dimension=2, nPlanes=nOut)

        # async sparse convolution (SC, not VSC)
        asyn_layer_1 = asynNonValidSparseConvolution2D(
            dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, first_layer=True, filter_stride=stride,
            use_bias=False, device=DEVICE
        )
        asyn_layer_2 = asynNonValidSparseConvolution2D(
            dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, first_layer=False, filter_stride=stride,
            use_bias=False, device=DEVICE
        )
        # set weights equal to facebook sparseconv layer
        asyn_layer_1.weight.data = fb_layer_1.weight.squeeze(1).to(torch.device("cpu"))
        asyn_layer_2.weight.data = fb_layer_2.weight.squeeze(1).to(torch.device("cpu"))
        for i_test in tqdm.tqdm(range(10)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')
            sequence_length = 10

            # ---- Create Input ----
            out = test_util.createInput(
                nIn=nIn, spatial_dimensions=spatial_dimensions, asynchronous_input=True,
                sequence_length=sequence_length, simplified=False
            )
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            # prepare data for fb layer
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
            features = torch.tensor(features, dtype=torch.float)
            # compute ground truth
            # compute convolution
            fb_input = input_layer([batch_locations, features])
            fb_tmp_1 = fb_layer_1(fb_input)
            fb_output_tmp_1 = sparse_to_dense(fb_tmp_1)
            fb_tmp_2 = fb_layer_2(fb_tmp_1)
            fb_output = sparse_to_dense(fb_tmp_2)
            # debug
            # print("\n\n")
            # print(sparse_to_dense(input_layer([batch_locations, features])))
            # print("\n\n")
            # print(fb_model.weight.squeeze(1))
            # print("\n\n")
            # print(fb_output)
            # print("\n\n")

            # ---- Asynchronous VGG ----
            with torch.no_grad():
                for i_sequence in range(sequence_length):
                    x_asyn = [None] * 5
                    x_asyn[0] = torch.tensor(asyn_update_locations[i_sequence], dtype=torch.float).to(DEVICE)
                    x_asyn[1] = torch.tensor(asyn_input[i_sequence], dtype=torch.float).to(DEVICE)
                    # debug
                    # print("\n\n")
                    # print(x_asyn[1].float().data.cpu().numpy())
                    # forward
                    tmp = asyn_layer_1.forward(
                        update_location=x_asyn[0].to(torch.device("cpu")),
                        feature_map=x_asyn[1].to(torch.device("cpu")),
                        active_sites_map=None,
                        rule_book_input=None,
                        rule_book_output=None
                    )
                    asyn_output = asyn_layer_2.forward(
                        update_location=tmp[0],
                        feature_map=tmp[1],
                        active_sites_map=tmp[2],
                        rule_book_input=tmp[3],
                        rule_book_output=tmp[4]
                    )
                    asyn_output = list(asyn_output)
                    print('--------Sequence %s----------' % i_sequence)
                    # debug
                    # print("\n\n")
                    # print(asyn_model.weight.data)
                    # print("\n\n")
                    # print(asyn_output[1].float().data.cpu().numpy())
                    # print("\n\n")
                    # print("initial feature map:")
                    # print(x_asyn[1].numpy())
                    # print("\n\n")
                    # print("feature map between layers:")
                    # print(tmp[1].float().data.cpu().numpy())
                    # print("\n\n")
                    # print("\n\n")

            # reset internal state
            asyn_layer_1.reset()
            asyn_layer_2.reset()

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(tmp[1].float().data.cpu().numpy(),
                                               fb_output_tmp_1.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(tmp[1].float().data.cpu().numpy(),
                                               fb_output_tmp_1.squeeze(0).detach().numpy(), decimal=5)

            print("flag")

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy(), decimal=5)


if __name__ == '__main__':
    unittest.main()
