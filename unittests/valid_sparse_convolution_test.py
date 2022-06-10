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
# from layers.conv_layer_2D import asynNonValidSparseConvolution2D
# from layers.legacy_conv_layer_2D import asynSparseConvolution2D
from layers.asyn_sparse_conv_model import SparseConvolutionSamePadding

from training.trainer import AbstractTrainer
import utils.test_util as test_util

# DEVICE = torch.device("cuda:0")
DEVICE = torch.device("cpu")

spatial_dimensions = [2, 2]
nIn = 1
nOut = 1
filter_size = 3


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
        output_spatial_size = spatial_dimensions
        for i_test in tqdm.tqdm(range(10)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')
            sequence_length = 10

            # ---- Facebook sparse convolution with same padding as ground truth ----
            # main layer
            fb_model = scn.SubmanifoldConvolution(
                dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, bias=False
            )
            # compute padding and required spacial size
            # fb_model.set_return_padded_size(False)
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
                nIn=nIn, spatial_dimensions=spatial_dimensions, asynchronous_input=True,
                sequence_length=sequence_length, simplified=False
            )
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            # async valid sparse convolution (VSC)
            asyn_model = asynSparseConvolution2D(
                dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, first_layer=True,
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
                        update_location=x_asyn[0].long().to(torch.device("cpu")),
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
        # seed = 294013692
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print('Asynchronous N-Layer Test')
        # override
        output_spatial_size = spatial_dimensions
        for i_test in tqdm.tqdm(range(10)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')
            sequence_length = 10

            # ---- Facebook sparse convolution with same padding as ground truth ----
            # main layer
            fb_layer_1 = scn.SubmanifoldConvolution(
                dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, bias=False
            )
            fb_layer_2 = scn.SubmanifoldConvolution(
                dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, bias=False
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

            # async valid sparse convolution (VSC)
            asyn_layer_1 = asynSparseConvolution2D(
                dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, first_layer=True,
                use_bias=False, device=DEVICE
            )
            asyn_layer_2 = asynSparseConvolution2D(
                dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, first_layer=False,
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
                        update_location=x_asyn[0].long().to(torch.device("cpu")),
                        feature_map=x_asyn[1].to(torch.device("cpu")),
                        active_sites_map=None,
                        rule_book_input=None,
                        rule_book_output=None
                    )
                    asyn_output = asyn_layer_2.forward(
                        update_location=tmp[0].long(),
                        feature_map=tmp[1],
                        active_sites_map=tmp[2],
                        rule_book_input=tmp[3],
                        rule_book_output=tmp[4]
                    )
                    asyn_output = list(asyn_output)
                    print('--------Sequence %s----------' % i_sequence)
                    # debug
                    # print("\n\n")
                    # print(x_asyn[0])
                    # print("\n\n")
                    # print(x_asyn[1])
                    # print("\n\n")
                    # print(asyn_layer_1.weight.data)
                    # print("\n\n")
                    # print(tmp[0].double().cpu().numpy())
                    # print("\n\n")
                    # print(tmp[1].float().data.cpu().numpy())
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
        # seed = 294013692
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print('Asynchronous N-Layer Test')
        # override
        output_spatial_size = spatial_dimensions
        # ---- Facebook sparse convolution with same padding as ground truth ----
        # main layer
        fb_layer_1 = scn.SubmanifoldConvolution(
            dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, bias=False
        )
        fb_layer_2 = scn.SubmanifoldConvolution(
            dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, bias=False
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
        # async valid sparse convolution (VSC)
        asyn_layer_1 = asynSparseConvolution2D(
            dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, first_layer=True,
            use_bias=False, device=DEVICE
        )
        asyn_layer_2 = asynSparseConvolution2D(
            dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, first_layer=False,
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
                        update_location=x_asyn[0].long().to(torch.device("cpu")),
                        feature_map=x_asyn[1].to(torch.device("cpu")),
                        active_sites_map=None,
                        rule_book_input=None,
                        rule_book_output=None
                    )
                    asyn_output = asyn_layer_2.forward(
                        update_location=tmp[0].long(),
                        feature_map=tmp[1],
                        active_sites_map=tmp[2],
                        rule_book_input=tmp[3],
                        rule_book_output=tmp[4]
                    )
                    asyn_output = list(asyn_output)
                    print('--------Sequence %s----------' % i_sequence)
                    # debug
                    # print("\n\n")
                    # print(x_asyn[0])
                    # print("\n\n")
                    # print(x_asyn[1])
                    # print("\n\n")
                    # print(asyn_layer_1.weight.data)
                    # print("\n\n")
                    # print(tmp[0].double().cpu().numpy())
                    # print("\n\n")
                    # print(tmp[1].float().data.cpu().numpy())
                    # print("\n\n")
                    # print(asyn_output[1].float().data.cpu().numpy())
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

    def test_reset_stacked_vsc_sc_against_SCN_conv2d_artificial_input(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        # generate/set seed
        seed = np.random.randint(2**32-1)
        # seed = 3175697831  # TODO
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print('Asynchronous N-Layer Test')
        # override
        output_spatial_size = spatial_dimensions
        # ---- Facebook sparse convolution with same padding as ground truth ----
        # main layer
        fb_layer_1 = scn.SubmanifoldConvolution(
            dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, bias=False
        )
        fb_layer_2 = SparseConvolutionSamePadding(
            dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, filter_stride=1, bias=False,
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
        # async valid sparse convolution (VSC)
        asyn_layer_1 = asynSparseConvolution2D(
            dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, first_layer=True,
            use_bias=False, device=DEVICE
        )
        asyn_layer_2 = asynNonValidSparseConvolution2D(
            dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size,  first_layer=True,
            use_bias=False, device=DEVICE
        )
        # set weights equal to facebook sparseconv layer
        asyn_layer_1.weight.data = fb_layer_1.weight.squeeze(1).to(torch.device("cpu"))
        asyn_layer_2.weight.data = fb_layer_2.weight.squeeze(1).to(torch.device("cpu"))
        for i_test in tqdm.tqdm(range(100)):
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
                        update_location=x_asyn[0].long().to(torch.device("cpu")),
                        feature_map=x_asyn[1].to(torch.device("cpu")),
                        active_sites_map=None,
                        rule_book_input=None,
                        rule_book_output=None
                    )
                    asyn_output = asyn_layer_2.forward(
                        update_location=tmp[0].long(),
                        feature_map=tmp[1],
                        active_sites_map=tmp[2],
                        rule_book_input=tmp[3],
                        rule_book_output=tmp[4]
                    )
                    asyn_output = list(asyn_output)
                    print('--------Sequence %s----------' % i_sequence)
                    # debug
                    # print("\n\n")
                    # print(x_asyn[0])
                    # print("\n\n")
                    # print(x_asyn[1])
                    # print("\n\n")
                    # print(asyn_layer_1.weight.data)
                    # print("\n\n")
                    # print(tmp[0].double().cpu().numpy())
                    # print("\n\n")
                    # print(tmp[1].float().data.cpu().numpy())
                    # print("\n\n")
                    # print(asyn_output[1].float().data.cpu().numpy())
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

    def test_reset_stacked_sc_vsc_against_SCN_conv2d_artificial_input(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        # generate/set seed
        seed = np.random.randint(2**32-1)
        # seed = 3175697831  # TODO
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print('Asynchronous N-Layer Test')
        # override
        output_spatial_size = spatial_dimensions
        # ---- Facebook sparse convolution with same padding as ground truth ----
        # main layer
        fb_layer_1 = SparseConvolutionSamePadding(
            dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, filter_stride=1, bias=False,
            batch_size=1
        )
        fb_layer_2 = scn.SubmanifoldConvolution(
            dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size, bias=False
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
        # async valid sparse convolution (VSC)
        asyn_layer_1 = asynNonValidSparseConvolution2D(
            dimension=2, nIn=nIn, nOut=nOut, filter_size=filter_size, first_layer=True,
            use_bias=False, device=DEVICE
        )
        asyn_layer_2 = asynSparseConvolution2D(
            dimension=2, nIn=nOut, nOut=nOut, filter_size=filter_size,  first_layer=True,
            use_bias=False, device=DEVICE
        )
        # set weights equal to facebook sparseconv layer
        asyn_layer_1.weight.data = fb_layer_1.weight.squeeze(1).to(torch.device("cpu"))
        asyn_layer_2.weight.data = fb_layer_2.weight.squeeze(1).to(torch.device("cpu"))
        for i_test in tqdm.tqdm(range(100)):
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
                        update_location=x_asyn[0].long().to(torch.device("cpu")),
                        feature_map=x_asyn[1].to(torch.device("cpu")),
                        active_sites_map=None,
                        rule_book_input=None,
                        rule_book_output=None
                    )
                    asyn_output = asyn_layer_2.forward(
                        update_location=tmp[0].long(),
                        feature_map=tmp[1],
                        active_sites_map=tmp[2],
                        rule_book_input=tmp[3],
                        rule_book_output=tmp[4]
                    )
                    asyn_output = list(asyn_output)
                    print('--------Sequence %s----------' % i_sequence)
                    # debug
                    # print("\n\n")
                    # print(x_asyn[0])
                    # print("\n\n")
                    # print(x_asyn[1])
                    # print("\n\n")
                    # print(asyn_layer_1.weight.data)
                    # print("\n\n")
                    # print(tmp[0].double().cpu().numpy())
                    # print("\n\n")
                    # print(tmp[1].float().data.cpu().numpy())
                    # print("\n\n")
                    # print(asyn_output[1].float().data.cpu().numpy())
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
