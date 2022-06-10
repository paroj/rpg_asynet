"""
Example command: python -m unittests.sparse_VGG_test
"""
import random

import numpy as np
import torch
import tqdm
import unittest

MODEL_TO_TEST = "Yolo"  # ['Yolo', 'FB_ObjectDet']
DATASET = 'KittiVision'  # ['KittiVision', 'NCaltech101_ObjectDetection']

from dataloader.dataset import NCaltech101_ObjectDetection
if MODEL_TO_TEST == "Yolo":
    from yolo.model_sparse import AsynSparseYolo as AsynModel
    from yolo.model_sparse import SparseYolo as SynModel
elif MODEL_TO_TEST == 'FB_ObjectDet':
    from models.asyn_sparse_object_det import asynSparseObjectDet as AsynModel
    from models.facebook_sparse_object_det import FBSparseObjectDet as SynModel
else:
    raise ValueError("parameter 'MODEL_TO_TEST' has invalid value.")
from training.trainer import AbstractTrainer
import utils.test_util as test_util

# DEVICE = torch.device("cuda:0")
DEVICE = torch.device("cpu")
# pth = '../log/20201230-174940/checkpoints/model_step_1.pth'
pth = '../log/20210523-220051/checkpoints/model_step_155.pth'

spacial_output_size = [5, 7]


class TestSparseVGG(unittest.TestCase):
    def test_sparse_VGG_artificial_input(self):
        # generate/set seed
        seed = np.random.randint(2**32-1)
        # seed = 2816308579
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        print('Asynchronous N-Layer Test')
        for i_test in tqdm.tqdm(range(1)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')
            if DATASET == 'KittiVision':
                nr_classes = 8
            elif DATASET == 'NCaltech101_ObjectDetection':
                nr_classes = 101
            else:
                raise ValueError("parameter 'DATASET' has invalid value.")
            sequence_length = 4

            # # TODO
            # linear_input_channels = 1024
            # config = [
            #     # (7, 64, 2), 'M',  # 1
            #     # (3, 192), 'M',   # 2
            #     # (1, 128), (3, 256), (1, 256), (3, 512), 'M',  # 3
            #     # [(1, 256), (3, 512), 4], (1, 512), (3, 1024), 'M',  # 4
            #     [(1, 512), (3, 1024), 2],
            #     # (3, 1024),
            #     (3, 1024, 2), # 'M', # 5
            #     (3, 1024), # 'M',
            #     (3, 1024)  # 6
            # ]
            # spacial_output_size = [5, 7]

            # ---- Facebook VGG ----
            if MODEL_TO_TEST == "Yolo":
                fb_model = SynModel(
                    nr_classes=nr_classes, cnn_spatial_output_size=spacial_output_size,
                    # config=config, linear_input_channels=linear_input_channels
                    use_bias=True
                ).eval()
            else:  # elif MODEL_TO_TEST == 'FB_ObjectDet':
                fb_model = SynModel(nr_classes=nr_classes).eval()

            # fb_model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])
            spatial_dimensions = list(fb_model.get_input_spacial_size())

            # ---- Create Input ----
            out = test_util.createInput(nIn=2, spatial_dimensions=spatial_dimensions,
                                        asynchronous_input=True, sequence_length=sequence_length, simplified=False)
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            if MODEL_TO_TEST == "Yolo":
                asyn_model = AsynModel(
                    nr_classes=nr_classes, cnn_spatial_output_size=spacial_output_size, device=DEVICE,
                    # config=config, linear_input_channels=linear_input_channels
                )
                asyn_model.print_layers()  # TODO
            else:  # elif MODEL_TO_TEST == 'FB_ObjectDet':
                asyn_model = AsynModel(nr_classes=nr_classes, device=DEVICE)
            asyn_model.set_weights_equal(fb_model)

            # ---- Facebook VGG ----
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
            features = torch.tensor(features, dtype=torch.float)

            fb_output = fb_model([batch_locations, features])

            # ---- Asynchronous VGG ----
            with torch.no_grad():
                for i_sequence in range(sequence_length):
                    x_asyn = [None] * 5
                    x_asyn[0] = torch.tensor(asyn_update_locations[i_sequence], dtype=torch.float).to(DEVICE)
                    x_asyn[1] = torch.tensor(asyn_input[i_sequence], dtype=torch.float).to(DEVICE)

                    asyn_output = asyn_model.forward(x_asyn)

                    print('--------Sequence %s----------' % i_sequence)

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                               fb_output.squeeze(0).detach().numpy(), decimal=5)

    def test_scn_conv_bias(self):
        # generate/set seed
        seed = np.random.randint(2**32-1)
        # seed = 2816308579
        print(seed)
        # seed RNGs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        print('Asynchronous N-Layer Test')
        for i_test in tqdm.tqdm(range(1)):
            print('Test: %s' % i_test)
            print('#######################')
            print('#       New Test      #')
            print('#######################')
            if DATASET == 'KittiVision':
                nr_classes = 8
            elif DATASET == 'NCaltech101_ObjectDetection':
                nr_classes = 101
            else:
                raise ValueError("parameter 'DATASET' has invalid value.")
            sequence_length = 4

            # # TODO
            # linear_input_channels = 1024
            # config = [
            #     # (7, 64, 2), 'M',  # 1
            #     # (3, 192), 'M',   # 2
            #     # (1, 128), (3, 256), (1, 256), (3, 512), 'M',  # 3
            #     # [(1, 256), (3, 512), 4], (1, 512), (3, 1024), 'M',  # 4
            #     [(1, 512), (3, 1024), 2],
            #     # (3, 1024),
            #     (3, 1024, 2), # 'M', # 5
            #     (3, 1024), # 'M',
            #     (3, 1024)  # 6
            # ]
            # spacial_output_size = [5, 7]

            fb_model = SynModel(
                nr_classes=nr_classes, cnn_spatial_output_size=spacial_output_size,
                # config=config, linear_input_channels=linear_input_channels
                use_bias=True
            ).eval()

            # fb_model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])
            spatial_dimensions = list(fb_model.get_input_spacial_size())

            # ---- Create Input ----
            out = test_util.createInput(nIn=2, spatial_dimensions=spatial_dimensions,
                                        asynchronous_input=True, sequence_length=sequence_length, simplified=False)
            batch_input, batch_update_locations, asyn_input, asyn_update_locations = out

            fb_model_2 = SynModel(
                nr_classes=nr_classes, cnn_spatial_output_size=spacial_output_size,
                # config=config, linear_input_channels=linear_input_channels
                use_bias=False
            ).eval()
            state_dict = fb_model.state_dict()
            # TODO bias initialized to 0, instead of normal distribution -> bias irrelevant in untrained network
            print(state_dict['sparseModel.0.bias'])
            del state_dict['sparseModel.0.bias']
            del state_dict['sparseModel.3.bias']
            del state_dict['sparseModel.7.bias']
            del state_dict['sparseModel.10.bias']
            del state_dict['sparseModel.13.bias']
            del state_dict['sparseModel.16.bias']
            del state_dict['sparseModel.20.bias']
            del state_dict['sparseModel.23.bias']
            del state_dict['sparseModel.26.bias']
            del state_dict['sparseModel.29.bias']
            del state_dict['sparseModel.32.bias']
            del state_dict['sparseModel.35.bias']
            del state_dict['sparseModel.38.bias']
            del state_dict['sparseModel.41.bias']
            del state_dict['sparseModel.44.bias']
            del state_dict['sparseModel.47.bias']
            del state_dict['sparseModel.51.bias']
            del state_dict['sparseModel.54.bias']
            del state_dict['sparseModel.57.bias']
            del state_dict['sparseModel.60.bias']
            del state_dict['sparseModel.63.bias']
            del state_dict['sparseModel.66.bias']
            del state_dict['sparseModel.68.bias']
            del state_dict['sparseModel.71.bias']
            fb_model_2.load_state_dict(state_dict)

            # ---- Facebook VGG ----
            select_indices = tuple(batch_update_locations.T)
            features = batch_input[select_indices]
            batch_locations = torch.tensor(batch_update_locations, dtype=torch.long)
            features = torch.tensor(features, dtype=torch.float)

            fb_output = fb_model([batch_locations, features])

            fb_output_2 = fb_model_2([batch_locations, features])

            if fb_output.ndim == 4:
                np.testing.assert_almost_equal(fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0),
                                               fb_output_2.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
            else:
                np.testing.assert_almost_equal(fb_output.squeeze(0).detach().numpy(),
                                               fb_output_2.squeeze(0).detach().numpy(), decimal=5)

    def test_sparse_VGG_event_input(self):
        """Tests if output of sparse VGG is equivalent to the facebook implementation"""
        print('Asynchronous N-Layer Test')
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

        # ---- Facebook VGG ----
        fb_model = FBSparseObjectDet(nr_classes=nr_classes, small_out_map=small_out_map).eval()
        spatial_dimensions = fb_model.spatial_size

        events, bbs, histogram = train_dataset.__getitem__(0)

        histogram = torch.from_numpy(histogram[np.newaxis, :, :])
        histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(spatial_dimensions))
        histogram = histogram.permute(0, 2, 3, 1)
        locations, features = AbstractTrainer.denseToSparse(histogram)

        # ---- Facebook VGG ----
        fb_model.load_state_dict(torch.load(pth, map_location={'cuda:0': 'cpu'})['state_dict'])

        fb_output = fb_model([locations, features])

        asyn_model = asynSparseObjectDet(nr_classes=nr_classes, small_out_map=small_out_map, device=DEVICE)
        asyn_model.setWeightsEqual(fb_model)

        list_spatial_dimensions = [spatial_dimensions.cpu().numpy()[0], spatial_dimensions.cpu().numpy()[1]]
        input_histogram = torch.zeros(list_spatial_dimensions + [2])
        step_size = nr_last_events // sequence_length

        with torch.no_grad():
            for i_sequence in range(sequence_length):
                new_batch_events = events[(step_size*i_sequence):(step_size*(i_sequence + 1)), :]
                update_locations, new_histogram = asyn_model.generateAsynInput(new_batch_events, spatial_dimensions,
                                                                               original_shape=[height, width])
                input_histogram = input_histogram + new_histogram
                x_asyn = [None] * 5
                x_asyn[0] = update_locations[:, :2].to(DEVICE)
                x_asyn[1] = input_histogram.to(DEVICE)

                asyn_output = asyn_model.forward(x_asyn)
                print('--------Sequence %s----------' % i_sequence)

        if fb_output.ndim == 4:
            np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                           fb_output.squeeze(0).detach().numpy().transpose(1, 2, 0), decimal=5)
        else:
            np.testing.assert_almost_equal(asyn_output[1].float().data.cpu().numpy(),
                                           fb_output.squeeze(0).detach().numpy(), decimal=5)


if __name__ == '__main__':
    unittest.main()
