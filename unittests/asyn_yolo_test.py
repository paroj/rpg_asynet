import numpy as np
import torch
import tqdm
import unittest
import sys

from yolo.model_sparse import SparseYolo, AsynSparseYolo
import dataloader.dataset
from dataloader.loader import Loader


DATASET_PATH = "/media/user/Vault_4/HiWi_IGD/KITTI_dataset"  # cluster
# DATASET_PATH = "../../KITTI_Dataset"  # local


class TestAsynYolo(unittest.TestCase):

    def test_asyn_yolo_vs_syn_yolo(self):

        print("WARNING: very slow", file=sys.stderr)

        sequence_count = 1

        device = torch.device('cpu')

        nr_input_channels = 2

        model_for_input_size = SparseYolo(
            nr_classes=7, nr_input_channels=nr_input_channels,
            cnn_spatial_output_size=(5, 7),
            device=device)
        model_input_size = model_for_input_size.get_input_spacial_size()
        # override width/height setting to fit current spec
        print(model_input_size)
        height = int(model_input_size[0])
        width = int(model_input_size[1])

        train_dataset = dataloader.dataset.getDataloader('Kitty_ObjectDetection')(
            DATASET_PATH,
            'all',
            height,
            width,
            'timeframe',
            0,
            42,
            augmentation=False,
            mode='None',
            start_index_factor=0,
            end_index_factor=0.0002,  # only load one sample
            event_representation='histogram',
            shuffle=False
        )

        nr_classes = train_dataset.nr_classes

        train_loader = Loader(
            train_dataset, batch_size=1,
            device=device,
            num_workers=16, pin_memory=False,
            shuffle=False
        )

        syn_model = SparseYolo(
            nr_classes=nr_classes, nr_input_channels=nr_input_channels,
            cnn_spatial_output_size=(5, 7),
            batch_size=1,
            device=device
        )

        asyn_model = AsynSparseYolo(
            nr_classes=nr_classes,
            cnn_spatial_output_size=(
                [5, 7]
            ),
            device=device,
        )

        asyn_model.set_weights_equal(syn_model)

        def denseToSparse(dense_tensor):
            """
            Converts a dense tensor to a sparse vector.

            :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
            :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
            :return features: NumberOfActive x FeatureDimension
            """
            non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
            loc = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

            select_indices = non_zero_indices.split(1, dim=1)
            feat = torch.squeeze(dense_tensor[select_indices], dim=-2)

            return loc, feat

        pbar = tqdm.tqdm(total=len(train_loader), unit='Sample', unit_scale=True, file=sys.stdout)
        for i_batch, sample_batched in enumerate(train_loader):
            print(f"sample {i_batch}/{len(train_loader)}")
            events_batched, _, histogram = sample_batched

            i_sample = 0
            events = events_batched[events_batched[:, 4] == i_sample][:, :4].numpy()

            """
            compute asyn output
            """

            num_events = len(events)
            num_events_increment = num_events / sequence_count
            # for each sequence do:
            for i_sequence in range(sequence_count):
                print(f"sequence {i_sequence}/{sequence_count - 1}")
                events_sequence = events[
                                  int(num_events_increment * i_sequence): int(num_events_increment * (i_sequence + 1))
                                  ]
                if len(events_sequence) == 0:
                    print(f"NO EVENTS IN SEQUENCE {i_sequence} (total events in sample: {num_events})")
                    continue  # skip this sequence
                sequence_histogram = torch.from_numpy(train_dataset.generate_input_representation(
                    events_sequence,
                    (train_dataset.height, train_dataset.width)
                ))
                sequence_histogram = torch.nn.functional.interpolate(
                    sequence_histogram.permute(2, 0, 1).unsqueeze(0),
                    torch.Size(model_input_size)
                )
                sequence_histogram = sequence_histogram.squeeze().permute(1, 2, 0)
                # feed into network
                x_asyn = [None] * 5
                x_asyn[0] = torch.tensor(events_sequence[:, :2], dtype=torch.float).flip(-1).to(device)
                x_asyn[1] = sequence_histogram.float().to(device)
                # only use last output (after all events for this sample have been processed
                if i_sample < sequence_count - 1:
                    _ = asyn_model.forward_async(x_asyn)
                else:
                    asyn_model_output = asyn_model.forward(x_asyn)
            # then 'reset' the network to feed next sample
            asyn_model.reset()

            """
            compute syn output
            """

            # Change size to input size of sparse VGG
            histogram = torch.nn.functional.interpolate(
                histogram.permute(0, 3, 1, 2),
                torch.Size(model_input_size)
            )
            histogram = histogram.permute(0, 2, 3, 1)
            locations, features = denseToSparse(histogram)
            syn_model_output = syn_model([locations, features, histogram.shape[0]])

            """
            compare results
            """

            equal_to_decimal = 4  # reduce due to many layers manipulating the results sequentially
            if syn_model_output.ndim == 4:
                np.testing.assert_almost_equal(asyn_model_output[1].float().data.cpu().numpy(),
                                               syn_model_output.squeeze(0).detach().numpy().transpose(1, 2, 0),
                                               decimal=equal_to_decimal)
            else:
                np.testing.assert_almost_equal(asyn_model_output[1].float().data.cpu().numpy(),
                                               syn_model_output.squeeze(0).detach().numpy(), decimal=equal_to_decimal)

            pbar.update(1)
            # print something to force tqdm to flush the progress bar
            print(" ")
            pbar.refresh()


if __name__ == '__main__':
    unittest.main()
