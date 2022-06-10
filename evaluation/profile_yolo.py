"""
mimics one training epoch of selected networks, but only computing teh parts necessary for profiling the 'forward'
function(s)
"""

from config.settings import Settings
import torch
import torch.nn.functional
import dataloader.dataset
from dataloader.loader import Loader
import config.settings
from yolo.model_sparse import SparseYolo, AsynSparseYolo
from yolo.model import Darknet as DenseYolo, build_darknet as create_dense_yolo
import os
import sys
import tqdm
import datetime

import cProfile


# smaller network to avoid OOM on unoptimized async implementation
# CONFIG = [
#     (7, 32, 2), 'M',  # 1
#     # (3, 32), 'M',   # 2
#     # (1, 32),  'M',  # 4
#     # (3, 64, 2),  # 5
#     (3, 1024)
# ]

"""
YOLOv1 config
"""
CONFIG = [
    (7, 64, 2), 'M',  # 1
    (3, 192), 'M',   # 2
    (1, 128), (3, 256), (1, 256), (3, 512), 'M',  # 3
    [(1, 256), (3, 512), 4], (1, 512), (3, 1024), 'M',  # 4
    [(1, 512), (3, 1024), 2], (3, 1024), (3, 1024, 2),  # 5
    (3, 1024), (3, 1024)  # 6
]


class SettingsOverrideDataset(config.settings.Settings):

    def __init__(self, settings_yaml, generate_log=True):
        super().__init__(settings_yaml, generate_log)
        self.gpu_device = torch.device("cpu")

    def load_dataset(self, dataset: dict, dataset_name: str) -> dict:
        return dataset['kitty_objectdetection']


class YoloProfiler:

    def __init__(self, model: str, with_grad_or_cpp: bool, batch_size_or_sequence_count: int):

        settings_filepath = os.path.abspath("/code/config/settings.yaml")
        if not os.path.isfile(settings_filepath):
            settings_filepath_local = os.path.abspath("config/settings.yaml")
            if os.path.isfile(settings_filepath_local):
                settings_filepath = settings_filepath_local

        self.settings = SettingsOverrideDataset(settings_filepath, generate_log=True)
        self.model_name = model

        if self.settings.event_representation == 'histogram':
            self.nr_input_channels = 2
        elif self.settings.event_representation == 'dense_image':
            self.nr_input_channels = 3

        if self.model_name == 'yolo_asyn_sparse':
            self.cpp = with_grad_or_cpp
            self.with_grad = None
            self.sequence_count = batch_size_or_sequence_count
            self.settings.batch_size = 1
        else:
            self.cpp = None
            self.with_grad = with_grad_or_cpp
            self.batch_size = batch_size_or_sequence_count
            self.settings.batch_size = self.batch_size

        model_for_input_size = SparseYolo(
            config=CONFIG,
            nr_classes=7, nr_input_channels=self.nr_input_channels,
            cnn_spatial_output_size=(5, 7),
            device=self.settings.gpu_device)
        self.model_input_size = model_for_input_size.get_input_spacial_size()
        # override width/height setting to fit current spec
        print(self.model_input_size)
        self.settings.height = int(self.model_input_size[0])
        self.settings.width = int(self.model_input_size[1])

        self.dataset_builder = dataloader.dataset.getDataloader(self.settings.dataset_name)
        self.dataset_loader = Loader

        if False:  # self.model_name in ['dense_yolo', 'yolo_fb_sparse']:  # TODO
            self.train_dataset = self.dataset_builder(self.settings.dataset_path,
                                                      self.settings.object_classes,
                                                      self.settings.height,
                                                      self.settings.width,
                                                      self.settings.event_window_mode,
                                                      self.settings.nr_events_window,
                                                      self.settings.timeframe_events_window,
                                                      augmentation=False,
                                                      mode='training',
                                                      event_representation=self.settings.event_representation,
                                                      shuffle=False)
        elif self.model_name in ['yolo_fb_sparse', 'yolo_asyn_sparse', 'dense_yolo']:  #  == 'yolo_asyn_sparse':  # TODO
            self.train_dataset = self.dataset_builder(self.settings.dataset_path,
                                                      self.settings.object_classes,
                                                      self.settings.height,
                                                      self.settings.width,
                                                      self.settings.event_window_mode,
                                                      self.settings.nr_events_window,
                                                      self.settings.timeframe_events_window,
                                                      augmentation=False,
                                                      mode='None',
                                                      start_index_factor=0,
                                                      end_index_factor=0.02,  # 148: 0.02,  # 1042: 0.14,  # TODO 0.05
                                                      event_representation=self.settings.event_representation,
                                                      shuffle=False)
        else:
            raise ValueError(f"Invalid value for model name: '{self.model_name}'.")

        self.nr_train_epochs = int(self.train_dataset.nr_samples / self.settings.batch_size) + 1
        self.nr_classes = self.train_dataset.nr_classes
        self.object_classes = self.train_dataset.object_classes

        self.train_loader = self.dataset_loader(self.train_dataset, batch_size=self.settings.batch_size,
                                                device=self.settings.gpu_device,
                                                num_workers=self.settings.num_cpu_workers, pin_memory=False,
                                                shuffle=False)

        if self.model_name == 'yolo_fb_sparse':
            self.model = SparseYolo(
                config=CONFIG,
                nr_classes=self.nr_classes, nr_input_channels=self.nr_input_channels,
                cnn_spatial_output_size=(5, 7),
                batch_size=self.settings.batch_size,
                device=self.settings.gpu_device
            )
            self.model.to(self.settings.gpu_device)

        elif self.model_name == 'yolo_asyn_sparse':
            self.model = AsynSparseYolo(
                config=CONFIG,
                nr_classes=self.nr_classes,
                cnn_spatial_output_size=(
                    [5, 7]
                ),
                # cnn_spatial_output_size=(  # TODO
                #     [30, 42]
                # ),
                device=self.settings.gpu_device,
                cpp=self.cpp,
                # linear_input_channels=2  # TODO remove
            )
        elif self.model_name == 'dense_yolo':
            self.model = DenseYolo(
                config=CONFIG,
                nr_classes=self.nr_classes, nr_input_channels=self.nr_input_channels,
                cnn_spatial_output_size=(
                    [5, 7]
                ))
            self.model.to(self.settings.gpu_device)
        else:
            raise ValueError(f"Invalid value for model name: '{self.model_name}'.")

    def profile_one_epoch(self):
        if self.with_grad:
            with torch.no_grad():
                self.trainEpoch()
        else:
            self.trainEpoch()

    @staticmethod
    def denseToSparse(dense_tensor):
        """
        Converts a dense tensor to a sparse vector.

        :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
        :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
        :return features: NumberOfActive x FeatureDimension
        """
        non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
        locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

        select_indices = non_zero_indices.split(1, dim=1)
        features = torch.squeeze(dense_tensor[select_indices], dim=-2)

        return locations, features

    def trainEpoch(self):
        if self.model_name == 'yolo_fb_sparse':
            self.train_epoch_sparse()
        elif self.model_name == 'yolo_asyn_sparse':
            self.train_epoch_asyn_sparse()
        else:
            self.train_epoch_dense()

    def train_epoch_sparse(self):
        self.pbar = tqdm.tqdm(total=len(self.train_loader), unit='Batch', unit_scale=True, file=sys.stdout)
        for i_batch, sample_batched in enumerate(self.train_loader):
            print(f"batch {i_batch + 1}/{len(self.train_loader)}")
            event, bounding_box, histogram = sample_batched

            # confirm the same samples are used across different runs/models -> no shuffling or augmentation
            # import hashlib, pprint
            # print(hashlib.md5(pprint.pformat(sample_batched).encode('utf-8')).hexdigest())

            # Change size to input size of sparse VGG
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))
            histogram = histogram.permute(0, 2, 3, 1)
            # Change x, width and y, height
            bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[1].float()
                                          / self.settings.width).long()
            bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[0].float()
                                          / self.settings.height).long()
            locations, features = self.denseToSparse(histogram)
            model_output = self.model([locations, features, histogram.shape[0]])
            self.pbar.update(1)
            # print something to force tqdm to flush the progress bar
            print(" ")
            self.pbar.refresh()

    def train_epoch_asyn_sparse(self):
        self.pbar = tqdm.tqdm(total=len(self.train_loader), unit='Sample', unit_scale=True, file=sys.stdout)
        for i_batch, sample_batched in enumerate(self.train_loader):
            print(f"sample {i_batch}/{len(self.train_loader)}")
            events_batched, _, histogram = sample_batched

            # confirm the same samples are used across different runs/models -> no shuffling or augmentation
            # import hashlib, pprint
            # print(hashlib.md5(pprint.pformat(sample_batched).encode('utf-8')).hexdigest())

            # for each sample do:
            # assert self.settings.batch_size == 1
            for i_sample in range(self.settings.batch_size):
                # print(f"sample {i_sample}/{self.settings.batch_size - 1}")
                events = events_batched[events_batched[:, 4] == i_sample][:, :4].numpy()
                num_events = len(events)
                # print(f"total events: {num_events}")
                if num_events == 0:
                    print("EMPTY SAMPLE.")
                    continue
                if num_events == 1:
                    print("FAULTY SAMPLE.")
                    print(events)
                    print(events_batched)
                    continue
                num_events_increment = num_events / self.sequence_count
                # for each sequence do:
                for i_sequence in range(self.sequence_count):
                    # print(f"sequence {i_sequence}/{self.sequence_count - 1}")
                    events_sequence = events[
                        int(num_events_increment * i_sequence): int(num_events_increment * (i_sequence + 1))
                    ]
                    # print(f"events in sequence: {len(events_sequence)}")
                    # print(f"events in sequence: [{int(num_events_increment * i_sequence)}: {int(num_events_increment * (i_sequence + 1))}]")
                    # assert events_sequence is not None
                    # assert self.train_dataset.generate_input_representation(
                    #     events_sequence,
                    #     (self.train_dataset.height, self.train_dataset.width)
                    # ) is not None
                    if len(events_sequence) == 0:
                        print(f"NO EVENTS IN SEQUENCE {i_sequence} (total events in sample: {num_events})")
                        continue  # skip this sequence
                    histogram = torch.from_numpy(self.train_dataset.generate_input_representation(
                        events_sequence,
                        (self.train_dataset.height, self.train_dataset.width)
                    ))
                    histogram = torch.nn.functional.interpolate(histogram.permute(2, 0, 1).unsqueeze(0),
                                                                torch.Size(self.model_input_size))
                    histogram = histogram.squeeze().permute(1, 2, 0)
                    # feed into network
                    x_asyn = [None] * 5
                    x_asyn[0] = torch.tensor(events_sequence[:, :2], dtype=torch.float).flip(-1).to(self.settings.gpu_device)
                    x_asyn[1] = histogram.float().to(self.settings.gpu_device)
                    # only use last output (after all events for this sample have been processed
                    if i_sample < self.sequence_count - 1:
                        model_output = self.model.forward_async(x_asyn)
                    else:
                        model_output = self.model.forward(x_asyn)
                    del model_output  # assist with cleanup
                # then 'reset' the network to feed next sample
                self.model.reset()

            self.pbar.update(1)
            # print something to force tqdm to flush the progress bar
            print(" ")
            self.pbar.refresh()

    def train_epoch_dense(self):
        self.pbar = tqdm.tqdm(total=len(self.train_loader), unit='Batch', unit_scale=True, file=sys.stdout)
        for i_batch, sample_batched in enumerate(self.train_loader):
            print(f"batch {i_batch}/174")
            event, bounding_box, histogram = sample_batched
            # Change size to input size of sparse VGG
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))
            # Change x, width and y, height
            bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[1].float()
                                       / self.settings.width).long()
            bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[0].float()
                                       / self.settings.height).long()
            # Deep Learning Magic
            model_output = self.model(histogram)
            self.pbar.update(1)
            # print something to force tqdm to flush the progress bar
            print(" ")
            self.pbar.refresh()


def main():

    counter = 0
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    logdir = None
    while counter >= 0:
        try:
            if counter == 0:
                logdir = f"./profiling_results_{timestamp}"
            else:
                logdir = f"./profiling_results_{timestamp}_{counter}"
            print(os.path.abspath(logdir))
            os.makedirs(logdir)
            counter = -1
        except FileExistsError:
            counter += 1

    for model, with_grad_or_cpp, batch_size_or_sequence_count, name in [
        # ('yolo_fb_sparse', True, 30, 'syn_yolo_ca1000_samples'),
        # ('yolo_fb_sparse', True, 1, 'syn_yolo_ca1000_samples'),
        # ('dense_yolo', True, 1, 'dense_yolo_ca1000_samples'),
        # ('dense_yolo', True, 30, 'dense_yolo_ca1000_samples'),

        # ('dense_yolo', False, 30),
        # ('yolo_asyn_sparse', False, 2),
        # ('yolo_fb_sparse', False, 1, 'syn_yolo_14_samples'),

        # ('yolo_asyn_sparse', False, 2, 'asyn_optimized_yolo_148_samples'),
        # ('yolo_asyn_sparse', False, 5, 'asyn_optimized_yolo_148_samples'),
        # ('yolo_asyn_sparse', False, 1, 'asyn_optimized_yolo_148_samples'),
        # ('yolo_asyn_sparse', False, 10, 'asyn_optimized_yolo_148_samples'),
        # ('yolo_asyn_sparse', False, 1000, 'asyn_optimized_yolo_148_samples'),

        # ('yolo_asyn_sparse', False, 2, 'asyn_optimized_yolo_148_samples'),
        ('yolo_asyn_sparse', False, 3, 'asyn_optimized_yolo_148_samples'),
        ('yolo_asyn_sparse', False, 4, 'asyn_optimized_yolo_148_samples'),
        # ('yolo_asyn_sparse', False, 5, 'asyn_optimized_yolo_148_samples'),
        # ('yolo_asyn_sparse', False, 10, 'asyn_optimized_yolo_14_samples'),
        # ('yolo_asyn_sparse', False, 1000, 'asyn_optimized_yolo_148_samples'),

        # ('yolo_fb_sparse', False, 7, 'syn_yolo_14_samples'),
        # ('yolo_asyn_sparse', False, 2, 'asyn_yolo_14_samples'),
        # ('yolo_fb_sparse', False, 7, 'syn_yolo_14_samples'),
        # ('yolo_asyn_sparse', False, 5),
        # ('yolo_asyn_sparse', False, 10),
        # ('yolo_asyn_sparse', True, 1, 'asyn_cpp_submanifold_only_74samples'),
        # ('yolo_asyn_sparse', False, 1, 'asyn_submanifold_only_74samples'),
    ]:
        print(f"{model}, {with_grad_or_cpp}, {batch_size_or_sequence_count}")
        profiler = YoloProfiler(
            model=model,
            with_grad_or_cpp=with_grad_or_cpp,
            batch_size_or_sequence_count=batch_size_or_sequence_count
        )
        cProfile.runctx(
            "profiler.profile_one_epoch()",
            globals(), locals(),
            os.path.join(logdir, '_'.join((name, str(with_grad_or_cpp), str(batch_size_or_sequence_count))))
        )


if __name__ == "__main__":
    main()
