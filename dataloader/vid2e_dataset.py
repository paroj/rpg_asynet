from dataset import NCaltech101
import os
import tqdm
import random
import numpy as np
from os import listdir
from os.path import join
import event_representations as er
from numpy.lib import recfunctions as rfn

from dataloader.prophesee import dat_events_tools
from dataloader.prophesee import npy_events_tools




class Vid2ELoader(NCaltech101):
    def __init__(self, root, object_classes, height, width, nr_events_window=-1, augmentation=False, mode='training',
                 event_representation='histogram', shuffle=True):
        """
        Creates an iterator over the N_Caltech101 object recognition dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param event_representation: 'histogram' or 'event_queue'
        """
        if object_classes == 'all':
            self.object_classes = listdir(os.path.join(root, 'Vids'))
        else:
            self.object_classes = object_classes
        self.object_classes.sort()

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.nr_events_window = nr_events_window
        self.nr_classes = len(self.object_classes)
        self.event_representation = event_representation

        self.files = []
        self.class_labels = []
        self.bounding_box_list = []

        self.createDataset()
        self.nr_samples = len(self.files)

        if shuffle:
            zipped_lists = list(zip(self.files,  self.class_labels,  self.bounding_box_list))
            random.shuffle(zipped_lists)
            self.files,  self.class_labels,  self.bounding_box_list = zip(*zipped_lists)

    def createDataset(self):
        """Does a stratified training, testing, validation split"""
        np_random_state = np.random.RandomState(42)
        training_ratio = 0.7
        testing_ratio = 0.2
        # Validation Ratio will be 0.1

        for i_class, object_class in enumerate(self.object_classes):
            if object_class == 'BACKGROUND_Google':
                continue
            dir_path = os.path.join(self.root, 'Vids', object_class)
            sequences = listdir(dir_path)
            nr_samples = len(sequences)

            random_permutation = np_random_state.permutation(nr_samples)
            nr_samples_train = int(nr_samples*training_ratio)
            nr_samples_test = int(nr_samples*testing_ratio)

            if self.mode == 'training':
                start_idx = 0
                end_idx = nr_samples_train
            elif self.mode == 'testing':
                start_idx = nr_samples_train
                end_idx = nr_samples_train + nr_samples_test
            elif self.mode == 'validation':
                start_idx = nr_samples_train + nr_samples_test
                end_idx = nr_samples
            else:
                raise ValueError(f"Unsupported value for mode: '{self.mode}' "
                                 "(must be one of {{'training', 'testing', 'validation'}}).")

            for idx in random_permutation[start_idx:end_idx]:
                self.files.append(os.path.join(self.root, 'Vids', object_class, sequences[idx]))
                annotation_file = f"annotation {sequences[idx][3:]}"
                self.readBoundingBox(os.path.join(self.root, 'Annotations', object_class, annotation_file))
                self.class_labels.append(i_class)

    def readBoundingBox(self, file_path):
        f = open(file_path)
        annotations = np.fromfile(f, dtype=np.int16)
        f.close()
        self.bounding_box_list.append(annotations[2:10])

    def loadEventsFile(self, file_name):
        f = open(file_name, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        all_p = all_p.astype(np.float64)
        all_p[all_p == 0] = -1

        return np.column_stack((all_x, all_y, all_ts, all_p))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        class_label = self.class_labels[idx]
        bounding_box = self.bounding_box_list[idx]
        bounding_box = bounding_box.reshape([4, 2])
        # Set negative corners to zero
        bounding_box = np.maximum(bounding_box, np.zeros_like(bounding_box))
        filename = self.files[idx]
        events = self.loadEventsFile(filename).astype(np.float32)
        nr_events = events.shape[0]

        window_start = 0
        window_end = nr_events
        if self.augmentation:
            window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))
        if self.nr_events_window != -1:
            # Catch case if number of events in batch is lower than number of events in window.
            window_end = min(nr_events, window_start + self.nr_events_window)
        events = events[window_start:window_end, :]

        bounding_box = self.moveBoundingBox(bounding_box, events[-1, 2])

        if self.augmentation:
            events, bounding_box = random_flip_events_along_x(events, resolution=(self.height, self.width),
                                                              bounding_box=bounding_box)
            events, bounding_box = random_shift_events(events, resolution=(self.height, self.width),
                                                       bounding_box=bounding_box)

        # Required Format: ['x', 'y', 'w', 'h', 'class_id'].  (x, y) is top left point\
        new_format_bbox = np.concatenate([bounding_box[0, :], bounding_box[2, :] - bounding_box[0, :],
                                          np.array([class_label])])

        histogram = self.generate_input_representation(events, (self.height, self.width))

        return events, new_format_bbox[np.newaxis, :], histogram

    def moveBoundingBox(self, bounding_box, current_time):
        """
        Move bounding box according the motion of the event camera. Code was adopted from the matlab code provided by
        the dataset authors.
        """
        current_time = float(current_time)
        if current_time < 100e3:
            bounding_box[:, 0] = bounding_box[:, 0] + 3.5 * current_time / 100e3
            bounding_box[:, 1] = bounding_box[:, 1] + 7 * current_time / 100e3
        elif current_time < 200e3:
            bounding_box[:, 0] = bounding_box[:, 0] + 3.5 + 3.5 * (current_time - 100e3) / 100e3
            bounding_box[:, 1] = bounding_box[:, 1] + 7 - 7 * (current_time - 100e3) / 100e3
        elif current_time < 300e3:
            bounding_box[:, 0] = bounding_box[:, 0] + 7 - 7 * (current_time - 200e3) / 100e3
            bounding_box[:, 1] = bounding_box[:, 1]
        else:
            bounding_box[:, 0] = bounding_box[:, 0]
            bounding_box[:, 1] = bounding_box[:, 1]

        bounding_box = np.maximum(bounding_box, np.zeros_like(bounding_box))
        bounding_box[:, 0] = np.minimum(bounding_box[:, 0], np.ones_like(bounding_box[:, 0]) * self.width - 1)
        bounding_box[:, 1] = np.minimum(bounding_box[:, 1], np.ones_like(bounding_box[:, 1]) * self.height - 1)

        return bounding_box


