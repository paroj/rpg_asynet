import os
import tqdm
import random
import numpy as np
from os import listdir
from os.path import join
import event_representations as er
from numpy.lib import recfunctions as rfn
import sys
import torch

from dataloader.prophesee import dat_events_tools
from dataloader.prophesee import npy_events_tools

import utils.get_image_size
from convert_to_events import ImageSequenceToEventConverter
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn.functional

from abc import ABC, abstractmethod


def random_shift_events(events, max_shift=20, resolution=(180, 240), bounding_box=None):
    H, W = resolution
    if bounding_box is not None:
        if isinstance(bounding_box, list):
            min_x_bb = min((bb[0,0] for bb in bounding_box))
            min_y_bb = min((bb[0,1] for bb in bounding_box))
            max_x_bb = max((bb[2,0] for bb in bounding_box))
            max_y_bb = max((bb[2,1] for bb in bounding_box))
            x_shift = np.random.randint(-min(min_x_bb, max_shift),
                                        min(W - max_x_bb, max_shift), size=(1,))
            y_shift = np.random.randint(-min(min_y_bb, max_shift),
                                        min(H - max_y_bb, max_shift), size=(1,))
            for single_bounding_box in bounding_box:
                single_bounding_box[:, 0] += x_shift
                single_bounding_box[:, 1] += y_shift
        else:
            x_shift = np.random.randint(-min(bounding_box[0, 0], max_shift),
                                        min(W - bounding_box[2, 0], max_shift), size=(1,))
            y_shift = np.random.randint(-min(bounding_box[0, 1], max_shift),
                                        min(H - bounding_box[2, 1], max_shift), size=(1,))
            bounding_box[:, 0] += x_shift
            bounding_box[:, 1] += y_shift
    else:
        x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))

    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    if bounding_box is None:
        return events

    return events, bounding_box


def random_flip_events_along_x(events, resolution=(180, 240), p=0.5, bounding_box=None):
    H, W = resolution
    flipped = False
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
        flipped = True

    if bounding_box is None:
        return events

    if flipped:
        if isinstance(bounding_box, list):
            bounding_box = [flip_bounding_box(W, single_bounding_box) for single_bounding_box in bounding_box]
        else:
            bounding_box = flip_bounding_box(W, bounding_box)
    return events, bounding_box


def flip_bounding_box(W: int, bounding_box: np.ndarray):
    bounding_box[:, 0] = W - 1 - bounding_box[:, 0]
    return bounding_box[[1, 0, 3, 2]]


def getDataloader(name):
    dataset_dict = {'NCaltech101': NCaltech101,
                    'NCaltech101_ObjectDetection': NCaltech101_ObjectDetection,
                    'Kitty_ObjectDetection': KittyObjectDetection,
                    'Kitty_ObjectDetection_Image': KittyObjectDetectionImages,
                    'Prophesee': Prophesee,
                    'NCars': NCars}
    return dataset_dict.get(name)


class DatasetBase(ABC):

    def __init__(self, root, object_classes, height, width, event_window_mode='fixed', nr_events_window=-1,
                 timeframe_events_window=0, augmentation=False, mode='training', start_index_factor: float = None,
                 end_index_factor: float = None, event_representation='histogram',
                 shuffle=True):
        """
        Creates an iterator over the N_Caltech101 dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param start_index_factor: start factor of the range of selected samples, requires mode=None, disables order
                                   permutation of the samples
        :param end_index_factor: start factor of the range of selected samples, requires mode=None, disables order
                                 permutation of the samples
        :param event_representation: 'histogram' or 'event_queue'
        """
        self.root = root
        self.object_classes = object_classes
        self.mode = mode
        self.end_index_factor = end_index_factor
        self.start_index_factor = start_index_factor
        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.event_window_mode = event_window_mode
        self.nr_events_window = nr_events_window
        self.timeframe_events_window = timeframe_events_window
        self.event_representation = event_representation
        self.shuffle = shuffle
        # need to be initialized by each subclass
        self.nr_classes = -1
        self.files = []

    def __len__(self):
        return len(self.files)

    def generate_input_representation(self, events, shape):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        if self.event_representation == 'histogram':
            return self.generate_event_histogram(events, shape)
        elif self.event_representation == 'event_queue':
            return self.generate_event_queue(events, shape)

    @staticmethod
    def generate_event_histogram(events, shape):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events.T
        x = x.astype(np.int)
        y = y.astype(np.int)

        img_pos = np.zeros((H * W,), dtype="float32")
        img_neg = np.zeros((H * W,), dtype="float32")

        np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
        np.add.at(img_neg, x[p == -1] + W * y[p == -1], 1)

        histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, 2))

        return histogram

    @staticmethod
    def generate_event_queue(events, shape, K=15):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        H, W = shape
        events = events.astype(np.float32)

        if events.shape[0] == 0:
            return np.zeros([H, W, 2*K], dtype=np.float32)

        # [2, K, height, width],  [0, ...] time, [:, 0, :, :] newest events
        four_d_tensor = er.event_queue_tensor(events, K, H, W, -1).astype(np.float32)

        # Normalize
        four_d_tensor[0, ...] = four_d_tensor[0, 0, None, :, :] - four_d_tensor[0, :, :, :]
        max_timestep = np.amax(four_d_tensor[0, :, :, :], axis=0, keepdims=True)

        # four_d_tensor[0, ...] = np.divide(four_d_tensor[0, ...], max_timestep, where=max_timestep.astype(np.bool))
        four_d_tensor[0, ...] = four_d_tensor[0, ...] / (max_timestep + (max_timestep == 0).astype(np.float))

        return four_d_tensor.reshape([2*K, H, W]).transpose(1, 2, 0)

    @abstractmethod
    def __getitem__(self, idx):
        pass


class EventDatasetBase(DatasetBase):

    @abstractmethod
    def loadEventsFile(self, file_name):
        pass


class ObjectDetDatasetBase(EventDatasetBase):

    @abstractmethod
    def createDataset(self):
        pass

    def apply_event_window(self, events):
        nr_events = events.shape[0]
        # apply window based on selected mode
        if self.event_window_mode == 'timeframe':  # use last [x]]ms (33ms == 1 frame) worth of events
            # check non decreasing timestamps to simplify following filtering task
            # assert all(x <= y for x, y in zip([event[2] for event in events], [event[2] for event in events][1:]))  # TODO
            # timestamp is microseconds, use 33 last milliseconds
            # print(f"using {self.timeframe_events_window}ms window")
            timeframe_ms = self.timeframe_events_window
            timestamp_threshold = events[-1][2] - timeframe_ms * 1000  # starting timestamp for event window
            if events[0][2] >= timestamp_threshold:
                # all events after the 33ms window
                upper = 0
            elif events[-1][2] < timestamp_threshold:
                # all events before the 33ms window
                upper = len(events)
            else:
                # all events cover 125 milliseconds, use informed guess as starting point for binary search
                starting_point = int((len(events) - 1) * (125 - timeframe_ms) / 125)
                # first find lower using informed guess:
                lower = starting_point
                upper = starting_point
                while events[lower][2] >= timestamp_threshold:
                    upper = lower
                    lower = lower - int((len(events) - 1) * 1 / 125)
                # lower found, find upper using informed guess:
                while events[upper][2] < timestamp_threshold:
                    lower = upper
                    upper = upper + int((len(events) - 1) * 1 / 125)
                # upper and lower found, find target with binary search
                while upper - lower > 1:
                    current = lower + (upper - lower) // 2
                    if events[current][2] >= timestamp_threshold:
                        upper = current
                    else:
                        lower = current
            # threshold found, upper is first index with value above threshold
            events = events[upper:, :]
            # print(f"window_size: {len(events)}")
        else:
            # print("using fixed event window")
            window_start = 0
            window_end = nr_events
            if self.augmentation:
                window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))
            if self.nr_events_window != -1:
                # Catch case if number of events in batch is lower than number of events in window.
                window_end = min(nr_events, window_start + self.nr_events_window)
            events = events[window_start:window_end, :]
        return events

    def augment(self, events, bounding_boxes):
        if self.augmentation:
            events, bounding_boxes = random_flip_events_along_x(events, resolution=(self.height, self.width),
                                                                bounding_box=bounding_boxes)
            events, bounding_boxes = random_shift_events(events, resolution=(self.height, self.width),
                                                         bounding_box=bounding_boxes)
        return events, bounding_boxes

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


class NCaltech101(DatasetBase):

    def __init__(self, *args, **kwargs):
        """
        Creates an iterator over the N_Caltech101 dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param event_representation: 'histogram' or 'event_queue'
        """
        super().__init__(*args, **kwargs)
        root = os.path.join(self.root, self.mode)

        if self.object_classes == 'all':
            self.object_classes = listdir(root)
        self.labels = []

        for i, object_class in enumerate(self.object_classes):
            new_files = [join(root, object_class, f) for f in listdir(join(root, object_class))]
            self.files += new_files
            self.labels += [i] * len(new_files)

        self.nr_samples = len(self.labels)

        if self.shuffle:
            zipped_lists = list(zip(self.files, self.labels))
            random.seed(7)
            random.shuffle(zipped_lists)
            self.files, self.labels = zip(*zipped_lists)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        filename = self.files[idx]
        events = np.load(filename).astype(np.float32)
        nr_events = events.shape[0]

        window_start = 0
        window_end = nr_events
        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)
            window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))

        if self.nr_events_window != -1:
            # Catch case if number of events in batch is lower than number of events in window.
            window_end = min(nr_events, window_start + self.nr_events_window)

        events = events[window_start:window_end, :]

        histogram = self.generate_input_representation(events, (self.height, self.width))

        return events, label, histogram


class NCaltech101_ObjectDetection(ObjectDetDatasetBase):

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        if self.object_classes == 'all':
            self.object_classes = listdir(os.path.join(self.root, 'Caltech101'))
        self.object_classes.sort()

        self.nr_classes = len(self.object_classes)

        self.class_labels = []
        self.bounding_box_list = []

        self.createDataset()
        self.nr_samples = len(self.files)

        if self.shuffle:
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
            dir_path = os.path.join(self.root, 'Caltech101', object_class)
            image_files = listdir(dir_path)
            nr_samples = len(image_files)

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

            for idx in random_permutation[start_idx:end_idx]:
                self.files.append(os.path.join(self.root, 'Caltech101', object_class, image_files[idx]))
                annotation_file = 'annotation' + image_files[idx][5:]
                self.readBoundingBox(os.path.join(self.root, 'Caltech101_annotations', object_class, annotation_file))
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

        events = self.apply_event_window(events)

        bounding_box = self.moveBoundingBox(bounding_box, events[-1, 2])

        events, bounding_box = self.augment(events, bounding_boxes=bounding_box)

        # Required Format: ['x', 'y', 'w', 'h', 'class_id'].  (x, y) is top left point\
        new_format_bbox = np.concatenate([bounding_box[0, :], bounding_box[2, :] - bounding_box[0, :],
                                          np.array([class_label])])

        histogram = self.generate_input_representation(events, (self.height, self.width))

        return events, new_format_bbox[np.newaxis, :], histogram


class KittyObjectDetection(ObjectDetDatasetBase):

    def __init__(self, *args, **kwargs):
        """
        Creates an iterator over the eventified Kitty object detection dataset.

        :param root: path to dataset root
        :param object_classes: 'all' for all classes
        :param height: target height of dataset image
        :param width: target width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param event_representation: 'histogram' or 'event_queue'
        """
        super().__init__(*args, **kwargs)

        self.all_object_classes = [
            'Car', 'Van', 'Truck',
            'Tram',
            'Cyclist',
            'Pedestrian', 'Person_sitting',
            'Misc',
            'DontCare'
        ]
        self.default_ignored_object_classes = ['Misc', 'DontCare']
        if self.object_classes == 'all':
            self.object_classes = self.all_object_classes
            self.ignored_object_classes = self.default_ignored_object_classes
        else:
            self.ignored_object_classes = [
                object_class
                for object_class
                in self.default_ignored_object_classes
                if object_class not in self.object_classes
            ]
        self.object_classes.sort()

        self.max_number_bounding_boxes_per_sample = 24  # maximum number of BBs for single sample in dataset

        # TODO
        self.converter = ImageSequenceToEventConverter(
            base_folder=os.path.join(self.root, 'training/'),
            on_demand_mode=True
        )

        self.nr_classes = len(self.object_classes)

        # event file of each sample
        # handled in base class: self.files = []
        # for each sample: list containing id of class label (index in alphabetically sorted list of all labels)
        self.class_id_list_list = []
        # for each sample: list containing BB in same format as NCaltech101_ObjectDet BB
        self.bounding_box_list_list = []

        self.createDataset()
        self.nr_samples = len(self.files)

        if self.shuffle:
            zipped_lists = list(zip(self.files, self.class_id_list_list, self.bounding_box_list_list))
            random.shuffle(zipped_lists)
            self.files, self.class_id_list_list, self.bounding_box_list_list = zip(*zipped_lists)

    def createDataset(self):
        """Does a stratified training, testing, validation split"""
        np_random_state = np.random.RandomState(42)
        training_ratio = 0.7
        testing_ratio = 0.2
        # Validation Ratio will be 0.1

        # for i_class, object_class in enumerate(self.object_classes):
        #     if object_class == 'BACKGROUND_Google':
        #         continue
        events_dir_path = os.path.join(self.root, 'training/events_2')
        label_dir_path = os.path.join(self.root, 'training/label_2')

        # TODO
        # image_files = listdir(events_dir_path)
        image_files = self.converter.list_available_sequences()
        nr_samples = len(image_files)

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
        elif self.end_index_factor is not None:
            if self.start_index_factor is not None:
                start_idx = int(nr_samples * self.start_index_factor)
            else:
                start_idx = 0
            end_idx = int(nr_samples * self.end_index_factor)
            print(f"Selecting fixed amount of samples: [{start_idx}:{end_idx}].")
            # override random permutation to get the specified samples
            random_permutation = range(nr_samples)
        else:
            raise ValueError(f"invalid value passed for 'mode': {self.mode}.")

        for idx in random_permutation[start_idx:end_idx]:
            # self.files.append(os.path.join(events_dir_path, image_files[idx]))
            # filename, _ = os.path.splitext(image_files[idx])
            self.files.append(image_files[idx])
            filename, _ = os.path.splitext(os.path.split(image_files[idx][-1])[1])
            annotation_file = f"{filename}.txt"
            bounding_box_list, class_id_list = self.read_bounding_box_and_class_labels(
                os.path.join(label_dir_path, annotation_file)
            )
            self.bounding_box_list_list.append(bounding_box_list)
            self.class_id_list_list.append(class_id_list)

    def read_bounding_box_and_class_labels(self, file_path):
        # logic taken from https://github.com/umautobots/vod-converter/blob/master/vod_converter/kitty.py
        with open(file_path, "r") as f:
            annotations = f.readlines()
        bounding_box_list = []
        class_id_list = []
        for annotation in annotations:
            data = annotation.split(' ')
            class_label = data[0]
            if class_label in self.ignored_object_classes:
                continue
            if class_label not in self.all_object_classes:
                print(f"ignoring unknown class label: {class_label} (file {file_path})", file=sys.stderr)
                continue
            class_label_id = self.object_classes.index(class_label)
            x1 = np.int16(data[4][:-3])
            y1 = np.int16(data[5][:-3])
            x2 = np.int16(data[6][:-3])
            y2 = np.int16(data[7][:-3])
            bounding_box = np.asarray(
                (
                    x1, y1,
                    x2, y1,
                    x2, y2,
                    x1, y2
                ), dtype=np.int16
            )
            bounding_box_list.append(bounding_box)
            class_id_list.append(class_label_id)
        return bounding_box_list, class_id_list

    def loadEventsFile(self, file_name):
        with open(file_name, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.float64)

        all_x = np.uint32(raw_data[0::4])
        all_y = np.uint32(raw_data[1::4])
        all_p = raw_data[3::4]  # already is np.float64, already is +1 or -1
        all_ts = np.uint32(raw_data[2::4] * 1000000)  # convert seconds to microseconds

        return np.column_stack((all_x, all_y, all_ts, all_p))

    def load_events_file_on_demand(self, file_name):

        raw_data = self.converter.convert_one_sequence(file_name)

        all_x = np.uint32(raw_data[:, 0])
        all_y = np.uint32(raw_data[:, 1])
        all_p = raw_data[:, 3]  # already is np.float64, already is +1 or -1
        all_ts = np.uint32(raw_data[:, 2] * 1000000)  # convert seconds to microseconds

        return np.column_stack((all_x, all_y, all_ts, all_p))

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        class_ids = self.class_id_list_list[idx]
        bounding_boxes = self.bounding_box_list_list[idx]
        bounding_boxes = [bounding_box.reshape([4, 2]) for bounding_box in bounding_boxes]
        # Set negative corners to zero
        bounding_boxes = [np.maximum(bounding_box, np.zeros_like(bounding_box)) for bounding_box in bounding_boxes]
        filename = self.files[idx]
        events = self.load_events_file_on_demand(filename)  # .astype(np.float32)  # TODO

        # # debug TODO
        # events_old = np.copy(events)
        # bb_old = np.copy(bounding_boxes)

        # interpolate events and BBs to common "source image size"
        width, height = utils.get_image_size.get_image_size(filename[0])
        # '-1' to accurately scale the largest possible coordinate, which is size-1
        # without that, the maximum coordinate '(width - 1)' would be transformed to 'self.width - self.width/width',
        # which rounds to 'self.width' (self.width < 0.5*width) and causes an ArrayIndexOutOfBoundsError
        width_factor = (self.width - 1) / (width - 1)
        height_factor = (self.height - 1) / (height - 1)
        # bounding_boxes: [[x1, y1], [x2, y2], ...], dtype=np.int16
        bb_scale_factor = np.array((width_factor, height_factor))
        bounding_boxes = [np.around(bounding_box * bb_scale_factor).astype(np.int16) for bounding_box in bounding_boxes]
        # events: [[x1, y1, t1, p1], ...], dtype=np.float32, but only integers
        event_scale_factor = np.array((width_factor, height_factor, 1, 1))
        events = np.around(events * event_scale_factor).astype(np.float32)

        events = self.apply_event_window(events)

        bounding_boxes = [self.moveBoundingBox(bounding_box, events[-1, 2]) for bounding_box in bounding_boxes]

        events, bounding_boxes = self.augment(events=events, bounding_boxes=bounding_boxes)

        # Required Format: ['x', 'y', 'w', 'h', 'class_id'].  (x, y) is top left point\
        new_format_bboxes = [
            np.concatenate(
                [
                    bounding_box[0, :], bounding_box[2, :] - bounding_box[0, :],
                    np.array([class_id])
                ]
            )
            for bounding_box, class_id
            in zip(bounding_boxes, class_ids)
        ]

        histogram = self.generate_input_representation(events, (self.height, self.width))

        # debug TODO
        # import pickle
        # dumpfile_name = os.path.split(filename[0])[1].split('.')[0]
        # with open(f"/code/{dumpfile_name}.dump", "wb") as f:
        #     pickle.dump(
        #         {
        #             "file": dumpfile_name,
        #             "events": events,
        #             "events_full": events_scaled_full,
        #             "bounding_boxes": bounding_boxes,
        #             "events_old": events_old,
        #             "bounding_boxes_old": bb_old,
        #             "size": (self.height, self.width),
        #             "size_old": (height, width),
        #             "histogram": histogram,
        #         },
        #         f
        #     )

        nr_bbs = len(new_format_bboxes)
        new_format_bboxes = np.array(new_format_bboxes)
        # 5 == new_format_bboxes.shape[1]
        padding = np.zeros((self.max_number_bounding_boxes_per_sample - nr_bbs, 5), dtype=new_format_bboxes.dtype)
        return events, np.concatenate((new_format_bboxes, padding), axis=0), histogram, os.path.split(filename[0])[1].split('.')[0]  # TODO remove names


class KittyObjectDetectionImages(KittyObjectDetection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_converter = ToTensor()

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        class_ids = self.class_id_list_list[idx]
        bounding_boxes = self.bounding_box_list_list[idx]
        bounding_boxes = [bounding_box.reshape([4, 2]) for bounding_box in bounding_boxes]
        # Set negative corners to zero
        bounding_boxes = [np.maximum(bounding_box, np.zeros_like(bounding_box)) for bounding_box in bounding_boxes]
        filename = self.files[idx]

        # interpolate events and BBs to common "source image size"
        width, height = utils.get_image_size.get_image_size(filename[0])
        # '-1' to accurately scale the largest possible coordinate, which is size-1
        # without that, the maximum coordinate '(width - 1)' would be transformed to 'self.width - self.width/width',
        # which rounds to 'self.width' (self.width < 0.5*width) and causes an ArrayIndexOutOfBoundsError
        width_factor = (self.width - 1) / (width - 1)
        height_factor = (self.height - 1) / (height - 1)
        # bounding_boxes: [[x1, y1], [x2, y2], ...], dtype=np.int16
        bb_scale_factor = np.array((width_factor, height_factor))
        bounding_boxes = [np.around(bounding_box * bb_scale_factor).astype(np.int16) for bounding_box in bounding_boxes]

        # Required Format: ['x', 'y', 'w', 'h', 'class_id'].  (x, y) is top left point\
        new_format_bboxes = [
            np.concatenate(
                [
                    bounding_box[0, :], bounding_box[2, :] - bounding_box[0, :],
                    np.array([class_id])
                ]
            )
            for bounding_box, class_id
            in zip(bounding_boxes, class_ids)
        ]

        # load image from disk an convert to tensor
        image = self.read_image(filename[-1])
        # interpolate image to required size
        image = torch.nn.functional.interpolate(image, size=(self.height, self.width))
        # required format: [h, w, c], is [b, c, h, w]
        image = image.squeeze_(0)
        image = image.permute(1, 2, 0)

        nr_bbs = len(new_format_bboxes)
        new_format_bboxes = np.array(new_format_bboxes)
        # 5 == new_format_bboxes.shape[1]
        padding = np.zeros((self.max_number_bounding_boxes_per_sample - nr_bbs, 5), dtype=new_format_bboxes.dtype)
        assert image is not None
        return (
            torch.zeros((1, 4)), np.concatenate((new_format_bboxes, padding), axis=0), image,
            os.path.split(filename[0])[1].split('.')[0]
        )  # TODO remove names

    def read_image(self, path):
        image = Image.open(path)
        return self.image_converter(image).unsqueeze_(0)


class Prophesee(DatasetBase):

    def __init__(self, *args, **kwargs):
        """
        Creates an iterator over the Prophesee object recognition dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param event_representation: 'histogram' or 'event_queue'
        """
        super().__init__(*args, **kwargs)
        if self.mode == 'training':
            self.mode = 'train'
        elif self.mode == 'validation':
            self.mode = 'val'
        elif self.mode == 'testing':
            self.mode = 'test'

        file_dir = os.path.join('detection_dataset_duration_60s_ratio_1.0', self.mode)
        self.files = listdir(os.path.join(self.root, file_dir))
        # Remove duplicates (.npy and .dat)
        self.files = [os.path.join(file_dir, time_seq_name[:-9]) for time_seq_name in self.files
                      if time_seq_name[-3:] == 'npy']

        if self.nr_events_window == -1:
            self.nr_events_window = 250000

        self.max_nr_bbox = 15

        if self.object_classes == 'all':
            self.nr_classes = 2
            self.object_classes = ['Car', "Pedestrian"]
        else:
            self.nr_classes = len(self.object_classes)

        self.sequence_start = []
        self.createAllBBoxDataset()
        self.nr_samples = len(self.files)

        if self.shuffle:
            zipped_lists = list(zip(self.files,  self.sequence_start))
            random.seed(7)
            random.shuffle(zipped_lists)
            self.files,  self.sequence_start = zip(*zipped_lists)

    def createAllBBoxDataset(self):
        """
        Iterates over the files and stores for each unique bounding box timestep the file name and the index of the
         unique indices file.
        """
        file_name_bbox_id = []
        print('Building the Dataset')
        pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)

        for i_file, file_name in enumerate(self.files):
            bbox_file = os.path.join(self.root, file_name + '_bbox.npy')
            event_file = os.path.join(self.root, file_name + '_td.dat')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['ts'], return_index=True)

            for unique_time in unique_ts:
                sequence_start = self.searchEventSequence(event_file, unique_time, nr_window_events=250000)
                self.sequence_start.append(sequence_start)

            file_name_bbox_id += [[file_name, i] for i in range(len(unique_indices))]
            pbar.update(1)

        pbar.close()
        self.files = file_name_bbox_id

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        bbox_file = os.path.join(self.root, self.files[idx][0] + '_bbox.npy')
        event_file = os.path.join(self.root, self.files[idx][0] + '_td.dat')

        # Bounding Box
        f_bbox = open(bbox_file, "rb")
        # dat_bbox types (v_type):
        # [('ts', 'uint64'), ('x', 'float32'), ('y', 'float32'), ('w', 'float32'), ('h', 'float32'), (
        # 'class_id', 'uint8'), ('confidence', 'float32'), ('track_id', 'uint32')]
        start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        unique_ts, unique_indices = np.unique(dat_bbox['ts'], return_index=True)
        nr_unique_ts = unique_ts.shape[0]

        bbox_time_idx = self.files[idx][1]

        # Get bounding boxes at current timestep
        if bbox_time_idx == (nr_unique_ts - 1):
            end_idx = dat_bbox['ts'].shape[0]
        else:
            end_idx = unique_indices[bbox_time_idx+1]

        bboxes = dat_bbox[unique_indices[bbox_time_idx]:end_idx]

        # Required Information ['x', 'y', 'w', 'h', 'class_id']
        np_bbox = rfn.structured_to_unstructured(bboxes)[:, [1, 2, 3, 4, 5]]
        np_bbox = self.cropToFrame(np_bbox)

        const_size_bbox = np.zeros([self.max_nr_bbox, 5])
        const_size_bbox[:np_bbox.shape[0], :] = np_bbox

        # Events
        events = self.readEventFile(event_file, self.sequence_start[idx],  nr_window_events=self.nr_events_window)
        histogram = self.generate_input_representation(events, (self.height, self.width))

        return events, const_size_bbox.astype(np.int64), histogram

    def searchEventSequence(self, event_file, bbox_time, nr_window_events=250000):
        """
        Code adapted from:
        https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/io/psee_loader.py

        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_cirterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        term_criterion = nr_window_events // 2
        nr_events = dat_events_tools.count_events(event_file)
        file_handle = open(event_file, "rb")
        ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
        low = 0
        high = nr_events

        # binary search
        while high - low > term_criterion:
            middle = (low + high) // 2

            # self.seek_event(file_handle, middle)
            file_handle.seek(ev_start + middle * ev_size)
            mid = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=1)["ts"][0]

            if mid > bbox_time:
                high = middle
            elif mid < bbox_time:
                low = middle + 1
            else:
                file_handle.seek(ev_start + (middle - (term_criterion // 2) * ev_size))
                break

        file_handle.close()
        # we now know that it is between low and high
        return ev_start + low * ev_size

    def readEventFile(self, event_file, file_position, nr_window_events=250000):
        file_handle = open(event_file, "rb")
        # file_position = ev_start + low * ev_size
        file_handle.seek(file_position)
        dat_event = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=nr_window_events)
        file_handle.close()

        x = np.bitwise_and(dat_event["_"], 16383)
        y = np.right_shift(
            np.bitwise_and(dat_event["_"], 268419072), 14)
        p = np.right_shift(np.bitwise_and(dat_event["_"], 268435456), 28)
        p[p == 0] = -1
        events_np = np.stack([x, y, dat_event['ts'], p], axis=-1)

        return events_np

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        array_width = np.ones_like(np_bbox[:, 0]) * self.width - 1
        array_height = np.ones_like(np_bbox[:, 1]) * self.height - 1

        np_bbox[:, :2] = np.maximum(np_bbox[:, :2], np.zeros_like(np_bbox[:, :2]))
        np_bbox[:, 0] = np.minimum(np_bbox[:, 0], array_width)
        np_bbox[:, 1] = np.minimum(np_bbox[:, 1], array_height)

        np_bbox[:, 2] = np.minimum(np_bbox[:, 2], array_width - np_bbox[:, 0])
        np_bbox[:, 3] = np.minimum(np_bbox[:, 3], array_height - np_bbox[:, 1])

        return np_bbox


class NCars(DatasetBase):
    def __init__(self, *args, **kwargs):
        """
        Creates an iterator over the N_Caltech101 dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        :param event_representation: 'histogram' or 'event_queue'
        """
        super().__init__(*args, **kwargs)
        if self.mode == 'training':
            self.mode = 'train'
        elif self.mode == 'testing':
            self.mode = 'test'
        if self.mode == 'validation':
            self.mode = 'val'
        self.root = os.path.join(self.root, self.mode)
        self.nr_classes = len(self.object_classes)

        self.files = listdir(self.root)
        self.nr_samples = len(self.files)

        if self.shuffle:
            random.shuffle(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = np.loadtxt(os.path.join(self.root, self.files[idx], 'is_car.txt')).astype(np.int64)
        events = np.loadtxt(os.path.join(self.root, self.files[idx], 'events.txt'), dtype=np.float32)
        events[events[:, -1] == 0, -1] = -1
        nr_events = events.shape[0]

        window_start = 0
        window_end = nr_events
        if self.augmentation:
            events = random_shift_events(events, max_shift=10,resolution=(self.height, self.width))
            events = random_flip_events_along_x(events, resolution=(self.height, self.width))
            window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))

        if self.nr_events_window != -1:
            # Catch case if number of events in batch is lower than number of events in window.
            window_end = min(nr_events, window_start + self.nr_events_window)

        events = events[window_start:window_end, :]

        histogram = self.generate_input_representation(events, (self.height, self.width))

        return events, label, histogram
