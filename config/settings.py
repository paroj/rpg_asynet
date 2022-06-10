import os
import time
import yaml
import torch
import shutil
from typing import Union


class Settings:
    def __init__(self, settings_yaml, generate_log=True, overwrite_log_dir: Union[str, os.PathLike] = None):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- hardware ---
            hardware = settings['hardware']
            gpu_device = hardware['gpu_device']

            self.gpu_device = torch.device("cpu") if gpu_device == "cpu" else torch.device("cuda:" + str(gpu_device))

            self.num_cpu_workers = hardware['num_cpu_workers']
            if self.num_cpu_workers < 0:
                self.num_cpu_workers = os.cpu_count()

            # --- Model ---
            model = settings['model']
            self.model_name = model['model_name']

            # --- dataset ---
            dataset = settings['dataset']
            self.dataset_name = dataset['name']
            self.event_representation = dataset['event_representation']
            self.event_window_mode = dataset['event_window_mode']
            dataset_specs = self.load_dataset(dataset, self.dataset_name)

            self.dataset_path = dataset_specs['dataset_path']
            assert os.path.isdir(self.dataset_path), f"dataset not found at '{self.dataset_path}"
            self.object_classes = dataset_specs['object_classes']
            self.height = dataset_specs['height']
            self.width = dataset_specs['width']
            self.nr_events_window = dataset_specs['nr_events_window']
            self.timeframe_events_window = dataset_specs['timeframe_events_window']

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.discard_suboptimal_checkpoints = checkpoint['discard_suboptimal_checkpoints']
            assert isinstance(self.discard_suboptimal_checkpoints, bool)
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']
            self.use_pretrained = checkpoint['use_pretrained']
            self.pretrained_dense_vgg = checkpoint['pretrained_dense_vgg']
            self.pretrained_sparse_vgg = checkpoint['pretrained_sparse_vgg']
            self.pretrained_sparse_yolo = checkpoint['pretrained_sparse_yolo']
            self.pretrained_dense_yolo = checkpoint['pretrained_dense_yolo']

            # --- directories ---
            directories = settings['dir']
            log_dir = directories['log']

            # --- logs ---
            if generate_log:
                if overwrite_log_dir is None:
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    log_dir = os.path.join(log_dir, timestr)
                    os.makedirs(log_dir)
                else:
                    assert os.path.isdir(overwrite_log_dir)
                    log_dir = overwrite_log_dir
                settings_copy_filepath = os.path.join(log_dir, 'settings_default.yaml')
                shutil.copyfile(settings_yaml, settings_copy_filepath)
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                if not os.path.isdir(self.ckpt_dir):
                    os.mkdir(self.ckpt_dir)
                self.vis_dir = os.path.join(log_dir, 'visualization')
                if not os.path.isdir(self.vis_dir):
                    os.mkdir(self.vis_dir)
            else:
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                self.vis_dir = os.path.join(log_dir, 'visualization')

            # --- optimization ---
            optimization = settings['optim']
            self.max_epochs = optimization['max_epochs']
            self.batch_size = optimization['batch_size']
            self.init_lr = float(optimization['init_lr'])
            self.steps_lr = optimization['steps_lr']
            self.factor_lr = float(optimization['factor_lr'])

    def load_dataset(self, dataset: dict, dataset_name: str) -> dict:
        if dataset_name == 'NCaltech101':
            dataset_specs = dataset['ncaltech101']
        elif dataset_name == 'NCaltech101_ObjectDetection':
            dataset_specs = dataset['ncaltech101_objectdetection']
        elif dataset_name == 'Kitty_ObjectDetection' or self.dataset_name == 'Kitty_ObjectDetection_Image':
            dataset_specs = dataset['kitty_objectdetection']
        elif dataset_name == 'Prophesee':
            dataset_specs = dataset['prophesee']
        elif dataset_name == 'NCars':
            dataset_specs = dataset['ncars']
        else:
            raise ValueError(f"Unknown Dataset specified in settings_default.yaml: '{self.dataset_name}'.")
        return dataset_specs
