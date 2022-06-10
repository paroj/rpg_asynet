"""
Example usage: CUDA_VISIBLE_DEVICES=1, python train_yolo.py --settings_file "config/settings_default.yaml"
"""
import argparse
import os.path
import traceback

from config.settings import Settings
from training.trainer import FBSparseVGGModel
from training.trainer import DenseVGGModel
from training.trainer import SparseObjectDetModel
from training.trainer import YoloSparseObjectDetModel
from training.trainer import DenseObjectDetModel
from training.trainer import YoloDenseObjectDetModel


def main(settings_file: str = None, log_dir: str = None, resume_training: bool = None, resume_ckpt_file: str = None):
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml')  # , required=True)
    parser.add_argument('--log_dir', help='Path to log-dir for this specific training run')
    parser.add_argument('--resume_training', help='overwrite resume training flag of settings file')
    parser.add_argument('--resume_ckpt_file', help='overwrite path to checkpoint to resume training')

    # not used, but required for execution in pycharm console
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)

    args = parser.parse_args()

    if settings_file is not None:
        settings_filepath = settings_file
    elif args.settings_file is not None:
        settings_filepath = args.settings_file
    else:
        settings_filepath = "config/settings.yaml"

    if log_dir is not None:
        assert os.path.isdir(log_dir)
        settings = Settings(settings_filepath, generate_log=True, overwrite_log_dir=log_dir)
    elif args.log_dir is not None:
        assert os.path.isdir(args.log_dir)
        settings = Settings(settings_filepath, generate_log=True, overwrite_log_dir=args.log_dir)
    else:
        settings = Settings(settings_filepath, generate_log=True)

    if resume_training is not None:
        settings.resume_training = bool(resume_training)
    elif args.resume_training is not None:
        settings.resume_training = bool(args.resume_training)

    if resume_ckpt_file is not None:
        assert os.path.isfile(resume_ckpt_file)
        settings.resume_ckpt_file = resume_ckpt_file
    elif args.resume_ckpt_file is not None:
        assert os.path.isfile(args.resume_ckpt_file)
        settings.resume_ckpt_file = args.resume_ckpt_file

    if settings.model_name == 'fb_sparse_vgg':
        trainer = FBSparseVGGModel(settings)
    elif settings.model_name == 'dense_vgg':
        trainer = DenseVGGModel(settings)
    elif settings.model_name == 'fb_sparse_object_det':
        trainer = SparseObjectDetModel(settings)
    elif settings.model_name == 'dense_object_det':
        trainer = DenseObjectDetModel(settings)
    elif settings.model_name == 'yolo_fb_sparse':
        trainer = YoloSparseObjectDetModel(settings)
    elif settings.model_name == 'dense_yolo':
        trainer = YoloDenseObjectDetModel(settings)
    else:
        raise ValueError('Model name %s specified in the settings file is not implemented' % settings.model_name)

    trainer.train()
    # trainer.saveCheckpoint()  # TODO


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
