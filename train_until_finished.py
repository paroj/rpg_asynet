import subprocess
import time
import os
import train
"""
https://stackoverflow.com/a/44112591
"""


filename = '/code/train.py'
# create logdir
timestr = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("log1/", timestr)
os.makedirs(log_dir)
# log_dir = 'log1/sparse_yolo_hist_8ms_pretrained_2'  # overwrite to continue old run
checkpoint_dir = os.path.join(log_dir, "checkpoints")
while True:
    # find checkpoint file form last run
    try:
        checkpoints = os.listdir(checkpoint_dir)
    except FileNotFoundError:
        checkpoints = []
    try:
        most_recent_checkpoint = next([file for file in checkpoints if file.endswith('.pth')].__reversed__())
        # resume_training = f" --resume_training True --resume_ckpt_file /code/{checkpoint_dir}/{most_recent_checkpoint}"
        print(f"found checkpoint: {most_recent_checkpoint}")
        print("RESUMING...")
    except StopIteration:
        most_recent_checkpoint = None
        print("BEGINNING...")
        # resume_training = ''

    # """However, you should be careful with the '.wait()'"""
    # p = subprocess.Popen(f"python -u {filename} --log_dir {log_dir}{resume_training}", shell=True).wait()
    #
    # """#if your there is an error from running 'my_python_code_A.py',
    # the while loop will be repeated,
    # otherwise the program will break from the loop"""
    # if p != 0:
    #     continue
    # else:
    #     print("DONE.")
    #     break

    try:
        if most_recent_checkpoint is not None:
            train.main(
                log_dir=log_dir,
                resume_training=True,
                resume_ckpt_file=f"/code/{checkpoint_dir}/{most_recent_checkpoint}"
            )
        else:
            train.main(
                log_dir=log_dir
            )
        print("DONE.")
        break
    except RuntimeError:
        pass
