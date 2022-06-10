
import esim_py
import matplotlib.pyplot as plt
import os
from typing import List, Union

from image_event_dataloader.rpg_vid2e.esim_py.tests.plot_virtual_events import viz_events


# NCaltech101 ObjectDet Format:
#
# 'Each Example is a seperate binary file consisting of a list of events. Each event occupies 40bits as described below:
# - bit39-32: Xaddress (in pixels)
# - bit31-24: Yaddress (in pixels)
# - bit23: Polarity (0 for OFF, 1 for ON)
# - bit22-0: Timestamp (in microseconds)
#
# raw_data = np.fromfile(f, dtype=np.uint8)
# raw_data = np.uint32(raw_data)
# all_y = raw_data[1::5]
# all_x = raw_data[0::5]
# all_p = (raw_data[2::5] & 128) >> 7  # bit 7
# all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
#
# -> flat array [eventevent2event3...eventn]
# event: x_loc(8bit).y_loc(8bit).value(1bit).time(23bit)
# saved as 4 consecutive np.uint8:
#
#
# vid2e Format:
#
# dtype: float64
# axis 0: event_num
# axis 1: x, y, t, p
# time (t): seconds
# polarity (p): -1 for negative, 1 for positive  # TODO ON/OFF?
# x, y: position in pixels -> values greater than 511 -> not representable as np.uint8
# -> save as-is and convert in dataset loader

#
# def viz_events(events, resolution):
#     pos_events = events[events[:,-1]==1]
#     neg_events = events[events[:,-1]==-1]
#
#     image_pos = np.zeros(resolution[0]*resolution[1], dtype="uint8")
#     image_neg = np.zeros(resolution[0]*resolution[1], dtype="uint8")
#
#     _ = pos_events[:, 0]
#     _ = pos_events[:, 1]
#     _ = resolution[1]
#     _ = pos_events[:, -1]
#     np.add.at(image_pos, (pos_events[:,0]+pos_events[:,1]*resolution[1]).astype("int32"), pos_events[:,-1]**2)
#     np.add.at(image_neg, (neg_events[:,0]+neg_events[:,1]*resolution[1]).astype("int32"), neg_events[:,-1]**2)
#
#     image_rgb = np.stack(
#         [
#             image_pos.reshape(resolution),
#             image_neg.reshape(resolution),
#             np.zeros(resolution, dtype="uint8")
#         ], -1
#     ) * 50
#
#     return image_rgb


num_events_plot = 30000000

Cp, Cn = 0.1, 0.1
refractory_period = 1e-4
log_eps = 1e-3
use_log = True
H, W = 375, 1242


def plot_events(events):
    image_rgb = viz_events(events[:num_events_plot], [H, W])
    ax = plt.gca()
    ax.imshow(image_rgb)
    ax.axis('off')
    ax.set_title("Cp=%s Cn=%s" % (Cp, Cn))
    plt.show()


class ImageSequenceToEventConverter:

    def __init__(self, base_folder: str, on_demand_mode: bool, n_prec_imgs: int = 3, framerate: int = 24):

        self.base_folder = base_folder
        self.image_folder = os.path.join(self.base_folder, "image_2/")
        self.image_prev_folder = os.path.join(self.base_folder, "prev_2/")
        self.on_demand_mode = on_demand_mode
        if not self.on_demand_mode:
            self.event_output_folder = os.path.join(self.base_folder, "tmp/events_2/")
            if not os.path.isdir(self.event_output_folder):
                os.makedirs(self.event_output_folder)

        # timestamps_file = os.path.join(os.path.dirname(__file__), "data/images/timestamps.txt")

        self.esim = esim_py.EventSimulator(Cp,
                                           Cn,
                                           refractory_period,
                                           log_eps,
                                           use_log)

        self.esim.setParameters(Cp, Cn, refractory_period, log_eps, use_log)

        self.n_prec_imgs = n_prec_imgs
        self.framerate = framerate

        self.timestamps = [i/framerate for i in range(n_prec_imgs + 1)]

    def convert_one_sequence(self, image_list: List[Union[str, os.PathLike]]):
        return self.esim.generateFromStampedImageSequence(image_list, self.timestamps)

    def list_available_sequences(self) -> List[List[Union[str, os.PathLike]]]:
        scenes = os.listdir(self.image_folder)
        scenes = [scene for scene in scenes if scene.endswith(".png")]

        sequences = []

        for scene in scenes:
            img_name = scene.split('.')[0]
            # print(f"processing scene {img_name}...")
            image_list = []
            for i in range(self.n_prec_imgs, 0, -1):
                image_list.append(
                    os.path.join(
                        self.image_prev_folder,
                        f"{img_name}_{i:02d}.png"
                    )
                )
            image_list.append(
                os.path.join(
                    self.image_folder,
                    scene
                )
            )
            valid = True
            for image_path in image_list:
                if not os.path.isfile(image_path):
                    print(f"missing image: {image_path}.")
                    valid = False
                    break
            if valid:
                sequences.append(image_list)

        return sequences

    def convert_all(self):
        if self.on_demand_mode:
            raise NotImplementedError("Method 'convert_all()' not available in on_demand mode.")
        sequences = self.list_available_sequences()
        for sequence in sequences:
            events = self.convert_one_sequence(sequence)
            output_file = os.path.join(
                self.event_output_folder,
                f"{os.path.split(sequence[-1])}.bin"
            )
            # save to disk
            with open(output_file, "wb") as f:
                events.tofile(f)


if __name__ == "__main__":

    converter = ImageSequenceToEventConverter(
        base_folder="/media/user/Volume/HiWi_IGD/KITTI_dataset/training/",
        on_demand_mode=False
    )
    converter.convert_all()
