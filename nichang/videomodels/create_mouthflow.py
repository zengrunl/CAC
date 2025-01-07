import cv2
import random
import os
import numpy as np
from tqdm import tqdm

__all__ = ["Compose", "Normalize", "CenterCrop", "RgbToGray", "RandomCrop", "HorizontalFlip"]


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.0)
        delta_h = int(round((h - th)) / 2.0)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally."""

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames
def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([#RgbToGray(),  #源代码此处没有RgbToGray（）
                                Normalize( 0.0,255.0 ),
                                RandomCrop(crop_size),
                                HorizontalFlip(0.5),
                                Normalize(mean, std) ])
    preprocessing['val'] = Compose([#RgbToGray(),
                                Normalize( 0.0,255.0 ),
                                CenterCrop(crop_size),
                                Normalize(mean, std) ])
    preprocessing["test"] = preprocessing["val"]
    return preprocessing
src_dir = "/root/data1/LSR/LSR2/lrs2_rebuild/mouths/"
dst_dir = "/root/data1/LSR/LSR2/lrs2_rebuild/mouth_flow/"
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
os.makedirs(dst_dir, exist_ok=True)
video_files = [f for f in os.listdir(src_dir) if f.endswith('.npz')]
lipreading_preprocessing_func = get_preprocessing_pipelines()['train']
for video_file in tqdm(video_files, desc="Calculating optical flows"):
    video_path = os.path.join(src_dir, video_file)
    flow_path = os.path.join(dst_dir, video_file.replace('.npz', '.npy'))
    source_mouth = lipreading_preprocessing_func(np.load(video_path)["data"])
    T, H, W = source_mouth.shape
    flow = np.zeros(( T - 1, H, W, 2)) #2,2,49,88,88,2
     # 遍历每个样本，每个通道，每个帧对，计算光流
    for t in range(T - 1):
                 # a = mouth[i, j, t]
                 # print("mouth[i, j, t] shape: ", mouth[i, j, t].shape)
                 # print("mouth[i, j, t] dtype: ", mouth[i, j, t].dtype)
                 # 计算帧 t 和帧 t+1 之间的光流
        flow[t] = cv2.calcOpticalFlowFarneback(source_mouth[t], source_mouth[t + 1], None, 0.5, 3, 15,
                                                              3, 5, 1.2, 0)
    np.save(flow_path, flow)