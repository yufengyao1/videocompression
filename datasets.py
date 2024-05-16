import os
import cv2
import torch
import imageio
import numpy as np
from random import random
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop


class VideoDatasets(data.Dataset):
    def __init__(self,  transforms=None):
        self.transform = transforms
        folder = "videos/"
        files = os.listdir(folder)
        self.files = [folder+item for item in files]
        self.mp4_file = self.files[0]
        self.index = 0
        self.reader = None

    def __getitem__(self, index):
        if self.reader is None:
            num = np.random.randint(0, len(self.files))
            self.mp4_file = self.files[num]
            self.reader = imageio.get_reader(self.mp4_file, 'ffmpeg')  # 课堂视频
        metadata = self.reader.get_meta_data()
        duration = metadata["duration"]
        fps = metadata['fps']
        self.nframes = int(duration-1)*int(fps)
        frames = []
        for i in range(16):
            frame_rgb = self.reader.get_data(i+self.index)
            frames.append(frame_rgb)
        self.index += np.random.randint(200, 300)
        if self.index > self.nframes-200:
            self.index = 0
            self.reader = None

        frames = np.array(frames, dtype=np.float32)
        frames /= 255
        frames = frames.transpose((3, 0, 2, 1))
        return torch.from_numpy(frames)

    def __len__(self):
        return 100


if __name__ == '__main__':
    mp4_file = "/Users/lingoace/Desktop/mp4/tea_test.mp4"
    reader = imageio.get_reader(mp4_file, 'ffmpeg')  # 课堂视频
    metadata = reader.get_meta_data()
    duration = metadata["duration"]
    fps = metadata['fps']
    nframes = int(duration-1)*int(fps)
    print(nframes)

    frame_rgb = reader.get_data(1000)
    cv2.imshow('test', frame_rgb)
    cv2.imwrite('tmp.jpg', frame_rgb)
