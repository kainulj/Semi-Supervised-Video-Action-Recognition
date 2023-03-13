import random
from PIL import Image, ImageOps
import numpy as np
import torchvision

class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        h, w = frames[0].size

        top = np.random.randint(0, h - self.size)
        left = np.random.randint(0, w - self.size)

        out = [frame.crop((left, top, left + self.size, top + self.size)) for frame in frames]

        return out

class CenterCrop(object):

    def __init__(self, size):
        self.crop = torchvision.transforms.CenterCrop(size)

    def __call__(self, frames):

        out = [self.crop(frame) for frame in frames]

        return out

class RandomFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, frames):
        if random.random() < self.p:
            out = [frame.transpose(Image.FLIP_LEFT_RIGHT) for frame in frames]
            return out
        else:
            return frames

class RandomScale(object):
    def __init__(self, min_scale, max_scale):
        self.min = min_scale
        self.max = max_scale

    def __call__(self, frames):
        scale = random.uniform(self.min, self.max)
        h, w = frames[0].size
        new_h, new_w = int(h * scale), int(w * scale)

        out = [frame.resize((new_w, new_h)) for frame in frames]

        return out

class ToTensor(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, frames):

        out = [self.to_tensor(frame) for frame in frames]
        return out

class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, frames):

        out = [self.normalize(frame) for frame in frames]
        return out

