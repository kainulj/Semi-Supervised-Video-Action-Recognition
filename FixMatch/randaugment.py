# code in this file is adpated from
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from torchvision.transforms import ToTensor

PARAMETER_MAX = 10


def AutoContrast(images, **kwarg):
    images = [PIL.ImageOps.autocontrast(img) for img in images]
    return images


def Brightness(images, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    images = [PIL.ImageEnhance.Brightness(img).enhance(v) for img in images]
    return images


def Color(images, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    images = [PIL.ImageEnhance.Color(img).enhance(v) for img in images]
    return images


def Contrast(images, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    images = [PIL.ImageEnhance.Contrast(img).enhance(v) for img in images]
    return images


def Cutout(images, v, max_v, bias=0):
    if v == 0:
        return images
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(images[0].size))
    return CutoutAbs(images, v)


def CutoutAbs(images, v, **kwarg):
    w, h = images[0].size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    for i in range(len(images)):
      img = images[i].copy()
      PIL.ImageDraw.Draw(img).rectangle(xy, color)
      images[i] = img
    return images


def Equalize(images, **kwarg):
    images = [PIL.ImageOps.equalize(img) for img in images]
    return images


def Identity(images, **kwarg):
    return images


def Invert(images, **kwarg):
    images = [PIL.ImageOps.invert(img) for img in images]
    return images


def Posterize(images, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    images = [PIL.ImageOps.posterize(img, v) for img in images]
    return images


def Rotate(images, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    images = [img.rotate(v) for img in images]
    return images


def Sharpness(images, v, max_v, bias=0):
    images = [PIL.ImageEnhance.Sharpness(img).enhance(v) for img in images]
    return images


def ShearX(images, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    images = [img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)) for img in images]
    return images


def ShearY(images, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    images = [img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)) for img in images]
    return images


def Solarize(images, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    images = [PIL.ImageOps.solarize(img, 256 - v) for img in images]
    return images


def SolarizeAdd(images, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    for i, img in enumerate(images):
      img_np = np.array(img).astype(np.int)
      img_np = img_np + v
      img_np = np.clip(img_np, 0, 255)
      img_np = img_np.astype(np.uint8)
      images[i] = Image.fromarray(img_np)
    images = [PIL.ImageOps.solarize(img, threshold) for img in images]
    return images


def TranslateX(images, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * images[0].size[0])
    images = [img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)) for img in images]
    return images


def TranslateY(images, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * images[0].size[1])
    images = [img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)) for img in images]
    return images


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs

class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()
        self.toTensor = ToTensor()

    def __call__(self, images):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            images = op(images, v=v, max_v=max_v, bias=bias)
        images = CutoutAbs(images, int(224*0.5))
        images = [self.toTensor(img) for img in images ]
        return images