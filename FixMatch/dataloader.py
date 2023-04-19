# code in this file is adapted from
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py
# https://github.com/IBM/action-recognition-pytorch/blob/master/utils/video_dataset.py

import os 
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import ( ToTensor, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, Normalize, Compose, ToPILImage )
from PIL import Image
import random
from randaugment import RandAugmentMC

def sample_frames(frames, samples, tube_length):
    if frames % 2 == 1:
        frames -= 1
    highest_idx = frames - tube_length * samples
    if highest_idx <= 1:
        random_offset = 1
    else:
        random_offset = int(np.random.randint(1, highest_idx, 1))
    frame_idx = [int(random_offset + i * tube_length) % frames for i in range(samples)]
    return frame_idx

def labeled_augmentator(is_train):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    augments = []

    if is_train:
        scale = random.uniform(0.95, 1.3)
        new_size = int(240 * scale)
        augments += [
            RandomHorizontalFlip(0.5),
            RandomCrop(224, padding=int(224*0.125), padding_mode='reflect'),
            Normalize(mean, std)
        ]
    else:
        augments += [
            CenterCrop(224),
            Normalize(mean, std)
        ]
    augmentor = Compose(augments)
    return augmentor


class TransformFixMatch(object):
    def __init__(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.weak = Compose([
            RandomHorizontalFlip(),
            RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect')])

        self.toPILImage = ToPILImage()
        self.strong = Compose([
            RandAugmentMC(n=2, m=10)
            ])
        self.normalize = Compose([
            Normalize(mean, std)])

    def __call__(self, x):
        x = self.weak(x)
        strong = [self.toPILImage(img) for img in list(x)]
        strong = torch.stack(self.strong(strong))
        return self.normalize(x), self.normalize(strong)

def split_l_u(list_file, num_labeled, num_classes, seperator=' '):
        file_list = []
        labels = []
        for line in open(list_file):
            elements = line.strip().split(seperator)
            file_list.append([elements[0], int(elements[1]), int(elements[2]), int(elements[3])])
            labels.append(int(elements[3]))

        video_list = [FixMatchRecord(item[0], item[1], item[2], item[3]) for item in file_list]

        label_per_class = num_labeled // num_classes
        labels = np.array(labels)
        labeled_idx = []
        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            if(len(idx) < label_per_class):
                idx = np.random.choice(idx, label_per_class, True)
            else:
                idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)

        unlabeled_idx = np.delete(np.arange(len(labels)), labeled_idx)

        unlabeled_videos = [video_list[i] for i in unlabeled_idx]
        labeled_videos = [video_list[i] for i in labeled_idx]

        return labeled_videos, unlabeled_videos

def create_video_list(list_file, seperator=' '):
        file_list = []
        for line in open(list_file):
            elements = line.strip().split(seperator)
            file_list.append([elements[0], int(elements[1]), int(elements[2]), int(elements[3])])

        video_list = [FixMatchRecord(item[0], item[1], item[2], item[3]) for item in file_list]

        return video_list

class FixMatchRecord(object):
    def __init__(self, path, start_frame, end_frame, label):
        self.path = path
        self.video_id = os.path.basename(path)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.label = label

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1


class FixMatchDataSet(data.Dataset):

    def __init__(self, root_path, video_list, labeled, num_samples=16, tube_length=2,
                 image_tmpl='{:05d}.jpg', is_train=True, test_mode=False,
                 num_classes=None):

        self.root_path = root_path
        self.video_list = video_list
        self.num_samples = num_samples
        self.labeled = labeled
        self.num_frames = num_samples
        self.tube_length = tube_length
        self.image_tmpl = image_tmpl
        self.is_train = is_train
        self.test_mode = test_mode
        self.num_classes = num_classes
        self.to_tensor = ToTensor()



    def _load_image(self, directory, idx):

        def _safe_load_image(img_path):
            img_tmp = Image.open(img_path)
            img = img_tmp.copy()
            img_tmp.close()
            return img

        num_try = 0
        image_path_file = os.path.join(self.root_path, directory, self.image_tmpl.format(idx))
        img = None
        while num_try < 10:
            try:
                img = _safe_load_image(image_path_file)
                num_try = 10
            except Exception as e:
                print('[Will try load again] error loading image: {}, error: {}'.format(image_path_file, str(e)))
                num_try += 1

        if img is None:
            raise ValueError('[Fail 10 times] error loading image: {}'.format(image_path_file))

        return img

    def __getitem__(self, index):
        record = self.video_list[index]
        sampled_frames = sample_frames(record.num_frames, self.num_samples, self.tube_length)
        images = []
        for frame in sampled_frames:
            for i in range(self.tube_length):
                img = self.to_tensor(self._load_image(record.path, frame + i))

                images.append(img)

        if self.labeled:
            transforms = labeled_augmentator(self.is_train)
            images = torch.stack(images)
            images = transforms(images)
            images = torch.transpose(images, 0, 1)
            return images, record.label
        else:
            transforms = TransformFixMatch()
            images = torch.stack(images)
            weak, strong = transforms(images)
            weak, strong = torch.transpose(weak, 0, 1), torch.transpose(strong, 0, 1)
            return weak, strong


    def __len__(self):
        return len(self.video_list)
