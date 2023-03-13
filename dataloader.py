import os 
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import ( Normalize )
from videotransforms import ( CenterCrop, Normalize, RandomCrop, RandomFlip, ToTensor, RandomScale )
from PIL import Image

def sample_frames(frames, samples, tube_length):
    tube_inds = np.arange(1, frames, tube_length)
    if len(tube_inds) < samples:
        sample_inds = np.random.choice(tube_inds, samples, replace=True)
    else:
        sample_inds = np.random.choice(tube_inds, samples, replace=False)
    return np.sort(sample_inds)



def augmentator(is_train):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    augments = []

    if is_train:
        augments += [
            RandomFlip(0.5),
            RandomScale(0.95, 1.33),
            RandomCrop(224),
        ]
    else:
        augments += [ CenterCrop(224) ]

    augments += [
        ToTensor(),
        Normalize(mean, std)
    ]
    augmentor = transforms.Compose(augments)
    return augmentor

    


class VideoRecord(object):
    def __init__(self, path, start_frame, end_frame, label):
        self.path = path
        self.video_id = os.path.basename(path)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.label = label

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1


class VideoDataSet(data.Dataset):

    def __init__(self, root_path, list_file, num_samples=16, tube_length=2,
                 image_tmpl='{:05d}.jpg', is_train=True, test_mode=False, seperator=' ',
                 num_classes=None):

        self.root_path = root_path
        self.list_file = list_file
        self.num_samples = num_samples
        self.num_frames = num_samples
        self.tube_length = tube_length
        self.image_tmpl = image_tmpl
        self.is_train = is_train
        self.test_mode = test_mode
        self.seperator = seperator
        self.video_list = self.create_video_list()
        self.num_classes = num_classes
        self.transforms = augmentator(is_train)


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

    def create_video_list(self):
        file_list = []
        for line in open(self.list_file):
            elements = line.strip().split(self.seperator)
            file_list.append([elements[0], int(elements[1]), int(elements[2]), int(elements[3])])

        video_list = [VideoRecord(item[0], item[1], item[2], item[3]) for item in file_list]

        return video_list

    def __getitem__(self, index):
        record = self.video_list[index]
        sampled_frames = sample_frames(record.num_frames, self.num_samples, self.tube_length)
        images = []
        for frame in sampled_frames:
            for i in range(self.tube_length):
                img = self._load_image(record.path, frame + i)
                images.append(img)

        images = self.transforms(images)
        images = torch.stack(images)
        images = torch.transpose(images, 0, 1)
        return images, record.label

    def __len__(self):
        return len(self.video_list)