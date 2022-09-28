import glob
import os
import random

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


# letterbox resize image via PIL
def letterbox_resize(img, size, fill_color=(0, 0, 0), left_top=False):
    img_w, img_h = img.size
    w, h = size
    scale = min(w / img_w, h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    new_img = Image.new('RGB', size, fill_color)
    if left_top:
        start_x, start_y = 0, 0
    else:
        start_x, start_y = (w - new_w) // 2, (h - new_h) // 2
    new_img.paste(img, (start_x, start_y))
    return new_img


# crop PIL image
# param p: probability of crop
# param ratio: crop ratio in [0, 1] format
class RandomCrop(object):
    def __init__(self, p, ratio=0.8):
        self.p = p
        self.ratio = ratio

    def __call__(self, img):
        if random.random() < self.p:
            w, h = img.size
            ratio = self.ratio
            crop_w, crop_h = int(w * ratio), int(h * ratio)
            start_x, start_y = random.randint(0, w - crop_w), random.randint(0, h - crop_h)
            img = img.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
        return img


# a transform to make blur or noise
class RandomFilter(object):
    def __call__(self, img):
        return self.random_filter(img)

    @staticmethod
    def random_filter(img, filter_max=3):
        filters = [
            ImageFilter.BLUR,
            ImageFilter.SMOOTH,
            ImageFilter.SMOOTH_MORE,
        ]
        assert filter_max <= len(filters)
        selection = random.sample(filters, random.randint(1, filter_max))
        for i in selection:
            img = img.filter(i)
        return img





# class: cut the image into 2 parts in height direction averagely
# and then merge them in width direction
class CutMerge(object):
    def __call__(self, img):
        w, h = img.size
        original_img = img.copy()
        img1 = img.crop((0, 0, w, h//2))
        img2 = img.crop((0, h//2, w, h))
        img = Image.new('RGB', (w*2, h))
        img.paste(img1, (0, 0))
        img.paste(original_img, (w, 0))
        return img


class LPRDataset(Dataset):
    def __init__(self, root, target_size=(32, 96)):
        self.root = root
        self.img_paths = glob.glob(os.path.join(self.root, '*.jpg'))
        self.img_paths = sorted(self.img_paths)
        self.target_size = target_size
        h, w = self.target_size
        self.base = T.Compose([
            # CutMerge(),
            T.Lambda(lambda x: letterbox_resize(x, (w, h))),
            T.ToTensor(),
        ])
        self.transform = T.Compose([
            RandomCrop(p=1),
            T.Lambda(lambda x: letterbox_resize(x, (w, h))),
            RandomFilter(),
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        h, w = self.target_size
        # two images are returned, one is original image, the other is blurred image
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('L')
        original_img = self.base(img.copy())
        transform_img = self.transform(img)

        # random swap two images
        if random.random() > 0.5:
            original_img, transform_img = transform_img, original_img

        # generate a 0 tensor for first image is lost
        if random.random() < 0.1:
            original_img = torch.zeros((1, h, w), dtype=torch.float32)

        return original_img, transform_img

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    target_size = (64, 192)
    dataset = LPRDataset(root='data/train', target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        original_img, transform_img = data
        original_img = original_img.squeeze(0).numpy()
        transform_img = transform_img.squeeze(0).numpy()
        plt.subplot(1, 2, 1)
        plt.title(label='previous')
        plt.imshow(original_img[0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title(label='current')
        plt.imshow(transform_img[0], cmap='gray')
        plt.show()
        break
