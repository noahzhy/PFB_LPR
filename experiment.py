import os
import glob
import random

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T


def find_divider(img):
    # convert PIL image to numpy array
    img = np.array(img)
    h, w = img.shape
    start = h // 4
    end = start + h // 2
    # crop image
    img = img[start:end, :]
    # get std of each row
    std = np.std(img, axis=1)
    # get index of min std
    index = np.argmin(std)
    return index + start


def rebuild_image(img, index):
    w, h = img.size
    curr_w = int(w*.6)
    top_start = int(w*.2)
    top_end = int(w*.8)
    top = img.crop((top_start, 0, top_end, index)).resize((curr_w, h), Image.BILINEAR)
    bottom = img.crop((0, index, w, h)).resize((w, h), Image.BILINEAR)
    # merge two images via horizontal stack
    img = np.hstack((top, bottom))
    return img


if __name__ == '__main__':
    target_size = (64, 192)
    h, w = target_size
    imgs = glob.glob(os.path.join('data/train', '*.jpg'))
    # random select 10 images and show them in one figure
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    random_imgs = random.sample(imgs, 10)
    for i in range(10):
        ax = axes[i // 5, i % 5]
        img = Image.open(random_imgs[i]).convert('L').resize((w, h))
        ans = find_divider(img)
        img = rebuild_image(img, ans)
        # save image named as '000.jpg', '001.jpg', ...
        Image.fromarray(img).save(os.path.join('data/test', '{:03d}.jpg'.format(i)))
        # draw = ImageDraw.Draw(img)
        # draw.line((0, ans, w, ans), fill=255, width=1)
        ax.imshow(img, cmap='gray')
    plt.show()
