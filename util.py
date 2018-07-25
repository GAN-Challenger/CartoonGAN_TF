# -*- coding: utf-8 -*-

from scipy.misc import imread
from tensorlayer.prepro import crop, imresize


def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    return imread(path + file_name, mode='RGB')


def crop_sub_imgs_fn(x, is_random=True, wrg=256, hrg=256):
    x = crop(x, wrg=wrg, hrg=hrg, is_random=is_random)
    x /= 255. / 2.
    x -= 1.
    return x


def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic')
    x /= 255. / 2.
    x -= 1.
    return x
