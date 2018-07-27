# -*- coding: utf-8 -*-

from scipy.misc import imread
# from tensorlayer.prepro import crop, imresize
import cv2
import os


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


def resize_fn(x, scaled_size=256):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    scale_ratio = max(scaled_size / x.shape[0], scaled_size / x.shape[1]) + 0.05

    height = int(x.shape[0] * scale_ratio)
    width = int(x.shape[1] * scale_ratio)

    x = imresize(x, size=[width, height], interp='bicubic')
    return x


def smooth(input_dir, output_dir):
    def gaussian_blur(file_name):
        img = cv2.imread(os.path.join(input_dir, file_name))
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite(os.path.join(output_dir, file_name), blur)
        return blur

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for f_name in os.listdir(input_dir):
        gaussian_blur(f_name)


def resize(input_dir, output_dir, scaled_size=256):
    def img_resize(file_name):
        img = cv2.imread(os.path.join(input_dir, file_name))
        scale_ratio = max(scaled_size / img.shape[0], scaled_size / img.shape[1]) + 0.1

        height = int(img.shape[0] * scale_ratio)
        width = int(img.shape[1] * scale_ratio)

        scaled = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output_dir, file_name), scaled)
        return scaled

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for f_name in os.listdir(input_dir):
        img_resize(f_name)


if __name__ == '__main__':
    smooth('D:/Video frame/cartoon_gan/your_name_resize', 'D:/Video frame/cartoon_gan/your_name_resize_edge')
    # resize('D:/Video frame/cartoon_gan/your_name', 'D:/Video frame/cartoon_gan/your_name_resize')
    # resize('D:/Video frame/cartoon_gan/kingsman', 'D:/Video frame/cartoon_gan/kingsman_resize')





