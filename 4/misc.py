import numpy as np
import cv2
import os
import time

from composer import Composer


def memoize(func: callable) -> callable:
    cache = {}

    def memoized(*args):
        if args not in cache:
            cache[args] = func(*args)

        return cache[args]

    return memoized


def preprocess(path_to_folder: str = '4/data_hometask_4/',
               alpha: float = 1,
               beta: float = 1
               ) -> tuple[list[np.ndarray], list[np.ndarray]]:

    imgs_links = []
    masks_links = []
    for f in os.listdir(path_to_folder):
        f = path_to_folder + f
        if 'image' in f:
            imgs_links.append(f)
        elif 'mask' in f:
            masks_links.append(f)
    imgs_links.sort()
    masks_links.sort()
    imgs = [cv2.imread(f) for f in imgs_links]
    masks = [
        cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(bool) for f in masks_links
    ]

    return imgs, masks


def save_img(filename: str, img: np.ndarray):
    cv2.imwrite(filename, img)

