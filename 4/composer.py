import numpy as np
import cv2
from tqdm import tqdm, trange
import threading
import os

class Chain:
    def __init__(self,
                 img_chains: list[np.ndarray],
                 msk_chains: list[np.ndarray],
                 alpha: float = 1,
                 beta: float = 1
                 ) -> None:

        self.alpha = alpha
        self.beta = beta
        self.img_chains = img_chains
        self.msk_chains = msk_chains

    def point_weight(self, idx: int, k: int) -> float:
        raise NotImplementedError

    def edge_weight(self, idx1: int, k1: int, idx2: int, k2: int) -> float:
        raise NotImplementedError

    def compose(self) -> np.ndarray:
        raise NotImplementedError

# TODO add row looping and saving
# TODO add tests
# TODO add threading
# TODO add mask and photos reading


class Composer:
    def __init__(self,
                 images: list[np.ndarray],
                 masks: list[np.ndarray],
                 alpha: float = 1,
                 beta: float = 1
                 ) -> None:

        self.images = np.vstack(images),
        self.masks = np.vstack([m.astype(bool) for m in masks])
        self.alpha = alpha
        self.beta = beta

    def compose(self) -> np.ndarray:
        res = []
        for i in trange(self.images[0].shape[0]):
            res.append(
                Chain(self.images[:, i], self.masks[:, i])
            )
        return np.array(res)


def test(path_to_folder, alpha=1, beta=1):
    imgs = []
    masks = []
    for f in os.listdir(path_to_folder):
        f = path_to_folder + f
        if 'image' in f:
            imgs.append(f)
        elif 'mask' in f:
            masks.append(f)
    imgs.sort()
    masks.sort()
    imgs = [cv2.imread(f) for f in imgs]
    masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(bool) for f in masks]

    return imgs, masks


if __name__=='__main__':
    test('4/data_hometask_4/')
