import numpy as np


class Composer:
    def __init__(self,
                 images: list[np.ndarray],
                 masks: list[np.ndarray],
                 alpha: float = 1,
                 beta: float = 1
                 ) -> None:

        self.images = np.array(images),
        self.masks = np.array(masks)
        self.alpha = alpha
        self.beta = beta

        self.class_num = int(self.images.shape[0])
        self.height = int(self.images.shape[1])
        self.width = int(self.images.shape[2])

    def __get_points_weights(self) -> np.ndarray:
        raise NotImplementedError

    def __get_edges_weights(self) -> np.ndarray:
        raise NotImplementedError

    def __get_f(self) -> np.ndarray:
        raise NotImplementedError

    def compose(self) -> np.ndarray:
        raise NotImplementedError

