import numpy as np


class Composer:
    def __init__(self,
                 images: list[np.ndarray],
                 masks: list[np.ndarray],
                 alpha: float = 1,
                 beta: float = 1
                 ) -> None:

        self.images = np.array(images) / 255
        self.masks = np.array(masks)
        self.alpha = alpha
        self.beta = beta

        self.class_num = int(self.images.shape[0])
        self.height = int(self.images.shape[1])
        self.width = int(self.images.shape[2])

        self.__point_weights = self.__get_points_weights()
        self.__edges_weights = self.__get_edges_weights()
        self.__f = self.__get_f()

    def __get_points_weights(self) -> np.ndarray:
        point_weights = np.zeros((self.class_num, self.height, self.width+1))
        point_weights[:, :, :-1] = self.alpha * ~self.masks
        return point_weights

    def __get_edges_weights(self) -> np.ndarray:
        edge_weights = np.zeros(
            (self.class_num, self.class_num, self.height, self.width)
        )
        for k1 in range(self.class_num):
            for k2 in range(self.class_num):
                diff = np.linalg.norm(
                    self.images[k1] - self.images[k2], axis=2, ord=1
                )
                edge_weights[k1, k2, :, :-1] = self.beta * (diff[:, :-1] + diff[:, 1:])
        return edge_weights

    def __get_f(self) -> np.ndarray:
        f = np.zeros((self.class_num, self.height, self.width + 1))
        for w in range(self.width - 1, -1, -1):
            for k in range(self.class_num):
                f[k, :, w] = np.min(self.__point_weights[:, :, w + 1] + self.__edges_weights[k, :, :, w] + f[:, :, w + 1], axis=0)

        return f

    def compose(self) -> np.ndarray:
        labels = np.argmin(self.__f, axis=0)
        image = np.zeros((self.height, self.width, self.images.shape[-1]))
        for h in range(self.height):
            for w in range(self.width):
                image[h, w, :] = self.images[labels[h, w], h, w, :]

        return image * 255
