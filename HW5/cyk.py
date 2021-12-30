import numpy as np
from rules import Rules
from recognizer import Recognizer
from tqdm import trange


class CYK:
    def __init__(self,
                 rules: Rules,
                 recognizer: Recognizer,
                 terminal: list[str],
                 nonterminal: list[str],
                 image_symbol: str) -> None:
        self.rules = rules
        self.recognizer = recognizer
        self.terminal = terminal
        self.nonterminal = nonterminal
        self.image_symbol = image_symbol
        self.__f: list[tuple[int, int, int, int, str]] = []

    def __call__(self, image: np.ndarray) -> bool:
        self.__f = self.__init_f(image)

        H = np.vectorize(lambda _img, _label: self.H(_img, _label, image.shape), signature="(m,n)->()", excluded=[1, "_label"])
        V = np.vectorize(lambda _img, _label: self.V(_img, _label, image.shape), signature="(m,n)->()", excluded=[1, "_label"])
        R = np.vectorize(lambda _img, _label: self.R(_img, _label, image.shape), signature="(m,n)->()", excluded=[1, "_label"])

        indexes = np.arange(np.product(image.shape)).reshape(image.shape)
        for h in trange(1, image.shape[0]):
            for w in range(1, image.shape[1]):
                sl_windows = self.sliding_window(indexes, w, h)
                sl_windows = sl_windows.reshape((sl_windows.shape[0] * sl_windows.shape[1],
                                                 sl_windows.shape[2], sl_windows.shape[3]))
                for label in self.terminal:
                    bool_h = H(sl_windows, label)
                    bool_v = V(sl_windows, label)
                    bool_r = R(sl_windows, label)
                    for window in sl_windows[bool_h | bool_v | bool_r]:
                        min_h, max_h, min_w, max_w = self.__coords_from_idxs(window, image.shape)

                        self.__f.append((min_h, max_h, min_w, max_w, label))

    @staticmethod
    def sliding_window(array: np.ndarray, width: int, height: int) -> np.ndarray:
        shape = ((array.shape[0] - height) + 1,) + ((array.shape[1] - width) + 1,) + (height, width)
        strides = array.strides[:-2] + (array.strides[-2],) + (array.strides[-1],) + array.strides[-2:]
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    def f(self, i1, i2, j1, j2, n) -> bool:
        return (i1, i2, j1, j2, n) in self.__f

    def __init_f(self, image: np.ndarray) -> list[tuple[int, int, int, int, str]]:
        f: list[tuple[int, int, int, int, str]] = []

        flatten_image = image.flatten()
        indexes = np.arange(np.product(image.shape)).reshape(image.shape)

        q = np.vectorize(lambda _window, _label: self.recognizer.recognize(flatten_image[_window.flatten()].reshape(_window.shape), _label),
                         signature="(m,n)->()", excluded=[1, "_label"])

        for h in range(1, image.shape[0]):
            for w in range(1, image.shape[1]):
                sl_windows = self.sliding_window(indexes, w, h)
                sl_windows = sl_windows.reshape((sl_windows.shape[0] * sl_windows.shape[1],
                                                 sl_windows.shape[2], sl_windows.shape[3]))
                for label in self.terminal:
                    for window in sl_windows[q(sl_windows, label)]:
                        min_h, max_h, min_w, max_w = self.__coords_from_idxs(window, image.shape)

                        f.append((min_h, max_h, min_w, max_w, label))

        return f

    @staticmethod
    def __coords_from_idxs(idx_img: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        if len(idx_img) == 0:
            return -1, -1, -1, -1

        unraveled_h, unraveled_w = np.unravel_index(idx_img.flatten(), image_shape)

        min_h = min(unraveled_h)
        max_h = max(unraveled_h)
        min_w = min(unraveled_w)
        max_w = max(unraveled_w)

        return min_h, max_h, min_w, max_w

    def H(self, idx_img: np.ndarray, label: str, image_shape: tuple[int, int]) -> bool:
        for w in range(0, idx_img.shape[0]):  # TODO: idx_img.shape[0]+1?
            left_part = idx_img[:w, :]
            lmin_h, lmax_h, lmin_w, lmax_w = self.__coords_from_idxs(left_part, image_shape)

            right_part = idx_img[w:, :]
            rmin_h, rmax_h, rmin_w, rmax_w = self.__coords_from_idxs(right_part, image_shape)

            for nl in self.nonterminal:
                for nr in self.nonterminal:
                    if self.f(lmin_h, lmax_h, lmin_w, lmax_w, nl) \
                            and self.rules.gh(label, nl, nr) \
                            and self.f(rmin_h, rmax_h, rmin_w, rmax_w, nr):
                        return True

        return False

    def V(self, idx_img: np.ndarray, label: str, image_shape: tuple[int, int]) -> bool:
        for h in range(0, idx_img.shape[1]):  # TODO: idx_img.shape[1]+1?
            upper_part = idx_img[:h, :]
            umin_h, umax_h, umin_w, umax_w = self.__coords_from_idxs(upper_part, image_shape)

            down_part = idx_img[h:, :]
            dmin_h, dmax_h, dmin_w, dmax_w = self.__coords_from_idxs(down_part, image_shape)

            for nu in self.nonterminal:
                for nd in self.nonterminal:
                    if self.f(umin_h, umax_h, umin_w, umax_w, nu) \
                            and self.rules.gv(label, nu, nd) \
                            and self.f(dmin_h, dmax_h, dmin_w, dmax_w, nd):
                        return True

        return False

    def R(self, idx_img: np.ndarray, label: str, image_shape: tuple[int, int]) -> bool:
        min_h, max_h, min_w, max_w = self.__coords_from_idxs(idx_img, image_shape)

        for t in self.terminal:
            if self.f(min_h, max_h, min_w, max_w, t) and self.rules.g(label, t):
                return True

        return False
