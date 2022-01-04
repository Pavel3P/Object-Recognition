import numpy as np
from algorithms.cyk.rules import Rules
from algorithms.cyk.recognizer import Recognizer


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
        self._f: list[tuple[int, int, str]] = []

    def __call__(self,
                 image: np.ndarray,
                 w_step: int = 1,
                 h_step: int = 1,
                 min_h: int = 1,
                 min_w: int = 1,
                 window_size_h_step: int = 1,
                 window_size_w_step: int = 1) -> bool:
        self._f = self.init_f(image, w_step, h_step, min_h, min_w, window_size_h_step, window_size_w_step)

        indexes = np.arange(np.product(image.shape)).reshape(image.shape)
        for h in range(min_h, image.shape[0]+1, window_size_h_step):
            for w in range(min_w, image.shape[1]+1, window_size_w_step):
                sl_windows = self.sliding_window(indexes, w, h, w_step, h_step)
                sl_windows = sl_windows.reshape((sl_windows.shape[0] * sl_windows.shape[1],
                                                 sl_windows.shape[2], sl_windows.shape[3]))
                for label in self.nonterminal:
                    for window in sl_windows:
                        if self.H(window, label, min_w, window_size_w_step):
                            self._f.append((window[0, 0], window[-1, -1], label))
                        elif self.V(window, label, min_h, window_size_h_step):
                            self._f.append((window[0, 0], window[-1, -1], label))
                        elif self.R(window, label):
                            self._f.append((window[0, 0], window[-1, -1], label))

        return (0, image.shape[0] * image.shape[1] - 1, self.image_symbol) in self._f

    @staticmethod
    def sliding_window(array: np.ndarray,
                       width: int,
                       height: int,
                       w_step: int = 1,
                       h_step: int = 1) -> np.ndarray:
        shape = ((array.shape[0] - height) // h_step + 1,) + ((array.shape[1] - width) // w_step + 1,) + (height, width)
        strides = array.strides[:-2] + (array.strides[-2] * h_step,) + (array.strides[-1] * w_step,) + array.strides[-2:]
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    def init_f(self,
                 image: np.ndarray,
                 w_step: int = 1,
                 h_step: int = 1,
                 min_h: int = 1,
                 min_w: int = 1,
                 window_size_h_step: int = 1,
                 window_size_w_step: int = 1) -> list[tuple[int, int, str]]:
        f: list[tuple[int, int, str]] = []

        flatten_image = image.flatten()
        indexes = np.arange(np.product(image.shape)).reshape(image.shape)

        for h in range(min_h, image.shape[0]+1, window_size_h_step):
            for w in range(min_w, image.shape[1]+1, window_size_w_step):
                sl_windows = self.sliding_window(indexes, w, h, w_step, h_step)
                sl_windows = sl_windows.reshape((sl_windows.shape[0] * sl_windows.shape[1],
                                                 sl_windows.shape[2], sl_windows.shape[3]))
                for label in self.terminal:
                    for window in sl_windows:
                        if self.recognizer.recognize(flatten_image[window.flatten()].reshape(window.shape), label):
                            f.append((window[0, 0], window[-1, -1], label))

        return f

    def H(self, idx_img: np.ndarray, label: str, min_w: int = 1, w_step: int = 1) -> bool:
        for w in range(min_w, idx_img.shape[1], w_step):
            left_part = idx_img[:, :w]
            right_part = idx_img[:, w:]
            for nl in self.nonterminal + self.terminal:
                for nr in self.nonterminal + self.terminal:
                    if (left_part[0, 0], left_part[-1, -1], nl) in self._f \
                            and self.rules.gh(label, nl, nr) \
                            and (right_part[0, 0], right_part[-1, -1], nr) in self._f:
                        return True

        return False

    def V(self, idx_img: np.ndarray, label: str, min_h: int = 1, h_step: int = 1) -> bool:
        for h in range(min_h, idx_img.shape[0], h_step):
            upper_part = idx_img[:h, :]
            down_part = idx_img[h:, :]
            for nu in self.nonterminal + self.terminal:
                for nd in self.nonterminal + self.terminal:
                    if (upper_part[0, 0], upper_part[-1, -1], nu) in self._f \
                            and self.rules.gv(label, nu, nd) \
                            and (down_part[0, 0], down_part[-1, -1], nd) in self._f:
                        return True

        return False

    def R(self, idx_img: np.ndarray, label: str) -> bool:
        for t in self.terminal + self.nonterminal:
            if (idx_img[0, 0], idx_img[-1, -1], t) in self._f and self.rules.g(label, t):
                return True

        return False
