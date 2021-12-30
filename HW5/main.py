import cv2

from rules import Rules
from misc import construct_image
from recognizer import Recognizer
from cyk import CYK
from sys import argv
from time import time
from rules_variables import (horizontal,
            vertical, rename, terminal, nonterminal)


if __name__ == "__main__":
    if len(argv) == 4:
        high = argv[1]
        low = argv[2]
        result = argv[3]
    else:
        raise ValueError("Input is empty.")

    if len(argv) > 4:
        path_to_zero = argv[4]
        path_to_one = argv[5]
    else:
        path_to_zero = "terminal_symbols/0_w7-h8.png"
        path_to_one = "terminal_symbols/1_w7-h8.png"

    zero_sample = cv2.imread(path_to_zero, cv2.IMREAD_GRAYSCALE)
    one_sample = cv2.imread(path_to_one, cv2.IMREAD_GRAYSCALE)
    samples = {
        "0": zero_sample,
        "1": one_sample
    }

    rules = Rules()

    for h in horizontal:
        rules.create_gh(*h)

    for v in vertical:
        rules.create_gv(*v)

    for r in rename:
        rules.create_g(*r)

    recognizer = Recognizer(samples=samples)
    cyk = CYK(rules, recognizer, terminal, nonterminal, "i")

    height = zero_sample.shape[0]
    width = zero_sample.shape[1]

    image = construct_image(one_sample, zero_sample, high, low, result)

    start_t = time()
    ans = cyk(image, w_step=width, h_step=height, min_h=height, min_w=width, window_size_h_step=height, window_size_w_step=width)
    end_t = time()

    print(ans)
    print(f"Working time: {round(end_t - start_t, 2)}s")
    if ans:
        print(f"Transition: {cyk.f(0, 3*height-1, 0, width-1, 'v0_')}")
