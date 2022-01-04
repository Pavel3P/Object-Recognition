import cv2
from algorithms.cyk import CYK, Recognizer, Rules
from sys import argv
from time import time


horizontal = [
    ('i', 'i', 'v0'),
    ('i', 'i_', 'v0_'),
    ('i', 'v0', 'v0'),
    ('i', 'v0_', 'v0'),
    ('i', 'v1', 'v0_'),
    ('i', 'v1_', 'v0_'),

    ('i_', 'i', 'v1'),
    ('i_', 'i_', 'v1_'),
    ('i_', 'v0', 'v1'),
    ('i_', 'v0_', 'v1'),
    ('i_', 'v1', 'v1_'),
    ('i_', 'v1_', 'v1_')
]

vertical = [
    ('A00', '0', '0'),
    ('A01', '0', '1'),
    ('A10', '1', '0'),
    ('A11', '1', '1'),

    ('v0', 'A00', '0'),
    ('v0', 'A01', '1'),
    ('v0', 'A10', '1'),

    ('v0_', 'A11', '0'),

    ('v1', 'A00', '1'),

    ('v1_', 'A01', '0'),
    ('v1_', 'A10', '0'),
    ('v1_', 'A11', '1')
]

rename = [
    ('i', 'v0'),
    ('i_', 'v1')
]

terminal = ["0", "1"]
nonterminal = ["A00", "A01", "A10", "A11", "v0", "v1", "v0_", "v1_", "i", "i_"]


if __name__ == "__main__":
    if len(argv) == 4:
        image_to_check = argv[1]
        path_to_zero = argv[2]
        path_to_one = argv[3]
    else:
        # path_to_zero = "./data/cyk/0_w7-h8.png"
        # path_to_one = "./data/cyk/1_w7-h8.png"
        # image_to_check = "./data/cyk/example.png"
        raise ValueError("Input is empty.")

    image = cv2.imread(image_to_check, cv2.IMREAD_GRAYSCALE)
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

    start_t = time()
    ans = cyk(image, w_step=width, h_step=height, min_h=height, min_w=width, window_size_h_step=height, window_size_w_step=width)
    end_t = time()

    print(ans)
    print(f"Working time: {round(end_t - start_t, 2)}s")
    if ans:
        print(f"Transition: {(0, 3*height-1, 0, width-1, 'v0_') in cyk._f}")
