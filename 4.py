import sys
import time
from algorithms.csp import save_img, preprocess
from algorithms.csp import Composer


def main(path_to_folder: str,
         result_filename: str,
         alpha: float = 1,
         beta: float = 1):
    t1 = time.time()
    imgs, masks = preprocess(path_to_folder)
    ans = Composer(imgs, masks, alpha, beta).compose()
    save_img(result_filename, ans)
    print(f'Total executing time: {time.time() - t1}s')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    else:
        raise ValueError("You must specify at least path to input images and path to output's one.")

