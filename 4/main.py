import sys
import time

from misc import save_img, preprocess
from composer import Composer


def main(path_to_folder: str = 'data/',
         result_filename: str = 'data/result.png',
         alpha: float = 1,
         beta: float = 1):
    t1 = time.time()
    imgs, masks = preprocess(path_to_folder)
    ans = Composer(imgs, masks, alpha, beta).compose()
    save_img(result_filename, ans)
    print(f'Total executing time: {time.time() - t1}')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(
            sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
        )
