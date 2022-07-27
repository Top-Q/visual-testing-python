import os

import cv2

from definitions import root_dir


def get_image(name: str):
    img = os.path.join(root_dir, "resources", f"{name}.png")
    return cv2.imread(img)
