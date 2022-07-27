import os.path

from numpy import ndarray
from skimage.metrics import structural_similarity

import cv2
import numpy as np

from definitions import root_dir


def crop_from_template(full_img: ndarray, template: ndarray) -> ndarray:
    w, h, _ = template.shape
    res = cv2.matchTemplate(full_img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return full_img[max_loc[1]:max_loc[1] + w, max_loc[0]:max_loc[0] + h, :]


def compare_imgs(img0: ndarray, img1: ndarray, show_diff: bool = False):
    # Convert images to grayscale
    first_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, diff = structural_similarity(first_gray, second_gray, full=True)
    similarity = score * 100
    if show_diff:
        show_diffs(diff, img0, img1)
    return similarity


def show_diffs(diff, img0, img1):
    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type so we must convert the array
    # to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    # Threshold the difference image, followed by finding contours to
    # obtain the regions that differ between the two images
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # Highlight differences
    mask = np.zeros(img0.shape, dtype='uint8')
    filled = img1.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img0, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            cv2.drawContours(filled, [c], 0, (0, 255, 0), -1)
    cv2.imshow('first', img0)
    cv2.imshow('second', img1)
    cv2.imshow('diff', diff)
    cv2.imshow('mask', mask)
    cv2.imshow('filled', filled)
    cv2.waitKey()


if __name__ == '__main__':
    full_image = cv2.imread(os.path.join(root_dir, "resources", "imgs", "kitten.png"))
    template = cv2.imread(os.path.join(root_dir, "resources", "imgs", "kitten_template_different.png"))
    crop_img = crop_from_template(full_image, template)
    similarity = compare_imgs(crop_img, template)
    print("Similarity Score: {:.3f}%".format(similarity))