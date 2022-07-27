from infra.image_compare import crop_from_template, compare_imgs
from infra.image_repo import get_image

show_diffs = True


def test_identical_part():
    full_image = get_image("kitten")
    identical_part = get_image("kitten_part_identical")
    crop_img = crop_from_template(full_image, identical_part)
    similarity = compare_imgs(crop_img, identical_part, show_diff=show_diffs)
    assert similarity == 100


def test_not_identical_part():
    full_image = get_image("kitten")
    identical_part = get_image("kitten_part_not_identical")
    crop_img = crop_from_template(full_image, identical_part)
    similarity = compare_imgs(crop_img, identical_part, show_diff=show_diffs)
    assert similarity < 99
