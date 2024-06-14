import os
import logging
import sys
import time
from typing import Tuple


PIXELBRAIN_PATH = os.getenv("PIXELBRAIN_PATH", None)
MONGODB_ATLAS_KEY = os.getenv("MONGODB_ATLAS_KEY", None)
OPENAI_KEY = os.getenv("OPENAI_KEY", None)
PINECONE_KEY = os.getenv("PINECONE_KEY", None)


def get_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def create_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def increase_rectangle_space(
    x: float, y: float, w: float, h: float, ratio: float = 1.5
) -> Tuple[int, int, int, int]:
    """
    Increases a rectangle's space by a given ratio.
    :param x: x coordinate of the rectangle
    :param y: y coordinate of the rectangle
    :param w: width of the rectangle
    :param h: height of the rectangle
    :param ratio: the ratio to increase the rectangle space by
    :return: new x, new y, new w, new h
    """
    new_w = w * ratio
    new_h = h * ratio
    new_x = max(0, x - (new_w - w) * 0.5)
    new_y = max(0, y - (new_h - h) * 0.5)
    return int(new_x), int(new_y), int(new_w), int(new_h)


def get_cloudinary_images_from_prefix(prefix: str):
    import cloudinary.api

    raw_results = cloudinary.api.resources(type="upload", prefix=prefix)
    resources = raw_results["resources"]
    all_paths = [r["public_id"] for r in resources]
    return all_paths


def delete_image_from_cloudinary(image_public_id: str):
    """
    Deletes an image from Cloudinary using its public ID.

    Parameters:
    - image_public_id: The public ID of the image to delete.

    Returns:
    - A dictionary containing the result of the deletion operation.
    """
    import cloudinary.uploader

    result = cloudinary.uploader.destroy(image_public_id)
    return result
