from typing import List
import torch
from torchvision import transforms
from pixelbrain.pipeline import Preprocessor


class SSIMPreprocessor(Preprocessor):
    """
    Preprocessor that resizes images to a maximum size while preserving aspect ratio.
    """

    def __init__(self, max_image_size: int):
        """
        Initializes the SSIMPreprocessor with a maximum image size.

        :param max_image_size: The maximum size (either width or height) to which the image should be resized.
        """
        self._max_image_size = max_image_size
        self._resize_transform = transforms.Resize(max_image_size, antialias=True)

    def __call__(self, image_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Resizes the images to the maximum size while preserving aspect ratio.

        :param image_list: List of images to be preprocessed
        :return: List of preprocessed images
        """
        resized_images = []
        for image in image_list:
            resized_image = self._resize_transform(image)
            resized_images.append(resized_image)
        return resized_images
