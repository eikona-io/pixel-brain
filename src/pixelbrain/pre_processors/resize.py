from typing import List
import torch
from torchvision import transforms
from pixelbrain.pipeline import Preprocessor


class ResizePreprocessor(Preprocessor):
    """
    Preprocessor that only resizes images.
    """

    def __init__(self, size: int):
        """
        Initializes the ResizePreprocessor with a resize transformation.

        :param size: The size to which the images will be resized.
        """
        self._processing_steps = transforms.Compose(
            [
                transforms.Resize(size),
            ]
        )

    def __call__(self, image_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies the resize transformation on the list of images and returns a tensor of the resized images.

        :param image_list: List of images to be resized
        :return: Tensor of resized images
        """
        assert isinstance(image_list, list), "image_list must be a list of tensors"
        return [self._processing_steps(image) for image in image_list]
