from typing import List
import torch
from torchvision import transforms
from pixel_brain.pipeline import Preprocessor


class ResnetPreprocessor(Preprocessor):
    """
    Preprocessor for Resnet model. It performs resizing, center cropping and normalization on a list of images.
    """
    def __init__(self):
        """
        Initializes the ResnetPreprocessor with a sequence of transformations to be applied on the images.
        """
        self._processing_steps = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Lambda(lambda x: x / 256),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Applies the sequence of transformations on the list of images and returns a tensor of the transformed images.
        
        :param image_list: List of images to be preprocessed
        :return: Tensor of preprocessed images
        """
        assert isinstance(image_list, list), "image_list must be a list of tensors"
        return torch.stack([self._processing_steps(image) for image in image_list])
