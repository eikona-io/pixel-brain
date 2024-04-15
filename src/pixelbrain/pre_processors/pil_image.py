from typing import List
from pixelbrain.pipeline import Preprocessor
import torch
from torchvision import transforms
from PIL import Image

class PilImagePreprocessor(Preprocessor):
    def __init__(self) -> None:
        self._transforms = transforms.Compose([
            transforms.ToPILImage()
        ])
    def __call__(self, image_list: List[torch.Tensor]) -> List[Image.Image]:
        return [self._transforms(image) for image in image_list]
