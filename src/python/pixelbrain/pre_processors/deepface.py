from typing import List
from pixelbrain.pipeline import Preprocessor
import torch
from torchvision import transforms


class DeepfacePreprocessor(Preprocessor):
    def __init__(self) -> None:
        self._transforms = transforms.Compose([
            transforms.Lambda(lambda x : torch.transpose(x, 0, 2)), # CHW TO HWC
            transforms.Lambda(lambda x : torch.flip(x, dims=[2])) # BGR to RGB
        ])
    def __call__(self, image_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self._transforms(image) for image in image_list]