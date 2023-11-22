from typing import List
from pixelbrain.pipeline import Preprocessor
import torch
from torchvision import transforms


class DeepfacePreprocessor(Preprocessor):
    def __init__(self) -> None:
        self._transforms = transforms.Compose([
            transforms.Lambda(lambda x : torch.permute(x, [1, 2, 0])) # CHW TO HWC
        ])
    def __call__(self, image_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self._transforms(image) for image in image_list]