from torchvision import transforms
from typing import List
import torch
from pixelbrain.pipeline import Preprocessor


class HogsDetectorPreprocessor(Preprocessor):
    def __init__(self):
        self._processing_steps = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Grayscale(),
            transforms.Lambda(lambda x: torch.squeeze(x, 0)),
        ])

    def __call__(self, image_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack([self._processing_steps(image) for image in image_list])
