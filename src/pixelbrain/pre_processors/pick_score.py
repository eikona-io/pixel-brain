from typing import List
from pixelbrain.pipeline import Preprocessor
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor


class PickScorePreprocessor(Preprocessor):
    def __init__(self) -> None:
        self._transforms = transforms.Compose([
            transforms.ToPILImage()
        ])
        self._pre_processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
    def __call__(self, image_list: List[torch.Tensor]) -> List[Image.Image]:
        pils = [self._transforms(image) for image in image_list]
        processed_images = self._pre_processor(
            images=pils,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return processed_images['pixel_values']
