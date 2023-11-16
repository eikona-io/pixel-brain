from torchvision.models import resnet50, ResNet50_Weights
from pixel_brain.pipeline import PipelineModule
from pixel_brain.pre_processors.resnet_preprocessor import ResnetPreprocessor
from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
from typing import List, Tuple, Dict
import torch
from pixel_brain.modules.imagenet_classes import IMAGENET_CLASS_ID_TO_NAME, IMAGENET_CLASS_NAME_TO_ID
import numpy as np



class ResnetClassifierModule(PipelineModule):
    """
    Classifies images into one of imagenet classes and stores the class in database.
    It can receive as input list of classes to choose from (subset of imagenet classes),
    out of which it will pick the one with largest probability.
    """
    def __init__(self, data: DataLoader, database: Database, class_subset: List[str] = [], filters: Dict[str, str] = None):
        """
        Initializes the ResnetClassifierModule with data, database, batch_size and ResnetPreprocessor.
        Stores class_subset as member and initializes torch resnet trained on imagenet.

        :param data: DataLoader object with data
        :param database: Database object for storing processed tags
        :param batch_size: Size of the batch to be processed
        :param class_subset: List of classes to choose from
        """
        super().__init__(data, database, ResnetPreprocessor(), filters)
        self._class_subset = np.array([self._get_class_id(class_name) for class_name in class_subset]) if class_subset else None
        self._resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()

    def _get_class_id(self, class_name: str) -> int:
        """
        Converts class name to class id.

        :param class_name: Name of the class
        :return: Class id
        """
        if class_name not in IMAGENET_CLASS_NAME_TO_ID:
            raise ValueError(f"Class {class_name} is not an imagenet class")
        return IMAGENET_CLASS_NAME_TO_ID[class_name]

    def _get_class_name(self, class_id: int) -> int:
        """
        Converts class id to class name.

        :param class_id: id of the class
        :return: Class name
        """
        return IMAGENET_CLASS_ID_TO_NAME[class_id]

    def _process(self, image_ids: Tuple[str], processed_image_batch: torch.Tensor):
        """
        Produces class probabilities using resnet member, chooses class as argmax over produced probs
        (on class_subset if provided else all probs) and stores class for each image in database member.

        :param image_ids: Tuple of image ids
        :param processed_image_batch: Batch of preprocessed images
        """
        with torch.no_grad():
            outputs = self._resnet(processed_image_batch)
            if self._class_subset is not None:
                outputs = outputs[:, self._class_subset]
            _, predicted = torch.max(outputs, 1)
            if self._class_subset is not None:
                predicted = self._class_subset[predicted]

            for i, image_id in enumerate(image_ids):
                predicted_class_name = self._get_class_name(predicted[i].item())
                self._database.store_field(image_id, 'class', predicted_class_name)


