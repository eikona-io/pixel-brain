import os
import torchvision.transforms as transforms
from typing import List, Union, Dict
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipeline import PipelineModule
from overrides import overrides


class ImageSaveModule(PipelineModule):
    """
    A module to save processed images to a specified directory.
    """

    def __init__(self, data, database, save_dir, pre_processor=None, filters=None):
        """
        Initialize the ImageSaveModule.

        :param data: DataLoader object with data
        :param database: Database object for storing processed tags
        :param save_dir: Directory where images will be saved
        :param pre_processor: Preprocessor object to preprocess the data, if None, preprocessing won't be done.
        :param filters: (field_name, field_value) to apply on the dataloader before starting it processing (or a custom filter)
        """
        super().__init__(data, database, pre_processor, filters)
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)
        self._nof_saved_images = 0

    @overrides
    def _process(self, image_ids, processed_image_batch):
        """
        Process the data and save images.

        :param image_ids: List of image ids
        :param processed_image_batch: Batch of preprocessed images
        """
        for image_id, image_tensor in zip(image_ids, processed_image_batch):
            image = transforms.ToPILImage()(image_tensor)
            save_path = os.path.join(self._save_dir, f"{self._nof_saved_images}.png")
            image.save(save_path)
            self._nof_saved_images += 1