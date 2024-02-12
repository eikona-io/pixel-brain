from cloudinary.exceptions import AuthorizationRequired
import torchvision.transforms as transforms
from typing import List, Optional, Tuple
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from overrides import overrides 
from pixelbrain.utils import get_logger
import cloudinary, cloudinary.api
import requests
from PIL import Image
from io import BytesIO
from os import environ
import numpy as np
import torch


MAX_RESULTS = 1000


class CloudinaryDataLoader(DataLoader):
    """
    This class is a custom dataloader for loading images from Cloudinary
    Note - The cloudinary connection depends on the credentials being set as environment variables in the format:
    CLOUDINARY_URL=cloudinary://<api_key>:<api_secret>@<cloud_name>
    """
    def __init__(self, cloudinary_folder_prefix, database: Database):
        super().__init__(images_path=cloudinary_folder_prefix, database=database, batch_size=1, decode_images=True, load_images=True)
        self._logger = get_logger("CloudinaryDataLoader")
        if not environ.get('CLOUDINARY_URL'):
            self._logger.error("Cloudinary credentials not found. Please set them as environment variables.")
            raise ValueError("Cloudinary credentials not found. Please set them as environment variables.")
        self.test_cloudinary_connection()

    @overrides
    def __next__(self) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Returns the next batch of loaded images
        :returns:
        ids_batch: List[str]
        image_batch: List[torch.Tensor]
        """
        image_batch, ids_batch = [], []
        for _ in range(self._batch_size):
            if not self._image_paths:
                if not image_batch:
                    # no data left
                    raise StopIteration
                break
            image_url = self._image_paths.pop(0)
            image_id = image_url.split('/')[-1].split('.')[0]
            self._database.add_image(image_id, image_url)
            image = self._load_image(image_url) if self._load_images else None
            image_batch.append(image)
            ids_batch.append(image_id)
        return ids_batch, image_batch

    @overrides
    def _get_all_image_paths(self) -> List[str]:
        """
        Gets all image paths from the database if remote, or uses glob if local
        """
        raw_results = cloudinary.api.resources(type = "upload", prefix = self._images_path, max_results=MAX_RESULTS)
        resources = raw_results['resources']
        if len(resources) == 0:
            # self._logger.warning(f"No images found in {self._images_path}")
            return []
        else:
            # self._logger.info(f"Found {len(resources)} images in {self._images_path}")
            pass
        return [r['secure_url'] for r in resources]

    @overrides
    def _load_image(self, image_path):
        """
        Loads image from local or cloud
        """
        response = requests.get(image_path)
        img_pil = Image.open(BytesIO(response.content))
        transform = transforms.Compose([ 
                                transforms.PILToTensor() 
                                ])
        return transform(img_pil)

    def test_cloudinary_connection(self):
        try:
            cloudinary.api.ping()
            # self._logger.info("Cloudinary connection successful.")
        except AuthorizationRequired as e:
            # self._logger.error("Cloudinary connection failed. Please check your credentials.")
            raise e
        except Exception as e:
            # self._logger.error("Cloudinary connection failed. Please check your internet connection.")
            raise e
