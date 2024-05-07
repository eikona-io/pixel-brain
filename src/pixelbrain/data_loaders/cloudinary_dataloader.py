from cloudinary.exceptions import AuthorizationRequired
import torchvision.transforms as transforms
from typing import List, Optional, Tuple
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from overrides import overrides 
from pixelbrain.utils import get_logger
import cloudinary, cloudinary.api
import requests
from PIL import Image, ImageOps
from io import BytesIO
import os
from os import environ
import torch


MAX_RESULTS = 1000


class CloudinaryDataLoader(DataLoader):
    """
    This class is a custom dataloader for loading images from Cloudinary
    Note - The cloudinary connection depends on the credentials being set as environment variables in the format:
    CLOUDINARY_URL=cloudinary://<api_key>:<api_secret>@<cloud_name>
    """
    def __init__(self, cloudinary_folder_prefix, database: Database, batch_size: int = 1, is_recursive: bool = False):
        super().__init__(images_path=cloudinary_folder_prefix, database=database, batch_size=batch_size, 
                         decode_images=True, load_images=True, is_recursive=is_recursive)
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
        self._lazy_load_image_paths_if_needed()
        image_batch, ids_batch = [], []
        for _ in range(self._batch_size):
            if not self._image_paths:
                if not image_batch:
                    # no data left
                    raise StopIteration
                break
            image_public_id = self._image_paths.pop(0)
            image_url = cloudinary.CloudinaryImage(image_public_id).build_url()
            self._database.add_image(image_public_id, image_url)
            image = self._load_image(image_url) if self._load_images else image_url
            image_batch.append(image)
            ids_batch.append(image_public_id)
        return ids_batch, image_batch

    @overrides
    def _get_all_image_paths(self) -> List[str]:
        """
        Gets all the images public ids from the prefix provided
        """
        raw_results = cloudinary.api.resources(type = "upload", prefix = self._images_path, max_results=MAX_RESULTS)
        resources = raw_results['resources']
        if len(resources) == 0:
            return []
        all_paths = [r['public_id'] for r in resources]
        if not self.is_recursive:
            return [path for path in all_paths if os.path.split(path)[0] == self._images_path]
        else:
            return all_paths

    @overrides
    def _load_image(self, image_path):
        """
        Loads image from local or cloud
        """
        response = requests.get(image_path)
        img_pil = Image.open(BytesIO(response.content)).convert('RGB')
        img_pil = ImageOps.exif_transpose(img_pil)
        transform = transforms.Compose([ 
                                transforms.PILToTensor() 
                                ])
        return transform(img_pil)

    def test_cloudinary_connection(self):
        try:
            cloudinary.api.ping()
        except AuthorizationRequired as e:
            raise e
        except Exception as e:
            raise e
