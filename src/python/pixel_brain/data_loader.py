
import torch
from typing import List, Tuple
from tempfile import TemporaryDirectory
from pixel_brain.database import Database
import os
import boto3
import glob
import random
from torchvision.io import read_image, read_file


class DataLoader:
    """
    DataLoader class that loads and decodes images either from disk or S3
    """
    def __init__(self, images_path, database: Database, batch_size=1, decode_images=True):
        """
        Initializes the DataLoader with images path, database and batch size

        :param images_path: The path to the images. Can be a local path or an S3 path.
        :param database: The database object to use for storing image metadata.
        :param batch_size: The number of images to load at a time. Default is 1.
        :param decode_images: Whether to decode the images. Default is True.
        """
        self._images_path = images_path
        self._database = database
        self._batch_size = batch_size
        self._image_paths = self._get_all_image_paths()
        self._tempdir = TemporaryDirectory()
        self._decode_images = decode_images

    def __next__(self) -> List[Tuple[str, torch.Tensor]]:
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
            image_path = self._image_paths.pop(0)
            image_id = f"{image_path}"
            self._database.add_image(image_id, image_path)
            image = self._load_image(image_path)
            image_batch.append(image)
            ids_batch.append(image_id)
        return ids_batch, image_batch

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._image_paths) // self._batch_size

    def _load_image(self, image_path):
        """
        Loads image from local or cloud
        """
        if self._images_path.startswith('s3://'):
            # Load image from S3
            image = self._load_image_from_s3(image_path)
        else:
            # Load image from local
            image = self._load_image_from_local(image_path)
        return image

    def _load_image_from_s3(self, image_path):
        """
        Loads image from S3
        """
        s3 = boto3.client('s3')
        bucket_name, key = image_path.replace('s3://', '').split('/', 1)
        s3.download_file(bucket_name, key, os.path.join(self._tempdir.name, key))
        return self._read_image(os.path.join(self._tempdir.name, key))

    def _load_image_from_local(self, image_path):
        """
        Loads image from local
        """
        return self._read_image(image_path)

    def _get_all_image_paths(self):
        """
        Gets all image paths from the database if remote, or uses glob if local
        """
        if self._images_path.startswith('s3://'):
            # Query S3 for image paths
            s3 = boto3.client('s3')
            bucket_name = self._images_path.replace('s3://', '').split('/')[0]
            return [obj.key for obj in s3.list_objects(Bucket=bucket_name)['Contents']]
        else:
            # Use glob to find image paths locally
            return glob.glob(os.path.join(self._images_path, '**/*.*'), recursive=True)

    def clone(self):
        """
        Returns a clone of the dataloader at current time
        """
        return DataLoader(self._images_path, self._database, self._batch_size)


    def set_batch_size(self, batch_size: int):
        """
        Change batch size
        """
        self._batch_size = batch_size

    def _read_image(self, image_path):
        return read_image(image_path) if self._decode_images else read_file(image_path)



