import torch
from typing import List, Tuple, Union
from tempfile import TemporaryDirectory
from pixelbrain.database import Database
import os
import boto3
import glob
import tempfile
from torchvision.io import read_image, read_file, ImageReadMode
from abc import ABC, abstractmethod
import math
import requests
import copy
import threading
import time


class DataLoaderFilter(ABC):
    @abstractmethod
    def filter(self, database: Database, image_ids: List[str]) -> List[str]:
        """This defines a filter over the image_ids according to values in the database
        :param: database: Database object with image metadata
        :param: image_ids: list of image ids to filter from
        :return filtered image_ids out of the given image_ids
        """
        pass


class DataLoader:
    """
    DataLoader class that loads and decodes images either from disk, S3, or remote URLs
    """

    def __init__(
        self,
        images_path: Union[str, List[str]],
        database: Database,
        batch_size=1,
        decode_images=True,
        load_images=True,
        is_recursive: bool = True,
    ):
        """
        Initializes the DataLoader with images path, database and batch size

        :param images_path: The path to the images. Can be a local path, S3 path or web URLs.
        :param database: The database object to use for storing image metadata.
        :param batch_size: The number of images to load at a time. Default is 1.
        :param decode_images: Whether to decode the images. Default is True.
        :param decode_images: Whether to load the images. Default is True.
        """
        self._images_path = images_path
        self._database = database
        self._batch_size = batch_size
        self.is_recursive = is_recursive
        self._image_paths = None
        self._tempdir = TemporaryDirectory()
        self._decode_images = decode_images
        self._load_images = load_images
        self._url_cache = {}
        self._url_download_thread = None

        if isinstance(images_path, list) and any(
            url.startswith("http") for url in images_path
        ):
            self._start_url_download_thread()

    def _start_url_download_thread(self):
        """This thread will download the images from the URLs in the list asynchronously"""

        def download_images():
            for url in self._images_path:
                if url.startswith("http"):
                    response = requests.get(url)
                    response.raise_for_status()
                    temp_filename = f'{self._tempdir.name}/{url.split("/")[-1]}'
                    with open(temp_filename, "wb") as temp_file:
                        temp_file.write(response.content)
                        self._url_cache[url] = temp_file.name

        self._url_download_thread = threading.Thread(
            target=download_images, daemon=True
        )
        self._url_download_thread.start()

    def __next__(self) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Returns the next batch of loaded images
        :returns:
        ids_batch: List[str]
        image_batch: List[torch.Tensor] or List[str] if not load_images
        """

        self._lazy_load_image_paths_if_needed()
        image_batch, ids_batch = [], []

        for _ in range(self._batch_size):
            if not self._image_paths:
                if not image_batch:
                    # no data left
                    raise StopIteration
                break
            image_path = self._pop_image_path()
            image_id = f"{image_path}"
            self._database.add_image(image_id, image_path)
            image = self._load_image(image_path) if self._load_images else image_path
            image_batch.append(image)
            ids_batch.append(image_id)
        return ids_batch, image_batch

    def _pop_image_path(self):
        image_path = self._image_paths.pop(0)
        if os.path.exists(image_path):
            image_path = os.path.realpath(image_path)
        return image_path

    def _lazy_load_image_paths_if_needed(self):
        if self._image_paths is None:
            self._image_paths = self._get_all_image_paths()
            if not self._image_paths:
                raise ValueError(f"No images found in {self._images_path}")

    def __iter__(self):
        return self

    def __len__(self):
        self._lazy_load_image_paths_if_needed()
        return int(math.ceil(len(self._image_paths) / self._batch_size))

    def _load_image(self, image_path: str):
        """
        Loads image from local, cloud, or remote URL
        """
        if image_path.startswith("s3://"):
            # Load image from S3
            image = self._load_image_from_s3(image_path)
        elif image_path.startswith("http://") or image_path.startswith("https://"):
            # Load image from remote URL
            image = self._load_image_from_url(image_path)
        else:
            # Load image from local
            image = self._load_image_from_local(image_path)
        return image

    def _load_image_from_s3(self, image_path):
        """
        Loads image from S3
        """
        s3 = boto3.client("s3")
        bucket_name, key = image_path.replace("s3://", "").split("/", 1)
        s3.download_file(bucket_name, key, os.path.join(self._tempdir.name, key))
        return self._read_image(os.path.join(self._tempdir.name, key))

    def _load_image_from_local(self, image_path):
        """
        Loads image from local
        """
        return self._read_image(image_path)

    def _load_image_from_url(self, image_url):
        """
        Loads image from a remote URL
        """
        while image_url not in self._url_cache:
            time.sleep(0.1)  # Polling interval
        temp_file_path = self._url_cache[image_url]
        return self._read_image(temp_file_path)

    def _get_all_image_paths(self) -> List[str]:
        """
        Gets all image paths from the database if remote, or uses glob if local
        """
        if isinstance(self._images_path, list):
            # images urls were explicitly provided upon instantiation
            return copy.deepcopy(self._images_path)
        elif self._images_path.startswith("s3://"):
            # Query S3 for image paths
            s3 = boto3.client("s3")
            bucket_name = self._images_path.replace("s3://", "").split("/")[0]
            return [obj.key for obj in s3.list_objects(Bucket=bucket_name)["Contents"]]
        else:
            # Use glob to find image paths locally, only including common image file extensions
            image_extensions = ["jpg", "jpeg", "png", "PNG", "JPEG", "JPG"]
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(
                    glob.glob(
                        os.path.join(self._images_path, f"**/*.{ext}"),
                        recursive=self.is_recursive,
                    )
                )
            return image_paths

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
        return (
            read_image(image_path, ImageReadMode.RGB)
            if self._decode_images
            else read_file(image_path)
        )

    def _get_image_from_path(self, image_path: str) -> str:
        image_fullpath = os.path.realpath(image_path)
        image_doc = self._database.find_images_with_value("image_path", image_fullpath)
        if not image_doc:
            raise ValueError(f"Could not find image with image_path: {image_fullpath}")
        assert len(image_doc) == 1, "Only one image doc should have a certain path"
        return image_doc[0]

    def filter(self, field_name: str, field_value=None):
        """
        Filters images according to the values in database
        :param field_name: field to filter upon
        :param field_value: value to compare to. If none, will accept all field values (only check that field_name is present in metadata)
        """

        self._lazy_load_image_paths_if_needed()
        filtered_paths = []
        for image_path in self._image_paths:
            image_doc = self._get_image_from_path(image_path)
            if field_name in image_doc:
                if field_value is None:
                    filtered_paths.append(image_path)
                else:
                    if image_doc[field_name] == field_value:
                        filtered_paths.append(image_path)

        self._image_paths = filtered_paths

    def custom_filter(self, filter: DataLoaderFilter):
        self._lazy_load_image_paths_if_needed()
        image_ids = [
            self._get_image_from_path(path)["_id"] for path in self._image_paths
        ]
        filtered_ids = filter.filter(self._database, image_ids)
        self._image_paths = [
            self._database.find_image(id)["image_path"] for id in filtered_ids
        ]
