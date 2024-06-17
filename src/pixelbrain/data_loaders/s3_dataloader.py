import boto3
import os
from typing import Union, List, Optional
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
import copy
from overrides import overrides
import tempfile
from pixelbrain.utils import list_s3_objects
from botocore.exceptions import NoCredentialsError


class S3DataLoader(DataLoader):
    """
    S3DataLoader class that inherits from DataLoader and loads images specifically from an S3 bucket.
    """

    def __init__(
        self,
        s3_paths_or_prefix: Union[str, List[str]],
        bucket_name: str,
        database: Database,
        batch_size=1,
        decode_images=True,
        load_images=True,
        is_recursive: bool = True,
        max_items: Optional[int] = None,
    ):
        """
        Initializes the S3DataLoader with S3 paths, database, and batch size.

        :param s3_paths: The S3 prefix or list of S3 paths to the images.
        :param database: The database object to use for storing image metadata.
        :param batch_size: The number of images to load at a time. Default is 1.
        :param decode_images: Whether to decode the images. Default is True.
        :param load_images: Whether to load the images. Default is True.
        :param max_items: The maximum number of items to yield. Default is None (no limit).
        """
        super().__init__(
            s3_paths_or_prefix,
            database,
            batch_size,
            decode_images,
            load_images,
            is_recursive,
            max_items,
        )
        self._bucket_name = bucket_name
        self._tmpdir = tempfile.TemporaryDirectory()
        self._link_expiration = 3600  # 1hr

    def _load_image(self, image_path: str):
        """
        Loads image from S3
        """
        return self._load_image_from_s3(image_path)

    def _get_all_image_paths(self) -> List[str]:
        """
        Gets all image paths from the S3 bucket or list of S3 paths
        """
        if isinstance(self._images_path, list):
            # S3 paths were explicitly provided upon instantiation
            return copy.deepcopy(self._images_path)
        else:
            return list_s3_objects(self._bucket_name, self._images_path)

    def _load_image_from_s3(self, image_path):
        """
        Loads image from S3
        """
        s3 = boto3.client("s3")
        local_path = os.path.join(str(self._tmpdir.name), os.path.basename(image_path))
        s3.download_file(self._bucket_name, image_path, local_path)
        return self._read_image(local_path)

    @overrides
    def _filter_by_field(self):
        return "_id"

    def _get_image_path_from_image_id(self, image_id: str):
        s3_client = boto3.client("s3")
        try:
            response = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket_name, "Key": image_id},
                ExpiresIn=self._link_expiration,
            )
        except NoCredentialsError:
            raise ValueError("Credentials not available")

        return response
