from pixelbrain.pipeline import PipelineModule, DataLoader, DataLoaderFilter
import cloudinary.uploader
from typing import Union, Dict, List
from pixelbrain.database import Database
import os
from uuid import uuid4


class UploadToCloudinaryModule(PipelineModule):
    """
    Uploads images from a dataloader to cloudinary.
    Can upload all images or by a given field.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        database: Database,
        upload_prefix: str,
        filters: Union[Dict[str, str], DataLoaderFilter] = None,
    ):

        super().__init__(
            data=dataloader,
            database=database,
            filters=filters,
        )
        self._upload_prefix = upload_prefix
        self._image_idx = 0

    def _process(self, image_ids: List[str], image_paths: List[str]):
        for image_path in image_paths:
            remote_image_path = f"{self._upload_prefix}/{self._image_idx}"
            self._image_idx += 1
            cloudinary.uploader.upload(
                image_path,
                public_id=remote_image_path,
                unique_filename=False,
                overwrite=True,
            )
            image_url = cloudinary.CloudinaryImage(remote_image_path).build_url()
            self._database.add_image(remote_image_path, image_url)
