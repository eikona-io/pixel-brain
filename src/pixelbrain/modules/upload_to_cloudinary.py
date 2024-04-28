from pixelbrain.pipeline import PipelineModule, DataLoader, DataLoaderFilter
import cloudinary.uploader
from typing import Union, Dict, List
from pixelbrain.database import Database
from overrides import overrides
import os


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

    def _process(self, image_ids: List[str], image_paths: List[str]):
        for image_id, image_path in zip(image_ids, image_paths):
            remote_image_path = f"{self._upload_prefix}/{os.path.splitext(os.path.basename(image_id))[0]}"
            cloudinary.uploader.upload(
                image_path,
                public_id=remote_image_path,
                unique_filename=False,
                overwrite=True,
            )
            image_url = cloudinary.CloudinaryImage(remote_image_path).build_url()
            self._database.add_image(remote_image_path, image_url)
