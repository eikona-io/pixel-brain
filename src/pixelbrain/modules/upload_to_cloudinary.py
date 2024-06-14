from pixelbrain.pipeline import PipelineModule, DataLoader, DataLoaderFilter
import cloudinary.uploader
from typing import Union, Dict, List
from pixelbrain.database import Database
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
        max_retries: int = 5,
        filters: Union[Dict[str, str], DataLoaderFilter] = None,
    ):

        super().__init__(
            data=dataloader,
            database=database,
            filters=filters,
        )
        self._upload_prefix = upload_prefix
        self._max_retries = max_retries

    def _process(self, image_ids: List[str], image_paths: List[str]):
        for image_path in image_paths:
            remote_image_path = f"{self._upload_prefix}/{uuid4().hex[:7]}"
            retry_count = 0
            while retry_count < self._max_retries:
                try:
                    cloudinary.uploader.upload(
                        image_path,
                        public_id=remote_image_path,
                        unique_filename=False,
                        overwrite=True,
                        format="jpeg",
                    )
                    image_url = cloudinary.CloudinaryImage(
                        remote_image_path
                    ).build_url()
                    self._database.add_image(remote_image_path, image_url)
                    break
                except Exception as e:
                    if "File size too large" in str(e):
                        # cloudinary has a 20MB limit
                        # skip this file
                        print(f"Skipping file {image_path} due to file size too large")
                        break
                    retry_count += 1
                    if retry_count == self._max_retries:
                        raise e
