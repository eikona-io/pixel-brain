from pixelbrain.pipeline import PipelineModule, DataLoader, DataLoaderFilter
import cloudinary.uploader
from typing import Union, Dict, List
from pixelbrain.database import Database
from uuid import uuid4

# INSERT_YOUR_REWRITE_HERE


class UploadModuleBase(PipelineModule):
    """
    Base class for uploading images from a dataloader to a remote storage.
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

    def _upload_image(self, image_path: str, remote_image_path: str):
        raise NotImplementedError("Subclasses should implement this method")

    def _process(self, image_ids: List[str], image_paths: List[str]):
        for image_path in image_paths:
            remote_image_path = f"{self._upload_prefix}/{uuid4().hex[:7]}"
            retry_count = 0
            while retry_count < self._max_retries:
                try:
                    self._upload_image(image_path, remote_image_path)
                    break
                except Exception as e:
                    if "File size too large" in str(e):
                        # Skip this file if it exceeds the size limit
                        print(f"Skipping file {image_path} due to file size too large")
                        break
                    retry_count += 1
                    if retry_count == self._max_retries:
                        raise e


class UploadToCloudinaryModule(UploadModuleBase):
    """
    Uploads images from a dataloader to Cloudinary.
    """

    def _upload_image(self, image_path: str, remote_image_path: str):
        cloudinary.uploader.upload(
            image_path,
            public_id=remote_image_path,
            unique_filename=False,
            overwrite=True,
            format="jpeg",
        )
        image_url = cloudinary.CloudinaryImage(remote_image_path).build_url()
        self._database.add_image(remote_image_path, image_url)


class UploadToS3Module(UploadModuleBase):
    """
    Uploads images from a dataloader to S3.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        database: Database,
        upload_prefix: str,
        s3_bucket: str,
        max_retries: int = 5,
        filters: Union[Dict[str, str], DataLoaderFilter] = None,
    ):
        super().__init__(dataloader, database, upload_prefix, max_retries, filters)
        self._s3_bucket = s3_bucket

    def _upload_image(self, image_path: str, remote_image_path: str):
        import boto3

        s3_client = boto3.client("s3")
        s3_client.upload_file(image_path, self._s3_bucket, remote_image_path)
        image_url = f"s3://{self._s3_bucket}/{remote_image_path}"
        self._database.add_image(remote_image_path, image_url)
