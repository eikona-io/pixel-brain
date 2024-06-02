from pixelbrain.pipelines.upload_images_of_person_pipeline import (
    UploadImagesOfPersonPipeline,
)
from os.path import join
from os import listdir, makedirs
import tempfile
from pixelbrain.database import Database
from tests.test_utils import TEST_USER_ID, MockCloudinary
import cloudinary.uploader
import pytest
from unittest.mock import patch


def upload_images_of_person_pipelinerun_(person_image_path=None):
    local_temp_database = Database()
    with patch.object(cloudinary.uploader, "upload", MockCloudinary.uploader.upload):
        with tempfile.TemporaryDirectory() as tempdir:
            MockCloudinary.uploader.temp_dir = tempdir
            upload_prefix = f"user_photos/{TEST_USER_ID}/processed"
            processed_photos_dir = join(tempdir, upload_prefix)
            makedirs(processed_photos_dir, exist_ok=True)

            raw_images_path_or_paths = [
                "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716945703/user_photos/nightly_202405290121/raw/10.png",
                "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716945703/user_photos/nightly_202405290121/raw/11.png",
                "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716945703/user_photos/nightly_202405290121/raw/12.png",
                "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716945703/user_photos/nightly_202405290121/raw/13.png",
                "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716945703/user_photos/nightly_202405290121/raw/14.png",
                "https://res.cloudinary.com/dxgcobmaz/image/upload/v1717055415/user_photos/raw/2514506368613414.jpg",
            ]
            distance_threshold = 0.6
            max_results = 30

            pipeline = UploadImagesOfPersonPipeline(
                raw_images_path_or_paths,
                local_temp_database,
                upload_prefix,
                person_image_path=person_image_path,
                distance_threshold=distance_threshold,
                max_results=max_results,
                ssim_threshold=0.45,
            )
            pipeline.process()

            processed_files = [
                file
                for file in listdir(processed_photos_dir)
                if not file.startswith(".")
            ]
            assert (
                len(processed_files) == 4 if person_image_path else 6
            ), "4 or 6 images should be processed and uploaded (10 is a dup of 14 and last one is not the person)"


@pytest.mark.slow_suit
def test_upload_images_of_person_pipeline_with_person_image():
    upload_images_of_person_pipelinerun_(
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716945703/user_photos/nightly_202405290121/raw/9.png"
    )


@pytest.mark.slow_suit
def test_upload_images_of_person_pipeline_without_person_image():
    upload_images_of_person_pipelinerun_()
