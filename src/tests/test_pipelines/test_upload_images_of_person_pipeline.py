from pixelbrain.pipelines.upload_images_of_person_pipeline import (
    UploadImagesOfPersonPipeline,
)
from pixelbrain.database import Database
from tests.test_utils import TEST_USER_ID
import pytest
from moto import mock_aws
import boto3
from pixelbrain.utils import list_s3_objects


@mock_aws
def upload_images_of_person_pipelinerun_(person_image_path=None, nof_expected_files=4):
    local_temp_database = Database()
    s3 = boto3.client("s3", region_name="us-east-1")
    test_bucket_name = "test-bucket"
    s3.create_bucket(Bucket=test_bucket_name)

    upload_prefix = f"user_photos/{TEST_USER_ID}/processed"

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
        test_bucket_name,
        person_image_path=person_image_path,
        distance_threshold=distance_threshold,
        max_results=max_results,
        ssim_threshold=0.45,
    )
    pipeline.process()

    processed_files = list_s3_objects(test_bucket_name, upload_prefix)
    assert (
        len(processed_files) == nof_expected_files
    ), f"{nof_expected_files} images should be processed and uploaded (10 is a dup of 14 and last one is not the person)"


@pytest.mark.slow_suit
def test_upload_images_of_person_pipeline_with_person_image():
    upload_images_of_person_pipelinerun_(
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716945703/user_photos/nightly_202405290121/raw/9.png",
        4,
    )


@pytest.mark.slow_suit
def test_upload_images_of_person_pipeline_without_person_image():
    upload_images_of_person_pipelinerun_(nof_expected_files=5)
