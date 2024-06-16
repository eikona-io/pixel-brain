from pixelbrain.modules.upload import UploadToCloudinaryModule, UploadToS3Module
from pixelbrain.pipeline import Database, DataLoader
from tests.test_utils import DeleteDatabaseAfterTest
from pixelbrain.utils import PIXELBRAIN_PATH
import cloudinary.api
from uuid import uuid4
import boto3
from moto import mock_aws


def upload_to_cloudinary_run(image_paths, nof_results):
    database = Database()
    with DeleteDatabaseAfterTest(database):
        dataloader = DataLoader(image_paths, database, load_images=False)
        test_prefix = f"pixelbrain_tests/{uuid4()}"
        upload_to_cloudinary = UploadToCloudinaryModule(
            dataloader, database, test_prefix
        )
        upload_to_cloudinary.process()
        resources = cloudinary.api.resources(
            type="upload", prefix=test_prefix, max_results=10
        )
        # Delete the uploaded images after verification
        for resource in resources["resources"]:
            cloudinary.api.delete_resources([resource["public_id"]])
        assert (
            len(resources["resources"]) == nof_results
        ), f"There should be exactly {nof_results} images uploaded."


def test_upload_to_cloudinary():
    image_paths = [
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716945703/user_photos/nightly_202405290121/raw/10.png",
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1717317871/user_photos/663a083f-1ffe-476b-8cbf-1397a8b14547/raw/p0nxrm4c92q37wj3006x.jpg",
    ]
    upload_to_cloudinary_run(image_paths, 2)


def test_upload_to_cloudinary_large_file():
    image_paths = [
        f"{PIXELBRAIN_PATH}/src/tests/upload_modules/data/In-Car-1401.png"
    ]
    upload_to_cloudinary_run(image_paths, 0)


@mock_aws
def test_upload_to_s3():
    s3_bucket = "test-bucket"
    image_paths = [
        f"{PIXELBRAIN_PATH}/src/tests/upload_modules/data/In-Car-1401.png",
        f"{PIXELBRAIN_PATH}/src/tests/upload_modules/data/In-Car-1401.png",
    ]
    database = Database()
    with DeleteDatabaseAfterTest(database):
        dataloader = DataLoader(image_paths, database, load_images=False)
        test_prefix = f"pixelbrain_tests/{uuid4()}"
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=s3_bucket)
        upload_to_s3 = UploadToS3Module(dataloader, database, test_prefix, s3_bucket)
        upload_to_s3.process()
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=test_prefix)
        assert "Contents" in response, "No files were uploaded to S3."
        assert (
            len(response["Contents"]) == 2
        ), "There should be exactly 2 images uploaded to S3."
