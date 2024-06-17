import boto3
import pytest
from moto import mock_aws
from pixelbrain.data_loaders.s3_dataloader import S3DataLoader
from unittest.mock import MagicMock
from pixelbrain.utils import PIXELBRAIN_PATH
import torch


@pytest.fixture
def s3_setup():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        # Load an actual image from the local filesystem
        with open(f"{PIXELBRAIN_PATH}/assets/test_data/00363_00.jpg", "rb") as image1:
            s3.put_object(Bucket="test-bucket", Key="images/image1.jpg", Body=image1.read())

        with open(f"{PIXELBRAIN_PATH}/assets/test_data/00364_00.jpg", "rb") as image2:
            s3.put_object(Bucket="test-bucket", Key="images/image2.jpg", Body=image2.read())

        yield s3


@pytest.fixture
def database():
    return MagicMock()


def test_s3_dataloader_init(s3_setup, database):
    loader = S3DataLoader(
        s3_paths_or_prefix="images/",
        bucket_name="test-bucket",
        database=database,
        batch_size=2,
    )
    assert loader._bucket_name == "test-bucket"
    assert loader._batch_size == 2


def test_s3_dataloader_get_all_image_paths(s3_setup, database):
    loader = S3DataLoader(
        s3_paths_or_prefix=["images/image1.jpg", "images/image2.jpg"],
        bucket_name="test-bucket",
        database=database,
    )
    image_paths = loader._get_all_image_paths()
    assert len(image_paths) == 2
    assert "images/image1.jpg" in image_paths
    assert "images/image2.jpg" in image_paths


def test_s3_dataloader_get_all_image_paths_with_prefix(s3_setup, database):
    loader = S3DataLoader(
        s3_paths_or_prefix="images/",
        bucket_name="test-bucket",
        database=database,
    )
    image_paths = loader._get_all_image_paths()
    assert len(image_paths) == 2
    assert "images/image1.jpg" in image_paths
    assert "images/image2.jpg" in image_paths


def test_s3_dataloader_load_image(s3_setup, database, tmpdir):
    loader = S3DataLoader(
        s3_paths_or_prefix="images/",
        bucket_name="test-bucket",
        database=database,
    )
    loader._tempdir = tmpdir
    image_path = loader._load_image("images/image1.jpg")
    isinstance(image_path, torch.Tensor)


def test_s3_dataloader_next(s3_setup, database):
    loader = S3DataLoader(
        s3_paths_or_prefix="images/",
        bucket_name="test-bucket",
        database=database,
        batch_size=1,
    )
    ids_batch, image_batch = next(loader)
    assert len(ids_batch) == 1
    assert len(image_batch) == 1
    assert isinstance(image_batch[0], torch.Tensor)
    assert isinstance(ids_batch[0], str)


def test_s3_dataloader_next_batch_size_2(s3_setup, database):
    loader = S3DataLoader(
        s3_paths_or_prefix="images/",
        bucket_name="test-bucket",
        database=database,
        batch_size=2,
    )
    ids_batch, image_batch = next(loader)
    assert len(ids_batch) == 2
    assert len(image_batch) == 2
    assert isinstance(image_batch[0], torch.Tensor)
    assert isinstance(image_batch[1], torch.Tensor)
    assert isinstance(ids_batch[0], str)
    assert isinstance(ids_batch[1], str)



