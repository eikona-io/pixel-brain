from unittest.mock import MagicMock
import torch
from torchvision.io.image import read_file, decode_image, ImageReadMode
import pytest
import cloudinary
import requests
import pickle
from cloudinary.exceptions import AuthorizationRequired
from pixelbrain.data_loaders.cloudinary_dataloader import CloudinaryDataLoader
import os
from tests.test_utils import (
    MockCloudinary,
    MOCK_IMAGE_URL,
    MOCK_RESPONSE_PATH,
    MOCK_PREFIX,
    MOCK_CLOUD_NAME,
    MOCK_API_KEY,
    MOCK_API_SECRET,
    MOCK_IMAGE_ID,
    TEST_IMAGE_PATH,
    MOCK_PUBLIC_ID,
)


class MockRequests:
    @staticmethod
    def get(url):
        if url == MOCK_IMAGE_URL:
            with open(MOCK_RESPONSE_PATH, "rb") as f:
                response = pickle.load(f)
            return response
        else:
            return requests.Response()


class TestCloudinaryDataLoader:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Setup code
        self.database = MagicMock()
        self.cloudinary_folder_prefix = MOCK_PREFIX
        self.wrong_cloudinary_folder_prefix = "mock_wrong_prefix"
        self.cloudinary_cloud_name = MOCK_CLOUD_NAME
        self.cloudinary_api_key = MOCK_API_KEY
        self.cloudinary_api_secret = MOCK_API_SECRET
        self.wrong_cloudinary_api_secret = "mock_wrong_secret"
        self.test_image = decode_image(
            read_file(TEST_IMAGE_PATH), ImageReadMode.UNCHANGED
        )
        self.original_cloudinary_url = os.environ.get("CLOUDINARY_URL", "")
        os.environ["CLOUDINARY_URL"] = (
            f"cloudinary://{self.cloudinary_api_key}:{self.cloudinary_api_secret}@{self.cloudinary_cloud_name}"
        )

        self.old_api = cloudinary.api
        self.old_utils = cloudinary.utils
        self.old_requests_get = requests.get
        cloudinary.api = MockCloudinary.api
        cloudinary.utils = MockCloudinary.utils
        requests.get = MockRequests.get
        yield
        # teardown code
        os.environ["CLOUDINARY_URL"] = self.original_cloudinary_url
        cloudinary.api = self.old_api
        cloudinary.utils = self.old_utils
        requests.get = self.old_requests_get

    def get_loader(self):
        return CloudinaryDataLoader(self.cloudinary_folder_prefix, self.database)

    def test_init(self):
        good_loader = self.get_loader()
        assert good_loader._images_path == self.cloudinary_folder_prefix
        with pytest.raises(AuthorizationRequired):
            os.environ["CLOUDINARY_URL"] = (
                f"cloudinary://{self.cloudinary_api_key}:{self.wrong_cloudinary_api_secret}@{self.cloudinary_cloud_name}"
            )
            _ = CloudinaryDataLoader(self.cloudinary_folder_prefix, self.database)

    def test_next(self):
        loader = self.get_loader()
        ids_batch, image_batch = loader.__next__()
        assert ids_batch == [MOCK_PUBLIC_ID]
        assert torch.equal(image_batch[0], self.test_image)
        torch.allclose

    def test_load_image(self):
        loader = self.get_loader()
        assert torch.equal(self.test_image, loader._load_image(MOCK_IMAGE_URL))

    def test_get_all_image_paths(self):
        good_loader = self.get_loader()
        image_urls = good_loader._get_all_image_paths()
        assert image_urls == [MOCK_PUBLIC_ID]
        bad_loader = CloudinaryDataLoader(
            self.wrong_cloudinary_folder_prefix, self.database
        )
        image_urls = bad_loader._get_all_image_paths()
        assert image_urls == []
