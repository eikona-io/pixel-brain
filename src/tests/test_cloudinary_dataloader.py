from unittest.mock import patch, MagicMock
import torch
from torchvision.io.image import read_file, decode_image, ImageReadMode
import pytest
import cloudinary
import requests
import pickle
from cloudinary.api_client.execute_request import Response
from cloudinary.exceptions import AuthorizationRequired
from pixelbrain.data_loaders.cloudinary_dataloader import CloudinaryDataLoader
import os


root_dir = os.path.dirname(os.path.realpath(__file__))
MOCK_RESPONSE_PATH = os.path.join(root_dir, "mock_data", "mock_cloudinary_response.pkl")
TEST_IMAGE_PATH = os.path.join(root_dir, "mock_data", "test_image.jpg")
MOCK_CLOUD_NAME = "test_cloud"
MOCK_API_KEY = "test_key"
MOCK_API_SECRET = "test_secret"
MOCK_IMAGE_ID = "123"
MOCK_PREFIX = "user_photos/mock_user_name"
MOCK_PUBLIC_ID = f"{MOCK_PREFIX}/{MOCK_IMAGE_ID}"
MOCK_IMAGE_URL = f"https://res.cloudinary.com/{MOCK_CLOUD_NAME}/image/upload/v1706189351/{MOCK_PREFIX}/{MOCK_IMAGE_ID}.jpg"


class MockResponse:
     headers = {"x-featureratelimit-limit": '200', "x-featureratelimit-reset": '200', "x-featureratelimit-remaining": '200'}


class MockCloudinary:
    class api:
        @staticmethod
        def ping():
            cloudinary_url = os.environ.get('CLOUDINARY_URL')
            if not cloudinary_url:
                raise AuthorizationRequired("Cloudinary credentials not found. Please set them as environment variables.")
            cloud_name = cloudinary_url.split('@')[1].split('.')[0]
            api_key = cloudinary_url.split('//')[1].split(':')[0]
            api_secret = cloudinary_url.split(':')[2].split('@')[0]
            if cloud_name != MOCK_CLOUD_NAME or api_key != MOCK_API_KEY or api_secret != MOCK_API_SECRET:
                raise AuthorizationRequired("Invalid credentials")
            return Response({'status': 'ok'}, MockResponse())
        def resources(type='upload', prefix='', max_results=500):
            if prefix == MOCK_PREFIX:
                return {'resources': [{'secure_url': MOCK_IMAGE_URL, 'public_id': MOCK_PUBLIC_ID}]}
            else:
                return {'resources': []}
        
    class utils:
        @staticmethod
        def cloudinary_url(public_id, **combined_options):
            return [MOCK_IMAGE_URL]


class MockRequests:
    @staticmethod
    def get(url):
        if url == MOCK_IMAGE_URL:
            with open(MOCK_RESPONSE_PATH, 'rb') as f:
                response = pickle.load(f)
            return response
        else:
            return requests.Response()


cloudinary.api = MockCloudinary.api
cloudinary.utils = MockCloudinary.utils
requests.get = MockRequests.get


class TestCloudinaryDataLoader:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.database = MagicMock()
        self.cloudinary_folder_prefix = MOCK_PREFIX
        self.wrong_cloudinary_folder_prefix = "mock_wrong_prefix"
        self.cloudinary_cloud_name = MOCK_CLOUD_NAME
        self.cloudinary_api_key = MOCK_API_KEY
        self.cloudinary_api_secret = MOCK_API_SECRET
        self.wrong_cloudinary_api_secret = "mock_wrong_secret"
        self.test_image = decode_image(read_file(TEST_IMAGE_PATH), ImageReadMode.UNCHANGED)
        os.environ['CLOUDINARY_URL'] = f"cloudinary://{self.cloudinary_api_key}:{self.cloudinary_api_secret}@{self.cloudinary_cloud_name}"

    def get_loader(self):
        return CloudinaryDataLoader(self.cloudinary_folder_prefix, self.database)

    def test_init(self):
        good_loader = self.get_loader()
        assert good_loader._images_path == self.cloudinary_folder_prefix
        with pytest.raises(AuthorizationRequired):
            os.environ['CLOUDINARY_URL'] = f"cloudinary://{self.cloudinary_api_key}:{self.wrong_cloudinary_api_secret}@{self.cloudinary_cloud_name}"
            _ = CloudinaryDataLoader(self.cloudinary_folder_prefix, self.database)

    def test_next(self):
        loader = self.get_loader()
        ids_batch, image_batch = loader.__next__()
        assert ids_batch == [MOCK_IMAGE_ID]
        assert torch.equal(image_batch[0], self.test_image)
        torch.allclose

    def test_load_image(self):
        loader = self.get_loader()
        assert torch.equal(self.test_image, loader._load_image(MOCK_IMAGE_URL))        

    def test_get_all_image_paths(self):
        good_loader = self.get_loader()
        image_urls = good_loader._get_all_image_paths()
        assert image_urls == [MOCK_PUBLIC_ID]
        bad_loader = CloudinaryDataLoader(self.wrong_cloudinary_folder_prefix, self.database)
        image_urls = bad_loader._get_all_image_paths()
        assert image_urls == []