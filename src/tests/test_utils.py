from cloudinary.api_client.execute_request import Response
from cloudinary.exceptions import AuthorizationRequired
import os
import shutil
from typing import Any
from pixelbrain.utils import get_logger


root_dir = os.path.dirname(os.path.realpath(__file__))

TEST_USER_ID = "test_user_id"
MOCK_RESPONSE_PATH = os.path.join(root_dir, "mock_data", "mock_cloudinary_response.pkl")
MOCK_HUE_DATA_PATH = os.path.join(root_dir, "mock_data", "mock_images", "hue_pipeline_images")
TEST_IMAGE_PATH = os.path.join(root_dir, "mock_data", "test_image.jpg")
MOCK_CLOUD_NAME = "test_cloud"
MOCK_API_KEY = "test_key"
MOCK_API_SECRET = "test_secret"
MOCK_IMAGE_ID = "123"
MOCK_PREFIX = "user_photos/mock_user_name"
MOCK_PUBLIC_ID = f"{MOCK_PREFIX}/{MOCK_IMAGE_ID}"
MOCK_IMAGE_URL = f"https://res.cloudinary.com/{MOCK_CLOUD_NAME}/image/upload/v1706189351/{MOCK_PREFIX}/{MOCK_IMAGE_ID}.jpg"

logger = get_logger(__name__)


class MockResponse:
     headers = {"x-featureratelimit-limit": '200', "x-featureratelimit-reset": '200', "x-featureratelimit-remaining": '200'}


class MockCloudinary:
    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir
        if temp_dir:
            logger.info(f"Using temp dir {temp_dir}")
    
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
        
    class uploader:
        temp_dir = None
        @staticmethod
        def upload(file: Any, **options: Any) -> Any:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found")
            if not options.get('public_id'):
                raise ValueError("public_id is required")
            if MockCloudinary.uploader.temp_dir:
                # Test emulating a file upload
                mock_remote_image_path_suffix = f"user_photos/{TEST_USER_ID}/processed/{os.path.splitext(os.path.basename(file))[0]}"
                mock_remote_image_path = os.path.join(MockCloudinary.uploader.temp_dir, mock_remote_image_path_suffix)
                os.makedirs(os.path.dirname(mock_remote_image_path), exist_ok=True)
                shutil.copy(file, mock_remote_image_path)
                logger.info(f"Uploaded image {file} to mock {mock_remote_image_path_suffix}")


class DeleteDatabaseAfterTest:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.delete_db()
        if exc_type:
            raise exc_val


def profile_callstack(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result
    return wrapper


def assert_env_var_present(env_var: str):
    return env_var in os.environ and os.environ[env_var] != ""
