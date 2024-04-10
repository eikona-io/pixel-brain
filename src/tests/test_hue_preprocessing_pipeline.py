from pixelbrain.pipelines.hue_processing_pipeline import HueProcessingPipeline
from os.path import join
from os import listdir
import tempfile
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipelines.hue_processing_pipeline import HueProcessingPipeline
from tests.test_utils import TEST_USER_ID, MockCloudinary, MOCK_HUE_DATA_PATH
from uuid import uuid4
import cloudinary.uploader
import pytest


@pytest.mark.slow_suit
class TestHuePreprocessingPipeline:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.old_upload = cloudinary.uploader.upload
        self.local_temp_database = Database(database_id=uuid4().hex)
        yield
        # teardown code
        cloudinary.uploader.upload = self.old_upload
        self.local_temp_database.delete_db()

    def test_hue_preprocessing_pipeline(self):
        with tempfile.TemporaryDirectory() as tempdir:
            MockCloudinary.uploader.temp_dir = tempdir
            cloudinary.uploader.upload = MockCloudinary.uploader.upload
            dataloader = DataLoader(MOCK_HUE_DATA_PATH, self.local_temp_database)
            h = HueProcessingPipeline(join(tempdir, 'hue_pipeline'), dataloader, TEST_USER_ID)
            h.process()
            
            processed_photos_dir = join(tempdir, 'user_photos', TEST_USER_ID, 'processed')
            processed_files = [file for file in listdir(processed_photos_dir) if not file.startswith('.')]
            assert set(processed_files) == {'blond1_face0', 'blond2_face0', 'blond3_face0'}  # The main identity is the blond with 3 images
