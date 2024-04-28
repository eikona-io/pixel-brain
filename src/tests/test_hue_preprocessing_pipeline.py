from pixelbrain.pipelines.hue_processing_pipeline import HueProcessingPipeline
from os.path import join
from os import listdir, makedirs
import tempfile
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipelines.hue_processing_pipeline import HueProcessingPipeline
from tests.test_utils import TEST_USER_ID, MockCloudinary, MOCK_HUE_DATA_PATH
from uuid import uuid4
import cloudinary.uploader
import pytest
from unittest.mock import patch


@pytest.mark.slow_suit
def test_hue_preprocessing_pipeline():
    local_temp_database = Database(database_id=uuid4().hex)
    with patch.object(cloudinary.uploader, "upload", MockCloudinary.uploader.upload):
        with tempfile.TemporaryDirectory() as tempdir:
            MockCloudinary.uploader.temp_dir = tempdir
            dataloader = DataLoader(MOCK_HUE_DATA_PATH, local_temp_database)
            local_results_dir = join(tempdir, "hue_pipeline")
            upload_prefix = f"user_photos/{TEST_USER_ID}/processed"
            processed_photos_dir = join(
                tempdir, upload_prefix
            )
            makedirs(processed_photos_dir, exist_ok=True)
            h = HueProcessingPipeline(local_results_dir, dataloader, upload_prefix)
            h.process()

            processed_files = [
                file
                for file in listdir(processed_photos_dir)
                if not file.startswith(".")
            ]
            assert set(processed_files) == {
                "blond1_face0",
                "blond2_face0",
                "blond3_face0",
            }  # The main identity is the blond with 3 images


test_hue_preprocessing_pipeline()
