from pixelbrain.pipelines.hue_processing_pipeline import HueProcessingPipeline
from os.path import join
from os import listdir, makedirs
import tempfile
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipelines.hue_processing_pipeline import HueProcessingPipeline
from tests.test_utils import TEST_USER_ID, MockCloudinary, MOCK_HUE_DATA_PATH
from uuid import uuid4
import pytest
from unittest.mock import patch
from pixelbrain.utils import (
    get_cloudinary_images_from_prefix,
    delete_image_from_cloudinary,
)


@pytest.mark.slow_suit
def test_hue_preprocessing_pipeline():
    local_temp_database = Database(database_id=uuid4().hex)
    temp_user_id = str(uuid4())
    with tempfile.TemporaryDirectory() as tempdir:
        dataloader = DataLoader(MOCK_HUE_DATA_PATH, local_temp_database)
        local_results_dir = join(tempdir, "hue_pipeline")
        augmented_results_dir = join(tempdir, "augmented_hue_pipeline")
        upload_prefix = f"user_photos/{temp_user_id}/processed"
        h = HueProcessingPipeline(
            local_results_dir,
            augmented_results_dir,
            dataloader,
            local_temp_database,
            upload_prefix,
            min_nof_processed_images=5,
        )
        h.process()

        processed_results = get_cloudinary_images_from_prefix(upload_prefix)
        assert len(processed_results) == 5
        for result in processed_results:
            delete_image_from_cloudinary(result)
