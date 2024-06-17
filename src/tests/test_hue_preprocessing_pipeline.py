from os.path import join
import tempfile
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipelines.hue_processing_pipeline import HueProcessingPipeline
from tests.test_utils import MOCK_HUE_DATA_PATH
from uuid import uuid4
import pytest
from pixelbrain.utils import (
    get_s3_files_from_prefix,
    delete_s3_file,
)
from moto import mock_aws
import boto3


@pytest.mark.slow_suit
@mock_aws
def test_hue_preprocessing_pipeline():
    local_temp_database = Database(database_id=uuid4().hex)
    temp_user_id = str(uuid4())
    with tempfile.TemporaryDirectory() as tempdir:
        dataloader = DataLoader(MOCK_HUE_DATA_PATH, local_temp_database)
        local_results_dir = join(tempdir, "hue_pipeline")
        augmented_results_dir = join(tempdir, "augmented_hue_pipeline")
        upload_prefix = f"user_photos/{temp_user_id}/processed"
        mock_s3_bucket = "test_bucket"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=mock_s3_bucket)

        h = HueProcessingPipeline(
            local_results_dir,
            augmented_results_dir,
            mock_s3_bucket,
            dataloader,
            local_temp_database,
            upload_prefix,
            min_nof_processed_images=5,
        )
        h.process()

        processed_results = get_s3_files_from_prefix(mock_s3_bucket, upload_prefix)
        assert len(processed_results) == 5
        for result in processed_results:
            delete_s3_file(mock_s3_bucket, result)
