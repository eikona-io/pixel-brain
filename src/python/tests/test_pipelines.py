from pixel_brain.pipelines.fashion_tagging_pipeline import FashionTaggingPipeline
from pixel_brain.database import Database
import pytest


@pytest.mark.skip(reason="gpt4v rate limits..")
def test_fashion_tagging_pipeline():
    images_path = 'assets/test_data'

    database = Database(database_id="fashion_tagging_pipe_test")
    pipe = FashionTaggingPipeline(images_path, database)

    pipe.process()
    pass