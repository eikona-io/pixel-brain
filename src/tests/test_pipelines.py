from pixelbrain.pipelines.identity_tagging_pipeline import IdentityTaggingPipeline
from pixelbrain.database import Database
import pytest


def identity_tagging_pipeline_run(identifying_strategy, apply_people_detector=True):
    database = Database(database_id="fashion_tagging_pipe_subset")
    pipe = IdentityTaggingPipeline('assets/test_data', 
                                   database,
                                   identity_field_name=f'{identifying_strategy}_identity',
                                   apply_people_detector=apply_people_detector,
                                   identifying_strategy=identifying_strategy)

    pipe.process()
    images_with_identity = database.find_images_with_value(f"{identifying_strategy}_identity")
    assert len(images_with_identity) > 0, "should assign identities to some images"

@pytest.mark.slow_suit
def test_identity_tagging_pipeline_no_people_detector():
    identity_tagging_pipeline_run('pairwise', False)

@pytest.mark.skip(reason="gpt4v rate limits..")
def test_identity_tagging_pipeline():
    identity_tagging_pipeline_run('pairwise')