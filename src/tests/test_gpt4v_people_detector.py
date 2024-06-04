from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.gpt4v import (
    GPT4VPeopleDetectorModule,
    GPT4VPerfectEyesModule,
    GPT4VNoGenerationArtifactsModule,
)
from pixelbrain.utils import PIXELBRAIN_PATH
import pytest


def test_gpt4v_people_detector():
    database = Database()
    # test only one image to save cost
    data = DataLoader(
        [
            f"{PIXELBRAIN_PATH}/assets/test_data/00375_00.jpg",
            f"{PIXELBRAIN_PATH}/assets/test_data/00377_00.jpg",
        ],
        database,
        decode_images=False,
        batch_size=2,
    )
    module = GPT4VPeopleDetectorModule(data, database)
    module.process()

    metadata = database.get_all_images()
    for meta in metadata:
        assert "is_person" in meta
    database.delete_db()


def test_gpt4v_perfect_eyes():
    database = Database()
    # test only one image to save cost
    data = DataLoader(
        f"{PIXELBRAIN_PATH}/src/tests/test_gpt4v_data/",
        database,
        decode_images=False,
        batch_size=3,
    )
    module = GPT4VPerfectEyesModule(
        data, database, metadata_field_name="has_perfect_eyes"
    )
    module.process()

    metadata = database.get_all_images()
    for meta in metadata:
        assert "has_perfect_eyes" in meta
    database.delete_db()


def test_gpt4v_no_generation_artifacts():
    database = Database()
    # test only one image to save cost
    data = DataLoader(
        f"{PIXELBRAIN_PATH}/src/tests/test_gpt4v_data/",
        database,
        decode_images=False,
        batch_size=3,
    )
    module = GPT4VNoGenerationArtifactsModule(
        data, database, metadata_field_name="has_no_generation_artifacts"
    )
    module.process()

    metadata = database.get_all_images()
    for meta in metadata:
        assert "has_no_generation_artifacts" in meta
    database.delete_db()


@pytest.mark.asyncio
async def test_gpt4v_no_generation_artifacts_async():
    database = Database()
    # test only one image to save cost
    data = DataLoader(
        f"{PIXELBRAIN_PATH}/src/tests/test_gpt4v_data/",
        database,
        decode_images=False,
        batch_size=3,
    )
    module = GPT4VNoGenerationArtifactsModule(
        data, database, metadata_field_name="has_no_generation_artifacts"
    )
    await module.async_process()

    metadata = database.get_all_images()
    for meta in metadata:
        assert "has_no_generation_artifacts" in meta
    database.delete_db()
