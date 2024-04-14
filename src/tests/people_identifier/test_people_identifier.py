from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.embedders import FacenetEmbbedderModule
from pixelbrain.modules.people_identifier import PeopleIdentifierModule
import re
import pytest
from pixelbrain.utils import PIXELBRAIN_PATH
from tests.test_utils import DeleteDatabaseAfterTest


def get_identity_from_path(path):
    match = re.search(r"/(\d+)_\d+\.jpe?g", path)
    identity = match.group(1)
    return identity


def people_identifier_module_run(strategy, data_path, nof_images_per_subject):
    database = Database(
        database_id=f"people_identifier_test_{strategy}_{nof_images_per_subject}"
    )
    data = DataLoader(data_path, database)
    data2 = data.clone()

    with DeleteDatabaseAfterTest(database):
        # Create an instance of FacenetEmbbedderModule and process the data
        facenet_embedder = FacenetEmbbedderModule(data, database)
        facenet_embedder.process()

        # Create an instance of PeopleIdentifierModule with pairwise strategy and process the data
        people_identifier = PeopleIdentifierModule(
            data2, database, "face_embedding", strategy=strategy
        )
        people_identifier.process()

        # Retrieve all images from the database
        metadata = database.get_all_images()

        # Check if all images have been assigned an identity
        for image_meta in metadata:
            assigned_identity = image_meta.get("identity", None)
            if assigned_identity is not None:
                same_identity_images = database.find_images_with_value(
                    "identity", assigned_identity
                )
                same_identity_image_paths = [
                    meta["image_path"] for meta in same_identity_images
                ]
                orig_identities = [
                    get_identity_from_path(path) for path in same_identity_image_paths
                ]
                assert (
                    len(set(orig_identities)) == 1
                ), "Not all original identities are the same"
                if strategy == "hdbscan":
                    # hdbscan should find all subject photos
                    assert (
                        len(same_identity_images) == nof_images_per_subject
                    ), "Not all subject photos were found (found {}, expected {})".format(
                        len(same_identity_images), nof_images_per_subject
                    )


@pytest.mark.slow_suit
def test_people_identifier_module_pairwise_strategy():
    people_identifier_module_run(
        "pairwise",
        f"{PIXELBRAIN_PATH}/assets/test_data/subjects",
        nof_images_per_subject=3,
    )


@pytest.mark.slow_suit
def test_people_identifier_module_hdbscan_strategy_multiple_people():
    people_identifier_module_run(
        "hdbscan",
        f"{PIXELBRAIN_PATH}/assets/test_data/subjects",
        nof_images_per_subject=3,
    )


@pytest.mark.slow_suit
def test_people_identifier_module_hdbscan_strategy_one_person():
    people_identifier_module_run(
        "hdbscan",
        f"{PIXELBRAIN_PATH}/src/tests/people_identifier/test_identity_1",
        nof_images_per_subject=6,  # 8 images for subject 1, in 2 we can't find a face
    )
    people_identifier_module_run(
        "hdbscan",
        f"{PIXELBRAIN_PATH}/src/tests/people_identifier/test_identity_2",
        nof_images_per_subject=12,  # 14 images for subject 2, in 2 we can't find a face
    )
