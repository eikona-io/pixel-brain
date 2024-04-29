from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.embedders import FacenetEmbbedderModule
from pixelbrain.modules.people_identifier import PeopleIdentifierModule, MostCommonIdentityFilter
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
        if strategy == 'pairwise':
            algo_kwargs = {distance_threshold:290, exclude_group:None}
        else:
            algo_kwargs = {}
        
        people_identifier = PeopleIdentifierModule(
            data2, database, "face_embedding", strategy=strategy, **algo_kwargs
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

def test_most_common_identity_filter():
    # Setup a mock database with a few images having different identities
    mock_db = Database(database_id="test_db")
    with DeleteDatabaseAfterTest(mock_db):
        mock_db.add_image("img1", "path/to/img1")
        mock_db.store_field("img1", "identity", "personA")
        mock_db.add_image("img2", "path/to/img2")
        mock_db.store_field("img2", "identity", "personA")
        mock_db.add_image("img3", "path/to/img3")
        mock_db.store_field("img3", "identity", "personB")

        # Initialize the MostCommonIdentityFilter
        identity_filter = MostCommonIdentityFilter("identity")

        # Test the filter
        image_ids = ["img1", "img2", "img3"]
        filtered_ids = identity_filter.filter(mock_db, image_ids)

        # Assert that only images with the most common identity are returned
        assert len(filtered_ids) == 2, "Filter should return two images"
        assert "img1" in filtered_ids, "img1 should be in the filtered list"
        assert "img2" in filtered_ids, "img2 should be in the filtered list"
        assert "img3" not in filtered_ids, "img3 should not be in the filtered list"