from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
from pixel_brain.modules.embedders import FacenetEmbbedderModule
from pixel_brain.modules.people_identifier import PeopleIdentifierModule
import re


def get_identity_from_path(path):
    match = re.search(r'/(\d+)_\d+.jpeg', path)
    identity = match.group(1)
    return identity


def people_identifier_module_run(strategy):
    database = Database(database_id=f"people_identifier_test_{strategy}")
    data = DataLoader("assets/test_data/subjects", database)
    data2 = data.clone()

    # Create an instance of FacenetEmbbedderModule and process the data
    facenet_embedder = FacenetEmbbedderModule(data, database)
    facenet_embedder.process()

    # Create an instance of PeopleIdentifierModule with pairwise strategy and process the data
    people_identifier = PeopleIdentifierModule(data2, database, "face_embedding", strategy=strategy)
    people_identifier.process()

    # Retrieve all images from the database
    metadata = database.get_all_images()

    # Check if all images have been assigned an identity
    for image_meta in metadata:
        assigned_identity = image_meta.get("identity", None)
        if assigned_identity is not None:
            same_identity_images = database.find_images_with_value("identity", assigned_identity)
            same_identity_image_paths = [meta['image_path'] for meta in same_identity_images]
            orig_identities = [get_identity_from_path(path) for path in same_identity_image_paths]
            if strategy == "pairwise":
                assert len(set(orig_identities)) == 1, "Not all original identities are the same"
            else:
                # hdbscan is less strict
                assert len(set(orig_identities)) <= 2, "Not all original identities are the same"
    database.delete_db()


def test_people_identifier_module_pairwise_strategy():
    people_identifier_module_run('pairwise')


def test_people_identifier_module_hdbscan_strategy():
    people_identifier_module_run('hdbscan')