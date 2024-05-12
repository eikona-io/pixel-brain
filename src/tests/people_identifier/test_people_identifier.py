from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.embedders import FacenetEmbbedderModule
from pixelbrain.modules.people_identifier import PeopleIdentifierModule, MostCommonIdentityFilter
import re
import os
from glob import glob
import pandas as pd
import numpy as np
import pytest
from pixelbrain.utils import PIXELBRAIN_PATH
from tests.test_utils import DeleteDatabaseAfterTest, remove_number_suffix
from sklearn.metrics import confusion_matrix, f1_score


def get_identity_from_path(path):
    match = re.search(r"/(\d+)_\d+\.jpe?g", path)
    identity = match.group(1)
    return identity


def people_identifier_module_run(strategy, data_path):
    database = Database(
        database_id=f"people_identifier_test_{strategy}_{os.path.basename(data_path)}"
    )
    data = DataLoader(data_path, database)
    data2 = data.clone()
    subject_name = os.path.basename(data_path).split('.')[0].lower()

    with DeleteDatabaseAfterTest(database):
        # Create an instance of FacenetEmbbedderModule and process the data
        facenet_embedder = FacenetEmbbedderModule(data, database)
        facenet_embedder.process()

        # Create an instance of PeopleIdentifierModule process the data
        people_identifier = PeopleIdentifierModule(
            data2, database, "face_embedding", strategy=strategy
        )
        people_identifier.process()

        # Retrieve all images from the database
        metadata = database.get_all_images()
        # Create a nice df
        df = pd.DataFrame(metadata)
        df['_id'] = df['_id'].apply(lambda id: remove_number_suffix(os.path.basename(id).split('.')[0]))
        df = df.drop(columns=['image_path', 'face_embedding'])
        df.rename(columns={'_id': 'true_id', 'identity': 'assigned_id'}, inplace=True)
        df['assigned_id'] = df['assigned_id'].fillna(-1)
        df['assigned_id'] = df['assigned_id'].map({id: idx for idx, id in enumerate(df['assigned_id'].unique()) if id != -1})
        df['assigned_id'] = df['assigned_id'].fillna(-1)
        df['assigned_id'] = df['assigned_id'].apply(lambda x: int(x))
        
        # Compute metrics
        most_common_assigned_id_for_target = df[df['true_id'] == subject_name]['assigned_id'].value_counts().idxmax()
        labels = (df['true_id'] == subject_name).to_numpy()
        predictions = (df['assigned_id'] == most_common_assigned_id_for_target).to_numpy()
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        f1 = f1_score(labels, predictions)
        return f1


@pytest.mark.slow_suit
def test_people_identifier():
    """
    Computes ROC scores for 5 different people with data after face extraction.
    We measure that the ...
    """
    test_data_dir = os.path.join(PIXELBRAIN_PATH, "assets", "tuning_data", "people_identifier", "real_data", "after_face_extractor")
    test_subject_dirs = glob(os.path.join(test_data_dir, '*'))
    f1_scores = []
    for subject_dir in test_subject_dirs:
        f1 = people_identifier_module_run("dbscan", subject_dir)
        f1_scores.append(f1)
    assert np.array(f1_scores).mean() >= 0.95


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