from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.grounded_sam_detector import GroundedSAMDetectorModule
import pytest


@pytest.mark.slow_suit
def test_grounded_sam_detector():
    database = Database(database_id="grounded_sam")
    # test only one image to save cost
    data = DataLoader("assets/test_data/", database)
    module = GroundedSAMDetectorModule(data, database, "human", "is_person")
    module.process()
    
    metadata = database.get_all_images()
    for meta in metadata:
        assert "is_person" in meta
    database.delete_db()

test_grounded_sam_detector()