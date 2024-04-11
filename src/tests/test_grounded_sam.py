from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
import pytest
from pixelbrain.utils import PIXELBRAIN_PATH


@pytest.mark.skip(reason="We don't want to install SAM on the CI at the moment")
@pytest.mark.slow_suit
def test_grounded_sam_detector():
    from pixelbrain.modules.grounded_sam_detector import GroundedSAMDetectorModule
    database = Database(database_id="grounded_sam")
    data = DataLoader(f"{PIXELBRAIN_PATH}/assets/test_data/", database)
    module = GroundedSAMDetectorModule(data, database, "human", "is_person")
    module.process()
    
    metadata = database.get_all_images()
    for meta in metadata:
        assert "is_person" in meta
    database.delete_db()