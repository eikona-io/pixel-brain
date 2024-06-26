from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.face_extractor import FaceExtractorModule
import tempfile
import os 
from pixelbrain.utils import PIXELBRAIN_PATH
import pytest
from tests.test_utils import DeleteDatabaseAfterTest
from uuid import uuid4


@pytest.mark.slow_suit
def test_face_extractor_module():
    database = Database(database_id=uuid4().hex)
    with DeleteDatabaseAfterTest(database):
        data = DataLoader(f"{PIXELBRAIN_PATH}/assets/test_data/", database)
        
        with tempfile.TemporaryDirectory() as tempdir:
            module = FaceExtractorModule(data, database, image_save_path=tempdir)
            module.process()
            num_images = len(os.listdir(tempdir))
            assert num_images == 31 + 2 * 3, "there should be 31 faces from portrait images and 2 * 3 faces from two triplete images"

def test_face_extractor_module_subset():
    database = Database(database_id=uuid4().hex)
    with DeleteDatabaseAfterTest(database):
        data = DataLoader(f"{PIXELBRAIN_PATH}/assets/test_data/subset", database)
        
        with tempfile.TemporaryDirectory() as tempdir:
            module = FaceExtractorModule(data, database, image_save_path=tempdir)
            module.process()
            num_images = len(os.listdir(tempdir))
            assert num_images == 1, "there should be 1 faces"