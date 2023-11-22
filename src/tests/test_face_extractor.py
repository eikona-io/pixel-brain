from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.face_extractor import FaceExtractorModule
import tempfile
import os 


def test_face_extractor_module():
    database = Database(database_id="face_extractor")
    # test only one image to save cost
    data = DataLoader("assets/test_data/", database)
    
    # assert len(data) == 42
    with tempfile.TemporaryDirectory() as tempdir:
        module = FaceExtractorModule(data, database, image_save_path=tempdir)
        module.process()
        num_images = len(os.listdir(tempdir))
        assert num_images == 31, "there should be 31 faces"
    
    database.delete_db()