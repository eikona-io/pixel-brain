from pixelbrain.modules.hogs_people_detector import HogsPeopleDetectorModule
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.utils import PIXELBRAIN_PATH

def test_hogs_people_detector_module():
    db = Database()
    data_loader = DataLoader(f'{PIXELBRAIN_PATH}/assets/test_data', db, batch_size=1)
    hogs_people_detector = HogsPeopleDetectorModule(data_loader, db)

    hogs_people_detector.process()
    metadata = db.get_all_images()
    for image_meta in metadata:
        assert "is_person" in image_meta
    db.delete_db()