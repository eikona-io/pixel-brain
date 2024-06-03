from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.gpt4v import GPT4VPeopleDetectorModule
from pixelbrain.utils import PIXELBRAIN_PATH


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
