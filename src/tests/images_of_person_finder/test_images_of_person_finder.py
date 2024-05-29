from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.images_of_person_finder import ImagesOfPersonFinder
from pixelbrain.utils import PIXELBRAIN_PATH
from tests.test_utils import DeleteDatabaseAfterTest


def images_of_person_finder_run(path_to_person_image: str):
    database = Database()
    data = DataLoader(
        f"{PIXELBRAIN_PATH}/src/tests/images_of_person_finder/data",
        database,
        batch_size=1,
    )

    module = ImagesOfPersonFinder(database, data, path_to_person_image)
    with DeleteDatabaseAfterTest(database):
        module.process()
        metadata = database.get_all_images()
        matched_persons = {
            image["image_path"]: image["matched_person"] for image in metadata
        }

        nof_matches = 0
        for image_path, is_person in matched_persons.items():
            assert isinstance(is_person, bool)
            if is_person:
                nof_matches += 1
        assert nof_matches == 6, "There should be exactly 6 matches"


def test_images_of_person_finder():
    path_to_person_image = (
        f"{PIXELBRAIN_PATH}/src/tests/images_of_person_finder/1_7.jpg"
    )
    images_of_person_finder_run(path_to_person_image=path_to_person_image)
