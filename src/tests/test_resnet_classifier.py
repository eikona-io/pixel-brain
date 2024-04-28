from pixelbrain.modules.resnet_classifier import ResnetClassifierModule
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.utils import PIXELBRAIN_PATH
from tests.test_utils import DeleteDatabaseAfterTest
from uuid import uuid4


def test_resnet_classifier_module():
    # Create a dummy database
    database_id = uuid4().hex
    database = Database(database_id=database_id)
    with DeleteDatabaseAfterTest(database):
        # Load images from assets/test_images
        images_path = f"{PIXELBRAIN_PATH}/assets/test_data"
        data_loader = DataLoader(images_path, database, batch_size=2)

        # Initialize ResnetClassifierModule
        resnet_classifier = ResnetClassifierModule(data_loader, database)

        resnet_classifier.process()

        processed_metadata = database.get_all_images()
        for image_meta in processed_metadata:
            assert "class" in image_meta


def test_resnet_classifier_module_2_classes():
    # Create a dummy database
    database_id = uuid4().hex
    database = Database(database_id=database_id)
    with DeleteDatabaseAfterTest(database):

        # Load images from assets/test_images
        images_path = f"{PIXELBRAIN_PATH}/assets/test_data"
        data_loader = DataLoader(images_path, database, batch_size=2)

        # Initialize ResnetClassifierModule
        resnet_classifier = ResnetClassifierModule(
            data_loader, database, class_subset=["sweatshirt", "tiger cat"]
        )

        resnet_classifier.process()

        processed_metadata = database.get_all_images()
        for image_meta in processed_metadata:
            assert "class" in image_meta
            assert image_meta["class"] == "sweatshirt"
