from pixelbrain.modules.resnet_classifier import ResnetClassifierModule
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.utils import PIXELBRAIN_PATH


def test_resnet_classifier_module():
    # Create a dummy database
    database = Database(database_id='test_db')

    # Load images from assets/test_images
    images_path = f'{PIXELBRAIN_PATH}/assets/test_data'
    data_loader = DataLoader(images_path, database, batch_size=2)

    # Initialize ResnetClassifierModule
    resnet_classifier = ResnetClassifierModule(data_loader, database)

    resnet_classifier.process()

    processed_metadata = database.get_all_images()
    for image_meta in processed_metadata:
        assert 'class' in image_meta
    # Delete the dummy database after the test
    database.delete_db()


def test_resnet_classifier_module_2_classes():
    # Create a dummy database
    database = Database(database_id='test_db')

    # Load images from assets/test_images
    images_path = f'{PIXELBRAIN_PATH}/assets/test_data'
    data_loader = DataLoader(images_path, database, batch_size=2)

    # Initialize ResnetClassifierModule
    resnet_classifier = ResnetClassifierModule(data_loader, database, class_subset=['sweatshirt', 'tiger cat'])

    resnet_classifier.process()

    processed_metadata = database.get_all_images()
    for image_meta in processed_metadata:
        assert 'class' in image_meta
        assert image_meta['class'] == 'sweatshirt'
    # Delete the dummy database after the test
    database.delete_db()