import os
import shutil
import tempfile
from PIL import Image
from torchvision.transforms import ToTensor
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.image_save import ImageSaveModule
from pixelbrain.utils import PIXELBRAIN_PATH
from test_utils import DeleteDatabaseAfterTest


def test_image_save_module():
    # Setup
    with tempfile.TemporaryDirectory() as save_dir:
        image_paths = [
            f"{PIXELBRAIN_PATH}/src/tests/mock_data/mock_images/hue_pipeline_images/blond1.jpeg",
            f"{PIXELBRAIN_PATH}/src/tests/mock_data/mock_images/hue_pipeline_images/blond2.jpeg",
        ]
        database = Database()
        with DeleteDatabaseAfterTest(database):
            data_loader = DataLoader(image_paths, database)
            image_save_module = ImageSaveModule(data_loader, database, save_dir)
            image_save_module.process()

            # Check if the number of files in the directory matches the number of images
            saved_files = os.listdir(save_dir)
            assert len(saved_files) == len(image_paths), f"Expected {len(image_paths)} images, but found {len(saved_files)}."