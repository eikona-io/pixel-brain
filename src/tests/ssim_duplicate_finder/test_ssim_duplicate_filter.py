from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.ssim_duplicate_filter import SSIMDuplicateFilter
from pixelbrain.utils import PIXELBRAIN_PATH
from tests.test_utils import DeleteDatabaseAfterTest


def test_ssim_duplicate_filter():
    database = Database()
    data_loader = DataLoader(
        f"{PIXELBRAIN_PATH}/src/tests/ssim_duplicate_finder/data",
        database,
    )

    ssim_filter = SSIMDuplicateFilter(data_loader, database, ssim_threshold=0.45)

    with DeleteDatabaseAfterTest(database):
        ssim_filter.process()

        metadata = database.get_all_images()
        nof_duplicates = sum(image["is_duplicate"] for image in metadata)

        assert nof_duplicates == 5, "There should be 5 duplicate images"