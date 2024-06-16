from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
import torch
import pytest
from pixelbrain.utils import PIXELBRAIN_PATH
from tests.test_utils import DeleteDatabaseAfterTest
from uuid import uuid4


def next_returns_correct_batch_size(batch_size):
    # Initializing the Database and DataLoader classes
    db = Database(uuid4().hex)
    with DeleteDatabaseAfterTest(db):
        dl = DataLoader(f"{PIXELBRAIN_PATH}/assets/test_data", db, batch_size)

        # Testing the __next__ method returns correct batch size
        batch_ids, batch_images = next(dl)
        assert len(batch_ids) == batch_size


def test_next_returns_batch_1():
    next_returns_correct_batch_size(1)


def test_next_returns_batch_2():
    next_returns_correct_batch_size(2)


def test_next_returns_batch_4():
    next_returns_correct_batch_size(4)


def test_next_returns_empty_list_when_no_images():
    # Initializing the Database and DataLoader classes with no images
    db = Database(uuid4().hex)
    with DeleteDatabaseAfterTest(db):
        dl = DataLoader(f"{PIXELBRAIN_PATH}/assets/no_images", db)

        # Testing the __next__ method raises stop iteration when empty
        with pytest.raises(StopIteration):
            batch = next(dl)


def next_returns_correct_image_tensor_and_id(decode):
    # Initializing the Database and DataLoader classes
    db = Database(uuid4().hex)
    with DeleteDatabaseAfterTest(db):
        dl = DataLoader(f"{PIXELBRAIN_PATH}/assets/test_data", db, decode_images=decode)

        # Testing the __next__ method returns correct image tensor
        batch_ids, batch_imgs = next(dl)
        assert isinstance(batch_ids[0], str)
        assert isinstance(batch_imgs[0], torch.Tensor)
        assert (
            len(batch_imgs[0].shape) == 3 if decode else len(batch_imgs[0].shape) == 1
        ), f"wrong shape for batch_images: {batch_imgs[0].shape}"


def test_next_returns_correct_image_tensor_and_id():
    next_returns_correct_image_tensor_and_id(True)


def test_next_returns_correct_image_tensor_and_id_no_decode():
    next_returns_correct_image_tensor_and_id(False)


def test_filter_method():
    # Initializing the Database and DataLoader classes
    db = Database("filter_method_test")
    with DeleteDatabaseAfterTest(db):
        dl = DataLoader(f"{PIXELBRAIN_PATH}/assets/test_data", db)
        dl2 = DataLoader(f"{PIXELBRAIN_PATH}/assets/test_data", db)
        # iterate through data to load it into database
        for i, (image_ids, _) in enumerate(dl2):
            if i < 3:
                # do not store any value for first images
                continue
            db.store_field(image_ids[0], "category", str(i % 2))

        # sanity
        assert len(dl) == 44, "there should be 44 images"
        dl.filter("category")
        assert len(dl) == 41, "there should be 41 image with category"
        dl.filter("category", "1")
        assert len(dl) == 21, "there should be 21 image with category 1"


def test_dataloader_with_remote_urls():
    # Initializing the Database and DataLoader classes with remote URLs
    db = Database()
    remote_urls = [
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716993612/user_photos/6c6a27d0-fd89-41bf-bc5b-22352cf0f049/raw/AKuaiKrUTEuBqjxP17Oa6txcypMVM0ecMh7tnFPcLX492vPjZ5Gh6nMRlrCJt_kK2hOFdzZkTjxMbxVJcNSRWLc1-gyOc_bUpw.jpg",
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716993612/user_photos/6c6a27d0-fd89-41bf-bc5b-22352cf0f049/raw/AKuaiKrUTEuBqjxP17Oa6txcypMVM0ecMh7tnFPcLX492vPjZ5Gh6nMRlrCJt_kK2hOFdzZkTjxMbxVJcNSRWLc1-gyOc_bUpw.jpg",
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716993612/user_photos/6c6a27d0-fd89-41bf-bc5b-22352cf0f049/raw/AKuaiKrUTEuBqjxP17Oa6txcypMVM0ecMh7tnFPcLX492vPjZ5Gh6nMRlrCJt_kK2hOFdzZkTjxMbxVJcNSRWLc1-gyOc_bUpw.jpg",
    ]
    with DeleteDatabaseAfterTest(db):
        dl = DataLoader(remote_urls, db, decode_images=True, load_images=True)

        # Testing the __next__ method returns correct image tensor
        batch_ids, batch_imgs = next(dl)
        assert len(batch_ids) == 1 and len(batch_imgs) == 1, "there should be 1 image"
        assert isinstance(batch_ids[0], str)
        assert isinstance(batch_imgs[0], torch.Tensor)
        assert (
            len(batch_imgs[0].shape) == 3
        ), f"wrong shape for batch_images: {batch_imgs[0].shape}"

        # Testing the length of the DataLoader
        assert len(dl) == 2, "there should be 2 images"

def test_dataloader_with_remote_urls_batch_size_3():
    # Initializing the Database and DataLoader classes with remote URLs
    db = Database()
    remote_urls = [
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716993612/user_photos/6c6a27d0-fd89-41bf-bc5b-22352cf0f049/raw/AKuaiKrUTEuBqjxP17Oa6txcypMVM0ecMh7tnFPcLX492vPjZ5Gh6nMRlrCJt_kK2hOFdzZkTjxMbxVJcNSRWLc1-gyOc_bUpw.jpg",
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716993612/user_photos/6c6a27d0-fd89-41bf-bc5b-22352cf0f049/raw/AKuaiKrUTEuBqjxP17Oa6txcypMVM0ecMh7tnFPcLX492vPjZ5Gh6nMRlrCJt_kK2hOFdzZkTjxMbxVJcNSRWLc1-gyOc_bUpw.jpg",
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716993612/user_photos/6c6a27d0-fd89-41bf-bc5b-22352cf0f049/raw/AKuaiKrUTEuBqjxP17Oa6txcypMVM0ecMh7tnFPcLX492vPjZ5Gh6nMRlrCJt_kK2hOFdzZkTjxMbxVJcNSRWLc1-gyOc_bUpw.jpg",
    ]
    with DeleteDatabaseAfterTest(db):
        dl = DataLoader(remote_urls, db, decode_images=True, load_images=True, batch_size=3)

        # Testing the __next__ method returns correct image tensor
        batch_ids, batch_imgs = next(dl)
        assert len(batch_ids) == 3 and len(batch_imgs) == 3, "there should be batch_size images"
        assert isinstance(batch_ids[0], str)
        assert isinstance(batch_imgs[0], torch.Tensor)
        assert (
            len(batch_imgs[0].shape) == 3
        ), f"wrong shape for batch_images: {batch_imgs[0].shape}"

        # Testing the length of the DataLoader
        assert len(dl) == 0, "there should be 0 images"


def test_dataloader_with_remote_urls_no_decode():
    # Initializing the Database and DataLoader classes with remote URLs without decoding images
    db = Database()
    remote_urls = [
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716993612/user_photos/6c6a27d0-fd89-41bf-bc5b-22352cf0f049/raw/AKuaiKrUTEuBqjxP17Oa6txcypMVM0ecMh7tnFPcLX492vPjZ5Gh6nMRlrCJt_kK2hOFdzZkTjxMbxVJcNSRWLc1-gyOc_bUpw.jpg",
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716993612/user_photos/6c6a27d0-fd89-41bf-bc5b-22352cf0f049/raw/AKuaiKrUTEuBqjxP17Oa6txcypMVM0ecMh7tnFPcLX492vPjZ5Gh6nMRlrCJt_kK2hOFdzZkTjxMbxVJcNSRWLc1-gyOc_bUpw.jpg",
        "https://res.cloudinary.com/dxgcobmaz/image/upload/v1716993612/user_photos/6c6a27d0-fd89-41bf-bc5b-22352cf0f049/raw/AKuaiKrUTEuBqjxP17Oa6txcypMVM0ecMh7tnFPcLX492vPjZ5Gh6nMRlrCJt_kK2hOFdzZkTjxMbxVJcNSRWLc1-gyOc_bUpw.jpg",
    ]
    with DeleteDatabaseAfterTest(db):
        dl = DataLoader(remote_urls, db, decode_images=False, load_images=False)

        # Testing the __next__ method returns correct image paths
        batch_ids, batch_imgs = next(dl)
        assert len(batch_ids) == 1 and len(batch_imgs) == 1, "there should be 1 image"
        assert isinstance(batch_ids[0], str)
        assert isinstance(batch_imgs[0], str)
        assert batch_imgs[0].startswith(
            "https://"
        ), f"wrong format for batch_images: {batch_imgs[0]}"

        # Testing the length of the DataLoader
        assert len(dl) == 2, "there should be 2 images"