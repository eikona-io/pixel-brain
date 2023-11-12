from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
import torch
import pytest


def next_returns_correct_batch_size(batch_size):
    # Initializing the Database and DataLoader classes
    db = Database()
    dl = DataLoader('assets/test_data', db, batch_size)

    # Testing the __next__ method returns correct batch size
    batch_ids, batch_images = next(dl)
    assert len(batch_ids) == batch_size
    db.delete_db()

def test_next_returns_batch_1():
    next_returns_correct_batch_size(1)

def test_next_returns_batch_2():
    next_returns_correct_batch_size(2)

def test_next_returns_batch_4():
    next_returns_correct_batch_size(4)

def test_next_returns_empty_list_when_no_images():
    # Initializing the Database and DataLoader classes with no images
    db = Database()
    dl = DataLoader('assets/no_images', db)

    # Testing the __next__ method raises stop iteration when empty
    with pytest.raises(StopIteration):
        batch = next(dl)
    db.delete_db()

def next_returns_correct_image_tensor_and_id(decode):
    # Initializing the Database and DataLoader classes
    db = Database()
    dl = DataLoader('assets/test_data', db, decode_images=decode)

    # Testing the __next__ method returns correct image tensor
    batch_ids, batch_imgs = next(dl)
    assert isinstance(batch_ids[0], str)
    assert isinstance(batch_imgs[0], torch.Tensor)
    assert len(batch_imgs[0].shape) == 3 if decode else len(batch_imgs[0].shape) == 1, f"wrong shape for batch_images: {batch_imgs[0].shape}"
    db.delete_db()
    
def test_next_returns_correct_image_tensor_and_id():
    next_returns_correct_image_tensor_and_id(True)
    
def test_next_returns_correct_image_tensor_and_id_no_decode():
    next_returns_correct_image_tensor_and_id(False)
    