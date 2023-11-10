from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
import torch


def test_next_returns_correct_batch_size():
    # Initializing the Database and DataLoader classes
    db = Database()
    dl = DataLoader('assets/test_data', db, batch_size=2)

    # Testing the __next__ method returns correct batch size
    batch = next(dl)
    assert len(batch) == 2

def test_next_returns_empty_list_when_no_images():
    # Initializing the Database and DataLoader classes with no images
    db = Database()
    dl = DataLoader('assets/no_images', db)

    # Testing the __next__ method returns an empty list when no images
    batch = next(dl)
    assert batch == []

def test_next_returns_correct_image_tensor():
    # Initializing the Database and DataLoader classes
    db = Database()
    dl = DataLoader('assets/test_data', db)

    # Testing the __next__ method returns correct image tensor
    batch = next(dl)
    assert isinstance(batch[0], torch.Tensor)