import pytest
from pixel_brain.database import Database

@pytest.fixture
def db():
    """
    This fixture creates a Database instance with a test database id.
    It is used to setup and teardown the database for each test.
    """
    db = Database(database_id='test_db')
    yield db
    db.delete_db()

def test_store_field(db):
    db.add_image('test_id', 'test_image_path')
    db.store_field('test_id', 'test_field', 'test_value')
    result = db.find_image('test_id')
    assert result['test_field'] == 'test_value'

def test_store_field_error(db):
    with pytest.raises(ValueError):
        db.store_field('non_existent_id', 'test_field', 'test_value')

def test_find_image(db):
    db.add_image('test_id', 'test_image_path')
    db.store_field('test_id', 'test_field', 'test_value')
    result = db.find_image('test_id')
    assert result['_id'] == 'test_id'

def test_find_image_error(db):
    assert db.find_image("non_existent_id") is None