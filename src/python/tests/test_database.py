import pytest
from pixel_brain.database import Database
import numpy as np


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

def test_local_vector_db_multiple_vectors(db):
    for i in range(10):
        db.add_image(f'test_id_{i}', 'test_image_path')
        random_vector = np.array([1, 2, 3]) + np.random.normal(0, np.sqrt(10), 3)
        db.store_field(f'test_id_{i}', 'test_field', random_vector)
    metas, dists = db.query_vector_field('test_field', np.array([1, 2, 3]), n_results=5)
    assert len(metas) == 5
    assert dists[0] < 100, "distance should be less then 100"

def test_local_vector_db_singel_vector(db):
    db.add_image('test_id', 'test_image_path')
    db.store_field('test_id', 'test_field', np.array([1, 2, 3]))
    metas, dists = db.query_vector_field('test_field', np.array([1, 2, 3]))
    assert len(metas) == 1
    assert dists[0] == 0.0, "same vector should have 0 distance"

def test_get_field(db):
    db.add_image('test_id', 'test_image_path')
    db.store_field('test_id', 'test_field', np.array([1, 2, 3]))
    db.store_field('test_id', 'test_field2', "test value")
    val1 = db.get_field('test_id', 'test_field')
    val2 = db.get_field('test_id', 'test_field2')
    db.delete_db()
    assert isinstance(val1, np.ndarray)
    assert isinstance(val2, str)

def test_find_images(db):
    db.add_image('test_id', 'test_image_path')
    db.store_field('test_id', 'test_field', 'test_value')
    db.store_field('test_id', 'test_field2', 'test_value2')
    # test with value provided
    result = db.find_images_with_value('test_field', 'test_value')
    assert len(result) == 1
    assert result[0]['_id'] == 'test_id'
    assert result[0]['test_field'] == 'test_value'
    result = db.find_images_with_value('test_field2', 'test_value2')
    assert len(result) == 1
    assert result[0]['_id'] == 'test_id'
    assert result[0]['test_field2'] == 'test_value2'
    # test without value provided
    result = db.find_images_with_value('test_field')
    assert len(result) == 1
    assert result[0]['_id'] == 'test_id'
    assert result[0]['test_field'] == 'test_value'
    result = db.find_images_with_value('test_field2')
    assert len(result) == 1
    assert result[0]['_id'] == 'test_id'
    assert result[0]['test_field2'] == 'test_value2'