import pytest
from pixel_brain.database import Database
import numpy as np
from pixel_brain.utils import MONGODB_ATLAS_KEY

def store_field_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='store_field_test')
    db.add_image('test_id', 'test_image_path')
    db.store_field('test_id', 'test_field', 'test_value')
    result = db.find_image('test_id')
    db.delete_db()
    assert result['test_field'] == 'test_value'

def test_store_field_local():
    store_field_run()

def test_store_field_remote():
    store_field_run(MONGODB_ATLAS_KEY)

def store_field_error_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='store_field_error_test')
    with pytest.raises(ValueError):
        db.store_field('non_existent_id', 'test_field', 'test_value')
    db.delete_db()

def test_store_field_error_local():
    store_field_error_run()

def test_store_field_error_remote():
    store_field_error_run(MONGODB_ATLAS_KEY)

def find_image_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='find_image_test')
    db.add_image('test_id', 'test_image_path')
    db.store_field('test_id', 'test_field', 'test_value')
    result = db.find_image('test_id')
    assert result['_id'] == 'test_id'
    db.delete_db()

def test_find_image_local():
    find_image_run()

def test_find_image_remote():
    find_image_run(MONGODB_ATLAS_KEY)

def find_image_error_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='find_image_error_test')
    assert db.find_image("non_existent_id") is None
    db.delete_db()

def test_find_image_error_local():
    find_image_error_run()

def test_find_image_error_remote():
    find_image_error_run(MONGODB_ATLAS_KEY)

def vector_db_multiple_vectors_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, mongo_vector_key=mongo_key, database_id='local_vector_db_multiple_vectors_test')
    for i in range(10):
        db.add_image(f'test_id_{i}', 'test_image_path')
        random_vector = np.array([1, 2, 3]) + np.random.normal(0, np.sqrt(10), 3)
        db.store_field(f'test_id_{i}', 'test_field', random_vector)
    metas, dists = db.query_vector_field('test_field', np.array([1, 2, 3]), n_results=5)
    assert len(metas) == 5
    assert dists[0] < 100, "distance should be less then 100"
    db.delete_db()

def test_local_vector_db_multiple_vectors_run():
    vector_db_multiple_vectors_run()

@pytest.mark.skip(reason="remote vector store not supported yet")
def test_remote_vector_db_multiple_vectors_run():
    vector_db_multiple_vectors_run(MONGODB_ATLAS_KEY)

def vector_db_singel_vector_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, mongo_vector_key=mongo_key, database_id='local_vector_db_singel_vectors_test')
    db.add_image('test_id', 'test_image_path')
    db.store_field('test_id', 'test_field', np.array([1, 2, 3]))
    metas, dists = db.query_vector_field('test_field', np.array([1, 2, 3]))
    assert len(metas) == 1
    assert dists[0] == 0.0, "same vector should have 0 distance"
    db.delete_db()

def test_local_vector_db_singel_vectors_run():
    vector_db_singel_vector_run()

@pytest.mark.skip(reason="remote vector store not supported yet")
def test_remote_vector_db_singel_vectors_run():
    vector_db_singel_vector_run(MONGODB_ATLAS_KEY)

def get_field_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, mongo_vector_key=mongo_key, database_id='get_field_test')
    db.add_image('test_id', 'test_image_path')
    db.store_field('test_id', 'test_field', np.array([1, 2, 3]))
    db.store_field('test_id', 'test_field2', "test value")
    val1 = db.get_field('test_id', 'test_field')
    val2 = db.get_field('test_id', 'test_field2')
    db.delete_db()
    assert isinstance(val1, np.ndarray)
    assert isinstance(val2, str)

def test_get_field_local():
    get_field_run()

@pytest.mark.skip(reason="remote vector store not supported yet")
def test_get_field_remote():
    get_field_run(MONGODB_ATLAS_KEY)

def find_images_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='find_images_test')
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
    db.delete_db()

def test_find_images_local():
    find_images_run()

def test_find_images_remote():
    find_images_run(MONGODB_ATLAS_KEY)