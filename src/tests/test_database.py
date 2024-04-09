import pytest
from pixelbrain.database import Database
import numpy as np
from pixelbrain.utils import MONGODB_ATLAS_KEY, PINECONE_KEY
import tempfile
import time
import pytest
from uuid import uuid4
from utils import profile_callstack, DeleteDatabaseAfterTest, StrictMock


def store_field_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='store_field_test')
    with DeleteDatabaseAfterTest(db):
        db.add_image('test_id', 'test_image_path')
        db.store_field('test_id', 'test_field', 'test_value')
        result = db.find_image('test_id')
        assert result['test_field'] == 'test_value'

def test_store_field_local():
    store_field_run()

def test_store_field_remote():
    store_field_run(MONGODB_ATLAS_KEY)

def store_field_error_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='store_field_error_test')
    with DeleteDatabaseAfterTest(db):
        with pytest.raises(ValueError):
            db.store_field('non_existent_id', 'test_field', 'test_value')

def test_store_field_error_local():
    store_field_error_run()

def test_store_field_error_remote():
    store_field_error_run(MONGODB_ATLAS_KEY)

def find_image_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='find_image_test')
    with DeleteDatabaseAfterTest(db):
        db.add_image('test_id', 'test_image_path')
        db.store_field('test_id', 'test_field', 'test_value')
        result = db.find_image('test_id')
        assert result['_id'] == 'test_id'

def test_find_image_local():
    find_image_run()

def test_find_image_remote():
    find_image_run(MONGODB_ATLAS_KEY)

def find_image_error_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='find_image_error_test')
    with DeleteDatabaseAfterTest(db):
        assert db.find_image("non_existent_id") is None

def test_find_image_error_local():
    find_image_error_run()

def test_find_image_error_remote():
    find_image_error_run(MONGODB_ATLAS_KEY)


def vector_db_multiple_vectors_run(mongo_key=None, pinecone_key=None):
    db_id = 'hue' if pinecone_key else str(uuid4())
    db = Database(mongo_key=mongo_key, pinecone_vector_key=pinecone_key, database_id=db_id)
    with DeleteDatabaseAfterTest(db):
        for i in range(10):
            db.add_image(f'test_id_{i}', 'test_image_path')
            random_vector = np.random.normal(0, np.sqrt(10), 512)
            db.store_field(f'test_id_{i}', 'test_field', random_vector)
        if pinecone_key:
            time.sleep(20) # sync pinecone
        metas, dists = db.query_vector_field('test_field', np.random.normal(0, np.sqrt(10), 512), n_results=5)
        assert len(metas) == 5
        assert dists[0] > 0, "distance should be more then 0"
        if pinecone_key:
            db._vector_db.delete(delete_all=True, namespace='hue-test_field')

def test_local_vector_db_multiple_vectors_run():
    vector_db_multiple_vectors_run()

@pytest.mark.slow_suit
def test_remote_vector_db_multiple_vectors_run():
    vector_db_multiple_vectors_run(MONGODB_ATLAS_KEY, PINECONE_KEY)

def vector_db_single_vector_run(mongo_key=None, pinecone_key=None):
    db_id = 'hue' if pinecone_key else str(uuid4())
    db = Database(mongo_key=mongo_key, pinecone_vector_key=pinecone_key, database_id=db_id)
    with DeleteDatabaseAfterTest(db):
        test_id = str(uuid4())
        db.add_image(test_id, 'test_image_path')
        rand_vec = np.random.uniform(0, 1, 512)
        rand_vec2 = np.random.uniform(0, 1, 512)
        db.store_field(test_id, 'test_field', rand_vec)
        if pinecone_key:
            time.sleep(15) # sync pinecone
        metas, dists = db.query_vector_field('test_field', rand_vec2)
        assert len(metas) == 1
        assert dists[0] > 0.0, "different vector should have greater than 0 distance"
        if pinecone_key:
            db._vector_db.delete(delete_all=True, namespace='hue-test_field')

def test_local_vector_db_single_vectors_run():
    vector_db_single_vector_run()

@pytest.mark.slow_suit
def test_remote_vector_db_single_vectors_run():
    vector_db_single_vector_run(MONGODB_ATLAS_KEY, PINECONE_KEY)

def get_field_run(mongo_key=None, pinecone_key=None):
    db_id = 'hue' if pinecone_key else str(uuid4())
    db = Database(mongo_key=mongo_key, pinecone_vector_key=pinecone_key, database_id=db_id)
    with DeleteDatabaseAfterTest(db):
        test_id = str(uuid4())
        db.add_image(test_id, 'test_image_path')
        db.store_field(test_id, 'test_field', np.random.uniform(0, 1, 512))
        if pinecone_key:
            time.sleep(20) # sync pinecone
        db.store_field(test_id, 'test_field2', "test value")
        val1 = db.get_field(test_id, 'test_field')
        val2 = db.get_field(test_id, 'test_field2')
        assert isinstance(val1, np.ndarray)
        assert isinstance(val2, str)
        if pinecone_key:
            db._vector_db.delete(delete_all=True, namespace='hue-test_field')

def test_get_field_local():
    get_field_run()

@pytest.mark.slow_suit
def test_get_field_remote():
    get_field_run(MONGODB_ATLAS_KEY, PINECONE_KEY)

def find_images_run(mongo_key=None):
    db = Database(mongo_key=mongo_key, database_id='find_images_test')
    with DeleteDatabaseAfterTest(db):
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

def test_find_images_local():
    find_images_run()

def test_find_images_remote():
    find_images_run(MONGODB_ATLAS_KEY)
    
def test_create_database_from_csv(mongo_key=None):
    # Create a new database
    db = Database(database_id='create_db_from_csv_test', mongo_key=mongo_key)
    with DeleteDatabaseAfterTest(db):
        # Add images and fields to the database
        db.add_image('image1', 'path1') 
        db.store_field('image1', 'field1', 'value1')
        db.add_image('image2', 'path2')
        db.store_field('image2', 'field2', 'value2')
        # Export the database to a CSV file
        with tempfile.TemporaryDirectory() as tempdir:
            db.export_to_csv(f'{tempdir}/test.csv')
            # Create a new database from the CSV file
            new_db = Database.create_from_csv(f'{tempdir}/test.csv', database_id='new_db')
            with DeleteDatabaseAfterTest(new_db):
                # Compare the images and fields in the two databases
                assert new_db.get_all_images() == db.get_all_images()
                assert new_db.get_field('image1', 'field1') == db.get_field('image1', 'field1')
                assert new_db.get_field('image2', 'field2') == db.get_field('image2', 'field2')
    
    # Delete the databases
    db.delete_db()
    new_db.delete_db()

def test_create_database_from_csv_local():
    test_create_database_from_csv()

def test_create_database_from_csv_remote():
    test_create_database_from_csv(MONGODB_ATLAS_KEY)

def filter_method_run(mongo_key=None):
    # Create a new database
    db = Database(database_id='test_db', mongo_key=mongo_key)
    with DeleteDatabaseAfterTest(db):
        # Add images and fields to the database
        db.add_image('image1', 'path1')
        db.store_field('image1', 'field1', 'value1')
        db.store_field('image1', 'field2', 'value2')
        db.add_image('image2', 'path2')
        db.store_field('image2', 'field1', 'value1')
        db.add_image('image3', 'path3')
        assert len(db.get_all_images()) == 3
        # Test filter method with field_value provided
        db.filter('field1', 'value1')
        assert len(db.get_all_images()) == 2
        # Test filter method with field_value=None
        db.filter('field2')
        assert len(db.get_all_images()) == 1
        # Test follow-up instruction
        db.filter('field2', 'value1')
        assert len(db.get_all_images()) == 0

def test_filter_method_local():
    filter_method_run()

def test_filter_method_remote():
    filter_method_run(MONGODB_ATLAS_KEY)

def test_filter_unidentified_people():
    # Create a new database
    db = Database(database_id='unident_test_db')
    with DeleteDatabaseAfterTest(db):
        # Add images and fields to the database
        db.add_image('image1', 'path1')
        db.store_field('image1', 'is_person', str(True))
        db.store_field('image1', 'assigned_identity', 'person1')
        db.add_image('image2', 'path2')
        db.store_field('image2', 'is_person', 'False')
        db.store_field('image2', 'assigned_identity', 'None')
        db.add_image('image3', 'path3')
        db.store_field('image3', 'is_person', 'True')
        assert len(db.get_all_images()) == 3
        # Test filter_unidentified_people method
        db.filter_unidentified_people()
        assert len(db.get_all_images()) == 2

def clone_row_method_run(mongo_key=None):
    # Create a new database
    db = Database(database_id='test_db', mongo_key=mongo_key)
    with DeleteDatabaseAfterTest(db):
        # Add images and fields to the database
        db.add_image('image1', 'path1')
        db.store_field('image1', 'field1', 'value1')
        db.add_image('image2', 'path2')
        # Clone row from image1 to image2
        db.clone_row('image1', 'image2')
        # Check if the field1 of image2 is the same as that of image1
        assert db.get_field('image2', 'field1') == 'value1'

def test_clone_row_method_local():
    clone_row_method_run()

def test_clone_row_method_remote():
    clone_row_method_run(MONGODB_ATLAS_KEY)