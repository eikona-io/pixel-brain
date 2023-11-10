from pymongo import MongoClient

class Database:
    """
    This class is used to interact with the MongoDB database.
    """
    def __init__(self, mongo_key: str = None, database_id: str = 'db'):
        """
        Initialize the Database class.
        
        :param mongo_key: The MongoDB connection string.
        :param database_id: The ID of the database to connect to.
        """
        if mongo_key:
            self._db = MongoClient(mongo_key)[database_id]
        else:
            self._db = MongoClient()[database_id]

    def add_image(self, image_id: str, image_path: str):
        """
        Add an image to the database
        :param image_id (str): image unique identifier
        :param image_path (str): image path (can be remote storage)
        """
        if self._db.images.find_one({'_id': image_id}):
            raise ValueError(f"Image ID {image_id} already exists in the database")

        self._db.images.update_one({'_id': image_id}, {'$set': {"image_path": image_path}}, upsert=True)

    def store_field(self, image_id: str, field_name: str, field_value: str):
        """
        Store a field in the database.
        
        :param image_id: The ID of the image.
        :param field_name: The name of the field to store.
        :param field_value: The value of the field to store.
        """
        if not self._db.images.find_one({'_id': image_id}):
            raise ValueError(f"Image ID {image_id} does not exist in the database")
        self._db.images.update_one({'_id': image_id}, {'$set': {field_name: field_value}}, upsert=True)

    def find_image(self, image_id: str) -> dict:
        """
        Find an image in the database.
        
        :param image_id: The ID of the image to find.
        :return: The image document.
        """
        return self._db.images.find_one({'_id': image_id})