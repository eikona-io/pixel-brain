from pymongo import MongoClient
import chromadb
import numpy as np
from typing import List, Tuple
import shutil
import random
import os


IN_VECTOR_STORE_STR = "IN_VECTOR_STORE"

class Database:
    """
    This class is used to interact with the MongoDB database.
    """
    def __init__(self, database_id: str = 'db', mongo_key: str = None, mongo_vector_key: str = None):
        """
        Initialize the Database class.

        :param mongo_key: The MongoDB connection string. if not provided will use local mongo.
        :param mongo_key: The MongoDB connection string for vector database. if not provided will use local chromadb
        :param database_id: The ID of the database to connect to.
        """
        if mongo_key:
            self._db = MongoClient(mongo_key)[database_id]
        else:
            self._db = MongoClient()[database_id]
        if mongo_vector_key:
            self._vector_db = MongoClient(mongo_vector_key)[database_id]
        else:
            self._local_vector_db_path = f"{os.getcwd()}/chroma/{database_id}"
            self._vector_db = chromadb.PersistentClient(self._local_vector_db_path)
        self._db_id = database_id

    def add_image(self, image_id: str, image_path: str):
        """
        Add an image to the database
        :param image_id (str): image unique identifier
        :param image_path (str): image path (can be remote storage)
        """
        if self._db.images.find_one({'_id': image_id}):
            # already have this image
            return

        self._db.images.update_one({'_id': image_id}, {'$set': {"image_path": image_path}}, upsert=True)

    def store_field(self, image_id: str, field_name: str, field_value: str or np.ndarray):
        """
        Store a field in the database.

        :param image_id: The ID of the image.
        :param field_name: The name of the field to store.
        :param field_value: The value of the field to store.
        """
        if not self._db.images.find_one({'_id': image_id}):
            raise ValueError(f"Image ID {image_id} does not exist in the database")

        if isinstance(field_value, np.ndarray):
            if field_name.find("-") != -1:
                raise ValueError("Field namd with vector values cannot have '-' in it")
            self._store_vector(image_id, field_name, field_value)

        else:
            self._db.images.update_one({'_id': image_id}, {'$set': {field_name: field_value}}, upsert=True)

    def _store_vector(self, image_id : str, field_name: str, embedding: np.ndarray):
        if isinstance(self._vector_db, MongoClient):
            assert False, "Remote vector store not implemented yet"
        else:
            index_fqn = f"{self._db_id}-{field_name}"
            index = self._vector_db.get_or_create_collection(index_fqn, embedding_function=None)
            index.upsert(image_id, embedding.tolist())
            self.store_field(image_id, field_name, IN_VECTOR_STORE_STR)

    def query_vector_field(self, field_name: str, query: np.ndarray, n_results=1) -> Tuple[List[dict], List[float]]:
        """
        Query the relevant vector index for n_results closest images
        and return closest results metadata and distance metric

        :param field_name: The name of the field to query.
        :param query: The query vector.
        :param n_results: The number of results to return. Default is 1.
        :return: A tuple containing a list of the closest result metadata and a list of distance metrics.
        """
        index_fqn = f"{self._db_id}-{field_name}"
        return self._query_vector(index_fqn, query, n_results)

    def _query_vector(self, index_fqn: str, query: np.ndarray, n_results):
        if isinstance(self._vector_db, MongoClient):
            assert False, "Remote vector store not implemented yet"
        else:
            try:
                index = self._vector_db.get_collection(index_fqn)
            except ValueError as err:
                raise RuntimeError(f"Cant find {index_fqn} in vector database")
            results = index.query(
                query.tolist(),
                n_results=n_results
            )
            results_meta = [self.find_image(image_id) for image_id in results['ids'][0]]
            results_dists = results['distances'][0]
            return results_meta, results_dists

    def find_image(self, image_id: str) -> dict:
        """
        Find an image in the database.

        :param image_id: The ID of the image to find.
        :return: The image document.
        """
        return self._db.images.find_one({'_id': image_id})

    def get_all_images(self) -> list:
        """
        Retrieve all images from the database.

        :return: A list of all image documents.
        """
        return list(self._db.images.find())

    def delete_db(self):
        """Delete database (use with caution)"""
        self._db.client.drop_database(self._db_id)
        if not isinstance(self._vector_db, MongoClient):
            # TODO support remote vector store
            shutil.rmtree(self._local_vector_db_path, ignore_errors=True)

    def get_field(self, image_id: str, field_name: str):
        """
        Get a field from an image document.

        :param image_id: The ID of the image to find.
        :param field_name: The name of the field to retrieve.
        :return: The value of the field.
        """
        image_doc = self.find_image(image_id)
        if image_doc is None:
            raise ValueError(f"Could not find {image_id} image")
        if field_name not in image_doc:
            raise ValueError(f"Field {field_name} not found in image document")
        field_value = image_doc[field_name]

        if field_value == IN_VECTOR_STORE_STR:
            if isinstance(self._vector_db, MongoClient):
                raise ValueError(f"{IN_VECTOR_STORE_STR} value is reserved for local vector store")
            index_fqn = f"{self._db_id}-{field_name}"
            try:
                index = self._vector_db.get_collection(index_fqn)
            except ValueError as err:
                raise RuntimeError(f"Cant find {index_fqn} in vector database, maybe it did not persist?")

            field_value = index.get(image_id, include=["embeddings"])['embeddings']
            assert len(field_value) == 1
            field_value = np.array(field_value[0])
        return field_value

    def find_images_with_value(self, field_name: str, value=None):
        """
        Find all images in the database that have a specific field value.

        :param field_name: The name of the field to find.
        :param value: The value of the field to find. If None, find all images that have this field.
        :return: A list of all image documents that match the field value.
        """
        if value is None:
            return list(self._db.images.find({field_name: {"$exists": True}}))
        else:
            return list(self._db.images.find({field_name: value}))
