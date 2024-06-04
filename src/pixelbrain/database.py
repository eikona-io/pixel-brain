from pymongo import MongoClient, ASCENDING, DESCENDING
from montydb import MontyClient
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional
import shutil
import pandas as pd
import os
import math
from tqdm import tqdm
from pinecone import Pinecone, Index
import pandas as pd
from uuid import uuid4
from motor.motor_asyncio import AsyncIOMotorClient

IN_VECTOR_STORE_STR = "IN_VECTOR_STORE"


class Database:
    """
    This class is used to interact with the MongoDB database.
    """

    def __init__(
        self,
        database_id: str = None,
        mongo_key: str = None,
        pinecone_vector_key: str = None,
    ):
        """
        Initialize the Database class.

        :param: database_id: The ID of the database to connect to. defaults to a random UUID
        :param mongo_key: The MongoDB connection string. if not provided will use local mongo.
        :param pinecone_vector_key: The pinecone connection string for vector database. if not provided will use local chromadb
        :param database_id: The ID of the database to connect to.
        """
        if not database_id:
            database_id = uuid4().hex[:8]
        if mongo_key:
            self._db = MongoClient(mongo_key)[database_id]
            self._async_db = AsyncIOMotorClient(mongo_key)[database_id]
            self._local_montydb_path = None
        else:
            # runs an bson(binary-json) mongo compatible client
            # TODO: omerh -> change to in-memory storage, but currently there is a bug in pytest when doing so
            self._local_montydb_path = f"{os.getcwd()}/monty/{database_id}"
            self._db = MontyClient(f"montydb:///{self._local_montydb_path}")[
                database_id
            ]
            self._async_db = None
        if pinecone_vector_key:
            # TODO: omerh -> change to creating a new index once we move out of free version
            self._vector_db = Pinecone(pinecone_vector_key).Index(database_id)
        else:
            self._local_vector_db_path = f"{os.getcwd()}/chroma/{database_id}"
            chroma_settings = Settings(anonymized_telemetry=False)
            self._vector_db = chromadb.PersistentClient(
                self._local_vector_db_path, settings=chroma_settings
            )
        self._db_id = database_id

    def add_image(self, image_id: str, image_path: str):
        """
        Add an image to the database
        :param image_id (str): image unique identifier
        :param image_path (str): image path (can be remote storage)
        """
        if self._db.images.find_one({"_id": image_id}):
            # already have this image
            return

        _ = self._db.images.update_one(
            {"_id": image_id}, {"$set": {"image_path": image_path}}, upsert=True
        )

    async def async_add_image(self, image_id: str, image_path: str):
        """
        Asynchronously add an image to the database
        :param image_id (str): image unique identifier
        :param image_path (str): image path (can be remote storage)
        """
        if isinstance(self._db, MontyClient):
            raise RuntimeError("Async operations are not supported for local MontyDB")
        if await self._async_db.images.find_one({"_id": image_id}):
            # already have this image
            return

        _ = await self._async_db.images.update_one(
            {"_id": image_id}, {"$set": {"image_path": image_path}}, upsert=True
        )

    def store_field(
        self,
        image_id: str,
        field_name: Union[str, int, float],
        field_value: str or np.ndarray,
    ):
        """
        Store a field in the database.

        :param image_id: The ID of the image.
        :param field_name: The name of the field to store.
        :param field_value: The value of the field to store.
        """
        if not self._db.images.find_one({"_id": image_id}):
            raise ValueError(f"Image ID {image_id} does not exist in the database")

        if isinstance(field_value, np.ndarray):
            if field_name.find("-") != -1:
                raise ValueError("Field namd with vector values cannot have '-' in it")
            _ = self._store_vector(image_id, field_name, field_value)

        else:
            _ = self._db.images.update_one(
                {"_id": image_id}, {"$set": {field_name: field_value}}, upsert=True
            )

    async def async_store_field(
        self,
        image_id: str,
        field_name: Union[str, int, float],
        field_value: Union[str, np.ndarray],
    ):
        """
        Asynchronously store a field in the database.

        :param image_id: The ID of the image.
        :param field_name: The name of the field to store.
        :param field_value: The value of the field to store.
        """
        if isinstance(self._db, MontyClient):
            raise RuntimeError("Async operations are not supported for local MontyDB")

        if not await self._async_db.images.find_one({"_id": image_id}):
            raise ValueError(f"Image ID {image_id} does not exist in the database")

        if isinstance(field_value, np.ndarray):
            raise NotImplementedError("Async vector storage is not implemented")

        else:
            await self._async_db.images.update_one(
                {"_id": image_id}, {"$set": {field_name: field_value}}, upsert=True
            )

    def is_async(self):
        return self._async_db is not None

    def _store_vector(self, image_id: str, field_name: str, embedding: np.ndarray):
        index_fqn = f"{self._db_id}-{field_name}"
        if isinstance(self._vector_db, Index):
            vec = {"id": image_id, "values": embedding.tolist()}
            _ = self._vector_db.upsert(vectors=[vec], namespace=index_fqn)
            _ = self.store_field(
                image_id, field_name, f"{IN_VECTOR_STORE_STR}:pinecone/{self._db_id}"
            )
        else:
            index = self._vector_db.get_or_create_collection(
                index_fqn, embedding_function=None
            )
            _ = index.upsert(image_id, embedding.tolist())
            _ = self.store_field(
                image_id,
                field_name,
                f"{IN_VECTOR_STORE_STR}:{self._local_vector_db_path}",
            )

    def query_vector_field(
        self,
        field_name: str,
        query: np.ndarray,
        n_results: int = 1,
        include_vectors: bool = False,
        include_meta: bool = True,
    ) -> Tuple[List[dict], List[float], Optional[List[np.ndarray]]]:
        """
        Query the relevant vector index for n_results closest images
        and return closest results metadata and distance metric

        :param field_name: The name of the field to query.
        :param query: The query vector.
        :param n_results: The number of results to return. Default is 1.
        :param include_vectors: Whether to include the vectors in the results. Default is False.
        :param include_meta: Whether to include the metadata in the results. Default is True.
        :return: A tuple containing a list of the closest result metadata and a list of distance metrics.
        """
        index_fqn = f"{self._db_id}-{field_name}"
        return self._query_vector(
            index_fqn, query, n_results, include_vectors, include_meta
        )

    def _query_vector(
        self,
        index_fqn: str,
        query: np.ndarray,
        n_results: int,
        include_vectors: bool,
        include_meta: bool,
    ):
        if isinstance(self._vector_db, Index):
            try:
                results = self._vector_db.query(
                    vector=query.tolist(), namespace=index_fqn, top_k=n_results
                )
            except ValueError as err:
                raise RuntimeError(f"Cant find {index_fqn} in vector database")
            results_meta = [
                self.find_image(match["id"]) for match in results["matches"]
            ]
            results_dists = [match["score"] for match in results["matches"]]
            return results_meta, results_dists
        else:
            try:
                index = self._vector_db.get_collection(index_fqn)
            except ValueError as err:
                raise RuntimeError(f"Cant find {index_fqn} in vector database")
            include = ["distances"]
            if include_vectors:
                include += ["embeddings"]
            results = index.query(query.tolist(), n_results=n_results, include=include)
            results_meta = (
                [self.find_image(image_id) for image_id in results["ids"][0]]
                if include_meta
                else None
            )
            results_dists = results["distances"][0]
            results_embeddings = None
            if include_vectors:
                results_embeddings = results["embeddings"][0]
                results_embeddings = [
                    np.array(embedding) for embedding in results_embeddings
                ]

            return results_meta, results_dists, results_embeddings

    def find_image(self, image_id: str) -> dict:
        """
        Find an image in the database.

        :param image_id: The ID of the image to find.
        :return: The image document.
        """
        return self._db.images.find_one({"_id": image_id})

    async def async_find_image(self, image_id: str) -> dict:
        """
        Asynchronously find an image in the database.

        :param image_id: The ID of the image to find.
        :return: The image document.
        """
        if isinstance(self._db, MontyClient):
            raise RuntimeError("Async operations are not supported for local MontyDB")
        return await self._async_db.images.find_one({"_id": image_id})

    def get_all_images(self) -> list:
        """
        Retrieve all images from the database.

        :return: A list of all image documents.
        """
        return list(self._db.images.find())

    def delete_db(self):
        """Delete database (use with caution)"""
        _ = self._db.client.drop_database(self._db_id)
        if not isinstance(self._vector_db, Index):
            # TODO support remote vector store
            _ = shutil.rmtree(self._local_vector_db_path, ignore_errors=True)
        if self._local_montydb_path:
            _ = shutil.rmtree(self._local_montydb_path, ignore_errors=True)

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

        if field_value.find(IN_VECTOR_STORE_STR) != -1:
            index_fqn = f"{self._db_id}-{field_name}"
            if isinstance(self._vector_db, Index):
                result = self._vector_db.fetch(ids=[image_id], namespace=index_fqn)
                if result:
                    result_id = list(result["vectors"].keys())
                    if len(result_id) > 1:
                        raise ValueError(f"Found more than 1 vector with id {image_id}")
                    elif len(result_id) == 0:
                        raise ValueError(f"No vector found with id {image_id}")
                    field_value = result["vectors"][result_id[0]]["values"]
                else:
                    raise ValueError(f"Vector for {image_id} not found in {index_fqn}")
            else:
                try:
                    index = self._vector_db.get_collection(index_fqn)
                except ValueError as err:
                    raise RuntimeError(
                        f"Cant find {index_fqn} in vector database, maybe it did not persist?"
                    )

                field_value = index.get(image_id, include=["embeddings"])["embeddings"]
                assert len(field_value) == 1
                field_value = field_value[0]
            field_value = np.array(field_value)
        return field_value

    def find_images_with_value(
        self, field_name: str, value=None, sort_by=None, ascending=True
    ):
        """
        Find all images in the database that have a specific field value, with an optional sorting.

        :param field_name: The name of the field to find.
        :param value: The value of the field to find. If None, find all images that have this field.
        :param sort_by: The field name to sort the results by. If None, no sorting is applied.
        :param ascending: Determines the sorting order. True for ascending, False for descending.
        :return: A list of all image documents that match the field value, optionally sorted.
        """
        query = (
            {field_name: {"$exists": True}} if value is None else {field_name: value}
        )
        sort_order = ASCENDING if ascending else DESCENDING
        if sort_by:
            return list(self._db.images.find(query).sort(sort_by, sort_order))
        else:
            return list(self._db.images.find(query))

    def find_images_with_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find all images in the database according to the filters.

        :param filters: A list of filters to apply to the image documents.
        :return: A list of all image documents that have all the specified fields.
        """
        query = {
            field_name: value if value is not None else {"$exists": True}
            for field_name, value in filters.items()
        }
        return list(self._db.images.find(query))

    def aggregate_on_field(
        self, field_name: str, sort_order: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Aggregate ids by a given field to a list, and sort by count if required.
        :param field_name: The field name to aggregate
        :param sort_order: The sort order of the aggregation. -1: descending, 1: ascending, 0: no sort
        :return: The aggregation result. In the form: [{'count': int, '_id_list': List[str]}]
        """
        # This is a workaround for the fact that montydb does not support aggregation pipeline
        df = pd.DataFrame(self.get_all_images())
        df = df[df[field_name].notna()]
        df_agg = df.groupby(field_name).agg(
            count=(field_name, "size"), _id_list=("_id", lambda x: list(x))
        )
        if sort_order != 0:
            df_agg = df_agg.sort_values(
                "count", ascending=(True if sort_order == 1 else False)
            )
        return df_agg.to_dict("records")

    def query_most_common(self, field_name: str, n: int = 1) -> List[dict]:
        """
        Query the most common values of a field.
        """
        # This is a workaround for the fact that montydb does not support aggregation pipeline
        image_docs_with_value = self.find_images_with_value(field_name)
        df = pd.DataFrame(image_docs_with_value)
        df = df.groupby(field_name).size().sort_values(ascending=False)
        return df.head(n).index.tolist()

    def export_to_csv(self, file_path: str):
        """
        Export the MongoDB database to a CSV file.

        :param file_path: The path of the CSV file to write to.
        """

        df = pd.DataFrame(self.get_all_images())
        _ = df.to_csv(file_path, index=False)

    @staticmethod
    def create_from_csv(
        csv_file_path: str,
        database_id: str = "db",
        mongo_key: str = None,
        mongo_vector_key: str = None,
    ):
        """
        Create a new database from a CSV file.
        :param csv_file_path: The path of the CSV file to import.
        :param database_id: The ID of the database to create.
        :param mongo_key: The MongoDB connection string. if not provided will use local mongo.
        :param mongo_key: The MongoDB connection string for vector database. if not provided will use local chromadb
        """
        db = Database(database_id, mongo_key, mongo_vector_key)
        df = pd.read_csv(csv_file_path)
        for _, row in tqdm(df.iterrows(), desc="Reading CSV file", total=len(df)):
            image_id = row["_id"]
            image_path = row["image_path"]
            _ = db.add_image(image_id, image_path)
            for field_name, field_value in row.items():
                if field_name not in ["_id", "image_path"]:
                    if isinstance(field_value, float) and math.isnan(field_value):
                        # nan's are auto generated for empty values in numpy
                        continue
                    _ = db.store_field(image_id, field_name, field_value)
        return db

    def filter(self, field_name: str, field_value=None):
        """
        Filter out rows from the MongoDB where field_name!=field_value.
        If field_value is None, keep only rows which have field_name no matter what the field_value is.

        :param field_name: The name of the field to filter.
        :param field_value: The value of the field to filter. Default is None.
        """
        if field_value is None:
            _ = self._db.images.delete_many({field_name: {"$exists": False}})
        else:
            _ = self._db.images.delete_many({field_name: {"$ne": field_value}})

    def filter_unidentified_people(
        self,
        is_person_field: str = "is_person",
        identity_field: str = "assigned_identity",
    ):
        _ = self._db.images.delete_many(
            {
                is_person_field: {"$in": ["True", True]},
                identity_field: {"$exists": False},
            }
        )

    def clone_row(self, source_image_id: str, target_image_id: str):
        """
        Clone a row values to another image.

        :param source_image_id: The ID of the source image.
        :param target_image_id: The ID of the target image.
        """
        source_image = self.find_image(source_image_id)
        if source_image is None:
            raise ValueError(
                f"Source image ID {source_image_id} does not exist in the database"
            )

        target_image = self.find_image(target_image_id)
        if target_image is None:
            raise ValueError(
                f"Target image ID {target_image_id} does not exist in the database"
            )

        for field_name, field_value in source_image.items():
            if field_name not in ["_id", "image_path"]:
                _ = self.store_field(target_image_id, field_name, field_value)

    def does_image_have_field(self, image_id: str, field_name: str) -> bool:
        """
        Check if an image has a specific field.

        :param image_id: The ID of the image.
        :param field_name: The name of the field to check.
        :return: True if the image has the field, False otherwise.
        """
        image_doc = self.find_image(image_id)
        return field_name in image_doc
