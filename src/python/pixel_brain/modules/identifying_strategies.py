from typing import List
import numpy as np
from datetime import datetime
from pixel_brain.database import Database
from abc import ABC, abstractmethod


class IdentifyingStrategy(ABC):
    """
    Base class for identifying strategies.
    """
    @abstractmethod
    def process(self, image_ids: List[str], image_vecs: List[np.ndarray]):
        """
        Process a list of image IDs and corresponding vectors.

        :param image_ids: List of image IDs.
        :param image_vecs: List of corresponding image vectors.
        """
        pass

    def post_process(self):
        """
        Optional post-processing step.
        """
        pass


class PairwiseIdentifyingStrategy(IdentifyingStrategy):
    """
    Identifying strategy that pairs images with the same identity.
    This is a very conservative strategy which only pairs images with distance smaller then the distance threshold.
    If an image has already been assigned an identity (for a case such as im1--im2-----im3), this identity will be adopted for the third image.
    This strategy will not cover images where the distance to other images of same instance is large.
    """
    def __init__(self, database: Database, vector_field_name: str, identity_field_name: str, distance_threshold: int, exclude_group: str = None):
        """
        Initialize the PairwiseIdentifyingStrategy.

        :param database: Database object.
        :param vector_field_name: Name of the vector field in the database.
        :param identity_field_name: Name of the identity field in the database (to store).
        :param distance_threshold: Distance threshold for pairing images. Above it images won't be paired.
                                    This should be stringent threshold so images that pass it are of the same identity for sure.
        :param exclude_group: Optional parameter specifying a field in the image metadata to exclude from pairing. Other images with the same value
                                in the exclude_group field won't be paired with the same identity.
        """
        self._db = database
        self._vector_field_name = vector_field_name
        self._identity_field_name = identity_field_name
        self._distance_threshold = distance_threshold
        self._exclude_group = exclude_group

    def process(self, image_ids: List[str], image_vecs: List[np.ndarray]):
        """
        Process a list of image IDs and corresponding vectors.

        :param image_ids: List of image IDs.
        :param image_vecs: List of corresponding image vectors.
        """

        for image_id, image_vec in zip(image_ids, image_vecs):
            image_doc = self._db.find_image(image_id)
            if self._identity_field_name in image_doc:
                # image was paired
                continue

            nearest_images, distances = self._db.query_vector_field(self._vector_field_name, image_vec, n_results=10)

            if self._exclude_group is not None:
                if self._exclude_group not in image_doc:
                    raise ValueError(f"Exclude group field {self._exclude_group} is used but image {image_id} does not have it")
                self_exclude_group_value = image_doc[self._exclude_group]

            for nearest_image, distance in zip(nearest_images, distances):
                if distance > self._distance_threshold:
                    # image too far away (also the next ones..)
                    break
                if self._exclude_group is not None and nearest_image.get(self._exclude_group, None) == self_exclude_group_value:
                    # image is in the same exclude group
                    continue

                nearest_image_id = nearest_image['_id']
                if nearest_image_id == image_id:
                    # same image
                    continue

                if self._identity_field_name in nearest_image:
                    # nearest image was already paired
                    identity = nearest_image[self._identity_field_name]
                else:
                    # new id required
                    # assign identity using datetime so later assigned identities will have later values
                    identity = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    self._db.store_field(nearest_image_id, self._identity_field_name, identity)
                self._db.store_field(image_id, self._identity_field_name, identity)
                break

class DBSCANIdentifyingStrategy(IdentifyingStrategy):
    def process(self, image_ids: List[str], image_vecs: List[np.ndarray]):
        pass

    def post_process(self):
        pass