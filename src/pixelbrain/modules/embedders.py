from typing import List, Dict
import torch
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipeline import PipelineModule
from deepface import DeepFace
import torch
import numpy as np
from pixelbrain.pre_processors.deepface import DeepfacePreprocessor


class FacenetEmbbedderModule(PipelineModule):
    """
    This module is responsible for embedding images using the Facenet model.
    Embeddings are stored in face_embedding field in database.
    Before embedding, the face is detected by Retinaface and extracted from the image.
    This is a very naive implementation using deepface library. it can be greatly improved.
    """

    def __init__(
        self,
        data: DataLoader,
        database: Database,
        filters: Dict[str, str] = None,
        embedding_field_name="face_embedding",
    ):
        super().__init__(data, database, DeepfacePreprocessor(), filters)
        self._embedding_field_name = embedding_field_name

    def _process(self, image_ids: List[str], processed_image_batch: List[torch.Tensor]):
        """
        Process the images by extracting the face and embedding it using the Facenet model.
        The resulting embeddings are stored in the database.

        :param image_ids: List of image ids
        :param processed_image_batch: Batch of preprocessed images
        """
        for image_id, image in zip(image_ids, processed_image_batch):
            try:
                face_embedding = DeepFace.represent(
                    image.numpy(),
                    model_name="Facenet512",
                    detector_backend="retinaface",
                    enforce_detection=True,
                )
                if len(face_embedding) != 1:
                    # more then one face
                    # we don't want this image
                    break
                self._database.store_field(
                    image_id,
                    self._embedding_field_name,
                    np.array(face_embedding[0]["embedding"]),
                )
            except Exception as err:
                # no face detetectad raises an error
                pass
