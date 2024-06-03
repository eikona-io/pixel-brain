from typing import List, Dict
import torch
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipeline import PipelineModule
from deepface import DeepFace
import torch
from pixelbrain.pre_processors.deepface import DeepfacePreprocessor
from PIL import Image
import os
import numpy as np
from pixelbrain.utils import increase_rectangle_space, get_logger


logger = get_logger(__name__)


class FaceExtractorModule(PipelineModule):
    """
    This module is responsible for extracting faces from images.
    The extracted faces are stored on disk and their metadata saved in the database.
    """

    def __init__(
        self,
        data: DataLoader,
        database: Database,
        filters: Dict[str, str] = None,
        h_ratio=0.6,
        w_ratio=0.6,
        image_save_path: str = None,
        clone_original_image_metadata=True,
        increase_face_ratio: float = 2.5,
        store_in_db: bool = True,
        minimal_face_size: int = 200 * 300,
    ):
        """
        Initialize the FaceExtractorModule.

        :param data: DataLoader object
        :param database: Database object
        :param filters: Dictionary of filters to apply on data before loading
        :param h_ratio: Height ratio relative to image height to keep after face extraction
        :param w_ratio: Width ratio relative to image width to keep after face extraction
        :param image_save_path: Path to save the extracted face images. If None then will use the path of the original image.
        :param clone_original_image_metadata: Boolean to decide if original image metadata should be cloned.
        :param store_in_db: Boolean to decide if the extracted faces should be stored in the database.
        """
        super().__init__(data, database, DeepfacePreprocessor(), filters)
        self._h_ratio = h_ratio
        self._w_ratio = w_ratio
        self._save_image_path = image_save_path
        self._clone_metadata = clone_original_image_metadata
        self._increase_face_ratio = increase_face_ratio
        self._store_in_db = store_in_db
        self._minimal_face_size = minimal_face_size

    def _process(self, image_ids: List[str], processed_image_batch: List[torch.Tensor]):
        """
        Process the images by extracting the face and storing it in the database.

        :param image_ids: List of image ids
        :param processed_image_batch: Batch of preprocessed images
        """
        # TODO: Move to a model that can take batches (both SAM and DeepFace can't)
        for image_id, image in zip(image_ids, processed_image_batch):
            image = image.numpy()
            extracted_faces = DeepFace.extract_faces(
                image, detector_backend="retinaface", enforce_detection=False
            )

            faces = [
                extracted_face["facial_area"] for extracted_face in extracted_faces
            ]
            for idx, extracted_face in enumerate(extracted_faces):
                detected_face = extracted_face["facial_area"]
                x, y, w, h, _, _ = detected_face.values()
                if (
                    (x == 0)
                    and (y == 0)
                    and (h == image.shape[0])
                    and (w == image.shape[1])
                ):
                    # This means he didn't find a face
                    logger.info(f"No face detected in image {image_id}")
                    continue

                masked_image = self.mask_faces(image, faces, idx_of_face_to_keep=idx)
                face_frame = self._get_face_frame(
                    masked_image, detected_face, ratio=self._increase_face_ratio
                )
                if (face_frame.shape[0] * face_frame.shape[1]) < self._minimal_face_size:
                    logger.info(
                        f"Face in image {image_id} is too small to extract. Skipping."
                    )
                    continue
                face_image_path = self._save_image(face_frame, image_id, idx)

                # add to db
                if self._store_in_db:
                    new_image_id = f"{image_id}_face{idx}"
                    self._database.add_image(new_image_id, face_image_path)
                    if self._clone_metadata:
                        self._database.clone_row(image_id, new_image_id)
                    self._database.store_field(new_image_id, "is_augmented_face", "True")

    def _save_image(
        self, extracted_face: np.ndarray, original_image_id: str, idx: int
    ) -> str:
        """
        Save the extracted face image.

        :param extracted_face: Extracted face as a numpy array
        :param original_image_id: ID of the original image
        :param idx: The index of the face in the image
        :return: Path of the saved face image
        """
        image_path = self._database.get_field(original_image_id, "image_path")
        image_dir = (
            self._save_image_path
            if self._save_image_path
            else os.path.dirname(image_path)
        )
        image_filename = os.path.basename(image_path).split(".")[0]
        face_path = f"{image_dir}/{image_filename}_face{idx}.png"

        extracted_face_pil = Image.fromarray(extracted_face.astype("uint8"), "RGB")
        extracted_face_pil.save(face_path)
        return face_path

    def _get_face_frame(
        self, image: np.ndarray, detected_face: Dict[str, int], ratio: float = 2.0
    ) -> np.ndarray:
        """
        Extract the face from the image.

        :param image: Image as a torch tensor
        :param detected_face: Detected face as a dictionary
        :return: Extracted face as a numpy array
        """
        x, y, w, h, _, _ = detected_face.values()
        scaled_x, scaled_y, scaled_w, scaled_h = increase_rectangle_space(
            x, y, w, h, ratio
        )
        face_image = image[
            scaled_y : scaled_y + scaled_h, scaled_x : scaled_x + scaled_w, :
        ]
        return face_image

    @staticmethod
    def mask_faces(
        image: np.ndarray, faces: List[Dict[str, int]], idx_of_face_to_keep: int = -1
    ) -> np.ndarray:
        """
        Masks all give faces from an image except the face at the given index.
        """
        masked_image = image.copy()
        for idx, face in enumerate(faces):
            if idx == idx_of_face_to_keep:
                continue
            x, y, w, h, _, _ = face.values()
            mask = np.ones_like(image)
            mask[y : y + h, x : x + w] = 0
            masked_image *= mask
        return masked_image
