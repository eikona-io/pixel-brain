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


class FaceExtractorModule(PipelineModule):
    """
    This module is responsible for extracting faces from images.
    The extracted faces are stored on disk and their metadata saved in the database.
    """
    def __init__(self, data: DataLoader,
                 database: Database,
                 filters: Dict[str, str] = None,
                 h_ratio = 0.6,
                 w_ratio = 0.6,
                 image_save_path: str = None,
                 clone_original_image_metadata = True):
        """
        Initialize the FaceExtractorModule.
        
        :param data: DataLoader object
        :param database: Database object
        :param filters: Dictionary of filters to apply on data before loading
        :param h_ratio: Height ratio relative to image height to keep after face extraction
        :param w_ratio: Width ratio relative to image width to keep after face extraction
        :param image_save_path: Path to save the extracted face images. If None then will use the path of the original image.
        :param clone_original_image_metadata: Boolean to decide if original image metadata should be cloned.
        """
        super().__init__(data, database, DeepfacePreprocessor(), filters)
        self._h_ratio = h_ratio
        self._w_ratio = w_ratio
        self._save_image_path = image_save_path
        self._clone_metadata = clone_original_image_metadata

    def _process(self, image_ids: List[str], processed_image_batch: List[torch.Tensor]):
        """
        Process the images by extracting the face and storing it in the database.
        
        :param image_ids: List of image ids
        :param processed_image_batch: Batch of preprocessed images
        """
        for image_id, image in zip(image_ids, processed_image_batch):
            try:
                extracted_face = DeepFace.extract_faces(image.numpy(),
                                                        detector_backend='retinaface',
                                                        enforce_detection=True)

                if len(extracted_face) != 1:
                    # more then one face
                    # we don't want this image
                    break

                detected_face = extracted_face[0]['facial_area']
                extracted_face = self._extract_face(image, detected_face)
                face_image_path = self._save_image(extracted_face, image_id)

                # add to db
                new_image_id = f'{image_id}_face'
                self._database.add_image(new_image_id, face_image_path)
                if self._clone_metadata:
                    self._database.clone_row(image_id, new_image_id)
                self._database.store_field(new_image_id, 'is_augmented_face', 'True')

            except Exception as err:
                # no face detetectad raises an error
                pass

    def _save_image(self, extracted_face: np.ndarray, original_image_id: str) -> str:
        """
        Save the extracted face image.
        
        :param extracted_face: Extracted face as a numpy array
        :param original_image_id: ID of the original image
        :return: Path of the saved face image
        """
        image_path = self._database.get_field(original_image_id, 'image_path')
        image_dir = self._save_image_path if self._save_image_path else  os.path.dirname(image_path)
        image_filename = os.path.basename(image_path).split(".")[0]
        face_path = f"{image_dir}/{image_filename}_face.png"

        extracted_face_pil = Image.fromarray(extracted_face.astype('uint8'), 'RGB')
        extracted_face_pil.save(face_path)
        return face_path

    def _extract_face(self, image: torch.Tensor, detected_face: Dict[str, int]) -> np.ndarray:
        """
        Extract the face from the image.
        
        :param image: Image as a torch tensor
        :param detected_face: Detected face as a dictionary
        :return: Extracted face as a numpy array
        """
        x, y, w, h = detected_face.values()
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        image_h, image_w, _ = image.shape
        target_image_h = int(image_h * self._h_ratio)
        target_image_w = int(image_w * self._w_ratio)
        h_offset_from_face_center = target_image_h / 2
        w_offset_from_face_center = target_image_w / 2

        start_x = int(max(face_center_x - w_offset_from_face_center, 0))
        start_y = int(max(face_center_y - h_offset_from_face_center, 0))
        end_x = int(min(face_center_x + w_offset_from_face_center, image_w))
        end_y = int(min(face_center_y + h_offset_from_face_center, image_h))
        face_image = image[start_y:end_y, start_x:end_x, :].numpy()
        return face_image
