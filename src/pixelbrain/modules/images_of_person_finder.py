from pixelbrain.pipeline import PipelineModule, Database, DataLoader
from typing import Dict, List, Union
from pixelbrain.pre_processors.deepface import DeepfacePreprocessor
import requests
from torchvision.io import read_image, ImageReadMode
from torch import Tensor
from deepface import DeepFace
import tempfile
from pixelbrain.utils import get_logger


logger = get_logger(__name__)


class ImagesOfPersonFinder(PipelineModule):
    def __init__(
        self,
        database: Database,
        data_loader: DataLoader,
        path_to_person_image: str,
        matched_person_field_name: str = "matched_person",
        distance_threshold: float = 0.6,
        max_nof_images: int = None,
        filters: Dict[str, str] = None,
    ):
        self._pre_processor = DeepfacePreprocessor()
        super().__init__(data_loader, database, self._pre_processor, filters)
        self._ground_truth_image = self._load_person_image(path_to_person_image)
        self._matched_person_field_name = matched_person_field_name
        self._distance_threshold = distance_threshold
        self._max_nof_images = max_nof_images
        self._found_images = 0

    def _load_person_image(self, path_to_person_image: str):
        if path_to_person_image.startswith(
            "http://"
        ) or path_to_person_image.startswith("https://"):
            with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_file:
                temp_file.write(requests.get(path_to_person_image).content)
                response = temp_file.name
                image = read_image(response, mode=ImageReadMode.RGB)
        else:
            image = read_image(path_to_person_image, mode=ImageReadMode.RGB)

        pre_processed_image = self._pre_processor([image])
        return pre_processed_image[0]

    def _process(
        self,
        image_ids: List[str],
        processed_image_batch: List[Tensor],
    ):
        for image_id, processed_image in zip(image_ids, processed_image_batch):
            if self._max_nof_images and self._found_images >= self._max_nof_images:
                break
            try:
                is_person_dict = DeepFace.verify(
                    img1_path=processed_image.numpy(),
                    img2_path=self._ground_truth_image.numpy(),
                    model_name="Facenet512",
                    detector_backend="retinaface",
                    enforce_detection=False,
                )
                is_person = is_person_dict["distance"] < self._distance_threshold
                self._database.store_field(
                    image_id,
                    self._matched_person_field_name,
                    is_person,
                )
                if is_person:
                    self._found_images += 1
            except ValueError as e:
                # this is a bug in deepface
                if str(e) == "min() arg is an empty sequence":
                    self._database.store_field(
                        image_id,
                        self._matched_person_field_name,
                        False,
                    )
                else:
                    raise
        logger.info(f"Face Similarity: Found {self._found_images} images of the person, out of {len(image_ids)} given images")
