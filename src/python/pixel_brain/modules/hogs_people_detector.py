import cv2
from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
from pixel_brain.pipeline import PipelineModule
from pixel_brain.pre_processors.hog_detector import HogsDetectorPreprocessor
from typing import List


class HogsPeopleDetectorModule(PipelineModule):
    def __init__(self, data: DataLoader, database: Database):
        super().__init__(data, database, HogsDetectorPreprocessor())
        self._detector = cv2.HOGDescriptor()
        self._detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _process(self, image_ids: List[str], processed_image_batch):
        for image_id, image in zip(image_ids, processed_image_batch):
            image = image.numpy()  # Convert torch.Tensor to numpy array for cv2
            boxes, weights = self._detector.detectMultiScale(image, winStride=(8, 8))
            is_person = len(boxes) > 0
            self._database.store_field(image_id, 'is_person', is_person)

