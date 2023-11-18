from pixelbrain.pipeline import PipelineModule
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from typing import List, Dict, Union
from pixelbrain.pre_processors.grounded_sam import GroundedSAMPreprocessor
from lang_sam import LangSAM
import torch


class GroundedSAMDetectorModule(PipelineModule):
    """
    GroundedSAMDetectorModule uses grounded SAM for detecting open set phrases in the photos
    and stores results in database under a specific metadata field.
    """
    def __init__(self, data: DataLoader, 
                 database: Database,
                 detection_string: str,
                 metadata_field_name: str,
                 filters: Dict[str, str] = None):
        """
        Initialize the GroundedSAMDetectorModule.
        
        :param data: DataLoader object with data
        :param database: Database object for storing processed tags
        :param detection_string: The string to detect in the photos
        :param metadata_field_name: The name of the metadata field to store the results
        :param filters: (field_name, field_value) to apply on the dataloader before starting it processing
        """
        super().__init__(data, database, GroundedSAMPreprocessor(), filters)
        self._detection_string = detection_string
        self._metadata_field_name = metadata_field_name
        self._grounded_sam = LangSAM()

    def _process(self, image_ids: List[str], processed_image_batch: List[torch.Tensor]):
        """
        Process the data using grounded sam for detecting the specific string in the photos
        and store the results in the database under the specific metadata field.
        
        :param image_ids: List of image ids
        :param processed_image_batch: Batch of preprocessed images
        """
        
        for img_id, image in zip(image_ids, processed_image_batch):
            masks, boxes, phrases, logits = self._grounded_sam.predict(image, self._detection_string)
            detection = str(any(phrase == self._detection_string for phrase in phrases))
            self._database.store_field(img_id, self._metadata_field_name, detection)


