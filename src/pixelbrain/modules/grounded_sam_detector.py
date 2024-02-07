from pixelbrain.pipeline import PipelineModule
import torchvision.transforms as transforms
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from typing import List, Dict, Union
from pixelbrain.pre_processors.grounded_sam import GroundedSAMPreprocessor
from PIL import Image
from lang_sam import LangSAM
import numpy as np
import torch
from os.path import join, isdir
from os import makedirs


class SAMResult:
    def __init__(self, image_id: str, image: Image.Image, masks: torch.Tensor, boxes: torch.Tensor, phrases: List[str]):
        """
        A struct object to store the result of the grounded sam detector.
        :param image_id: The id of the image
        :param image: The original image as a PIL Image object
        :param masks: The masks of the detected phrases of shape (num_masks, H, W)
        :param boxes: The bounding boxes of the detected phrases of shape (num_masks, 4)
        :param phrases: The detected phrases
        """
        self.image_id = image_id
        self.image = image
        self.masks = masks
        self.boxes = boxes
        self.phrases = phrases


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
        self.sam_results = []

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
            self.sam_results.append(SAMResult(img_id, image, masks, boxes, phrases))

            
    def _post_process(self, results_dir = '/tmp'):
        """
        Post process the results.
        """
        for ratio in ['closeup', 'medium', 'wide']:
            makedirs(join(results_dir, 'wo_background', ratio), exist_ok=True)
        transform = transforms.Compose([transforms.PILToTensor()])
        for result in self.sam_results:
            img_id = result.image_id
            image = result.image
            image_tensor = transform(image)
            masks = result.masks
            boxes = result.boxes
            image_surface = image_tensor.shape[-1] * image_tensor.shape[-2]
            for idx in range(masks.shape[0]):
                box_surface = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
                ratio_num = box_surface / image_surface
                if ratio_num > 0.7:
                    ratio = 'closeup'
                elif ratio_num > 0.3:
                    ratio = 'medium'
                else:
                    ratio = 'wide'
                
                masked_img = image_tensor * masks[idx]
                masked_image = Image.fromarray(np.transpose(masked_img.numpy(), (2, 1, 0)))
                masked_image.rotate(-90, expand=True).save(join(results_dir, 'wo_background', ratio, img_id + f'_{idx}.jpg'))
