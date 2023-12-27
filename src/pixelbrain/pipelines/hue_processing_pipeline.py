from typing import List
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipeline import TaggingPipeline
from pixelbrain.modules.face_extractor import FaceExtractorModule


class HueProcessingPipeline(TaggingPipeline):
    def __init__(self, images_path: str, 
                 database: Database):
        super().__init__(images_path, database)
        
        # might parameterize the batch sizes in the future
        face_extractor_data = DataLoader(images_path, self._database, batch_size=1, decode_images=True)

        self._data_processors = [FaceExtractorModule(face_extractor_data, database, h_ratio=0.4, w_ratio=0.4)]