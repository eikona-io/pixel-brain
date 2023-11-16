from typing import List
from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
from pixel_brain.pipeline import TaggingPipeline
from pixel_brain.modules.embedders import FacenetEmbbedderModule
from pixel_brain.modules.people_identifier import PeopleIdentifierModule
from pixel_brain.modules.gpt4v import GPT4VPeopleDetectorModule


class FashionTaggingPipeline(TaggingPipeline):
    def __init__(self, images_path: str, database: Database, pairwise_exclude_group: str = None):
        super().__init__(images_path, database)
        
        # might parameterize the batch sizes in the future
        people_detector_data = DataLoader(images_path, self._database, batch_size=1, decode_images=False)
        embedder_data = DataLoader(images_path, self._database, batch_size=1)
        people_identifier_data = DataLoader(images_path, self._database, batch_size=1, decode_images=False)
        
        embedding_filters = {"is_person": "True"} # process only images with people
        identify_filters = {"face_embedding": None} # process only images with faces
        self._modules = [
            GPT4VPeopleDetectorModule(people_detector_data, self._database),
            FacenetEmbbedderModule(embedder_data, self._database, embedding_filters),
            PeopleIdentifierModule(people_identifier_data, 
                                   self._database,
                                   filters=identify_filters,
                                   vector_field_name='face_embedding',
                                   identity_field_name='assigned_identity',
                                   pairwise_exclude_group=pairwise_exclude_group)
        ]