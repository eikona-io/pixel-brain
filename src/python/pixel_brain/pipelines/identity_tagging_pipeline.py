from typing import List
from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
from pixel_brain.pipeline import TaggingPipeline
from pixel_brain.modules.embedders import FacenetEmbbedderModule
from pixel_brain.modules.people_identifier import PeopleIdentifierModule
from pixel_brain.modules.gpt4v import GPT4VPeopleDetectorModule


class IdentityTaggingPipeline(TaggingPipeline):
    def __init__(self, images_path: str, 
                 database: Database,
                 identity_field_name: str = 'assigned_identity',
                 apply_people_detector: bool = True,
                 identifying_strategy: str = 'pairwise',
                 pairwise_exclude_group: str = None):
        super().__init__(images_path, database)
        
        # might parameterize the batch sizes in the future
        people_detector_data = DataLoader(images_path, self._database, batch_size=1, decode_images=False)
        embedder_data = DataLoader(images_path, self._database, batch_size=1)
        people_identifier_data = DataLoader(images_path, self._database, batch_size=1, decode_images=False)
        
        # apply person filter only if people detector is enabled
        embedding_filters = {"is_person": "True"} if apply_people_detector else None
        identify_filters = {"face_embedding": None} # process only images with faces

        # optional module
        self._modules = [GPT4VPeopleDetectorModule(people_detector_data, self._database)] if apply_people_detector else []
        # constant modules
        self._modules.extend([
            FacenetEmbbedderModule(embedder_data, self._database, embedding_filters),
            PeopleIdentifierModule(people_identifier_data, 
                                   self._database,
                                   filters=identify_filters,
                                   vector_field_name='face_embedding',
                                   identity_field_name=identity_field_name,
                                   pairwise_exclude_group=pairwise_exclude_group,
                                   strategy=identifying_strategy)
        ])