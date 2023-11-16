from typing import List, Dict
import torch
from pixel_brain.database import Database
from pixel_brain.data_loader import DataLoader
from pixel_brain.pipeline import PipelineModule
from pixel_brain.modules.identifying_strategies import PairwiseIdentifyingStrategy, HDBSCANIdentifyingStrategy


class PeopleIdentifierModule(PipelineModule):
    """
    Module for identifying people in images.
    """
    def __init__(self, data: DataLoader,
                 database: Database,
                 vector_field_name: str,
                 identity_field_name: str = 'identity',
                 strategy: str = 'pairwise',
                 distance_threshold: int = 290,
                 pairwise_exclude_group: str = None,
                 filters: Dict[str, str] = None):
        """
        Initialize the PeopleIdentifierModule.

        :param data: DataLoader object
        :param database: Database object
        :param vector_field_name: Name of the field in the database where the vector is stored
        :param identity_field_name: Name of the field in the database where to store the identity
        :param strategy: Strategy for identifying people, either 'pairwise' or 'hdbscan'
        :param distance_threshold: Threshold for the distance between vectors for identification, relevant only in 'pairwise' strategy
        :param pairwise_exclude_group: Group field to exclude images in pairwise identification, relevant only in 'pairwise' strategy
        """
        super().__init__(data, database, None, filters)
        self._vector_field_name = vector_field_name
        if strategy == 'pairwise':
            self._identify_strategy = PairwiseIdentifyingStrategy(database, vector_field_name, identity_field_name, distance_threshold, pairwise_exclude_group)
        elif strategy == 'hdbscan':
            self._identify_strategy = HDBSCANIdentifyingStrategy(database, identity_field_name)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _process(self, image_ids: List[str], processed_image_batch: List[torch.Tensor]):
        """
        Process a batch of images.
        It is assumed that an embedding module as already used to embed and store vectors in vector_field_name when this module is called.

        :param image_ids: List of image IDs
        :param processed_image_batch: List of processed image tensors
        """
        image_vecs = [self._database.get_field(image_id, self._vector_field_name) for image_id in image_ids]
        self._identify_strategy.process(image_ids, image_vecs)

    def _post_process(self):
        """
        Post-process after all batches have been processed.
        """
        self._identify_strategy.post_process()

