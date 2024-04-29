from typing import List, Dict
import torch
from pixelbrain.database import Database
from pixelbrain.data_loader import DataLoader, DataLoaderFilter
from pixelbrain.pipeline import PipelineModule
from pixelbrain.modules.identifying_strategies import (
    PairwiseIdentifyingStrategy,
    HDBSCANIdentifyingStrategy,
)


class PeopleIdentifierModule(PipelineModule):
    """
    Module for identifying people in images.
    """
    def __init__(self, 
                 data: DataLoader,
                 database: Database,
                 vector_field_name: str,
                 identity_field_name: str = 'identity',
                 strategy: str = 'hdbscan',
                 filters: Dict[str, str] = None,
                 **algo_params):
        """
        Initialize the PeopleIdentifierModule.

        :param data: DataLoader object
        :param database: Database object
        :param vector_field_name: Name of the field in the database where the vector is stored
        :param identity_field_name: Name of the field in the database where to store the identity
        :param strategy: Strategy for identifying people, either 'pairwise' or 'hdbscan'
        :param algo_params: Additional parameters for the strategy. For HDBSCAN primarly min_samples and min_cluster_size, for pairwise exclude_group and distance_threshold (290).
        """
        super().__init__(data, database, None, filters)
        self._vector_field_name = vector_field_name
        if strategy == 'pairwise':
            self._identify_strategy = PairwiseIdentifyingStrategy(database, vector_field_name, identity_field_name, **algo_params)
        elif strategy == 'hdbscan':
            self._identify_strategy = HDBSCANIdentifyingStrategy(database, identity_field_name, **algo_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _process(self, image_ids: List[str], processed_image_batch: List[torch.Tensor]):
        """
        Process a batch of images.
        It is assumed that an embedding module as already used to embed and store vectors in vector_field_name when this module is called.

        :param image_ids: List of image IDs
        :param processed_image_batch: List of processed image tensors
        """
        if not all(
            [
                self._database.does_image_have_field(image_id, self._vector_field_name)
                for image_id in image_ids
            ]
        ):
            # Not a face
            return
        image_vecs = [
            self._database.get_field(image_id, self._vector_field_name)
            for image_id in image_ids
        ]
        self._identify_strategy.process(image_ids, image_vecs)

    def _post_process(self):
        """
        Post-process after all batches have been processed.
        """
        self._identify_strategy.post_process()


class MostCommonIdentityFilter(DataLoaderFilter):
    def __init__(self, identity_field_name: str):
        self._identity_field_name = identity_field_name

    def _find_most_common_identity(self, database: Database):
        """
        Retrieves the most common identity from the database by aggregating and sorting the identity counts.
        """
        return database.query_most_common(self._identity_field_name, n=1)[0]

    def filter(self, database: Database, image_ids: List[str]) -> List[str]:
        """
        Filters out images that do not have the most common identity.
        """
        most_common_identity = self._find_most_common_identity(database)
        filtered_image_ids = []
        for image_id in image_ids:
            image_data = database.find_image(image_id)
            if (
                self._identity_field_name in image_data
                and image_data[self._identity_field_name] == most_common_identity
            ):
                filtered_image_ids.append(image_id)
        return filtered_image_ids
