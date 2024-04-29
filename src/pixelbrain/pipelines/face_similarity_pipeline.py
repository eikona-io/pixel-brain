from typing import List
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipeline import TaggingPipeline
from pixelbrain.modules.embedders import FacenetEmbbedderModule
from pixelbrain.modules.embedding_dsitance_scorer import EmbeddingDistanceScorerModule
from overrides import overrides
from uuid import uuid4


class FaceSimilarityPipeline(TaggingPipeline):
    """
    A pipeline for matching faces by comparing embeddings generated from two different data loaders.

    Attributes:
        tested_dataloader (DataLoader): The data loader for the images to be tested.
        compare_to_dataloader (DataLoader): The data loader for the images to compare against.
        database (Database): The database instance for storing and retrieving embeddings.
        scoring_strategy (str): The strategy to use for scoring the distance between embeddings. Defaults to "nearest".
        k_nearest (int): k nearest vectors to compare if k average_k_nearest strategy is used
    """

    def __init__(
        self,
        tested_dataloader: DataLoader,
        compare_to_dataloader: DataLoader,
        database: Database,
        scoring_strategy: str = "nearest",
        score_field_name: str = "face_similarity_score",
        n_closest_compare_to_to_consider: int = 40,
        k_nearest: int = 5,
        tested_embedding_field_name: str = f"face_embedding_tested_{uuid4().hex[:16]}",
        compare_to_embedding_field_name: str = f"face_embedding_compare_to_{uuid4().hex[:16]}",
    ):
        """
        Initializes the FaceMatcherPipeline with the necessary data loaders, database, and scoring strategy.

        Args:
            tested_dataloader (DataLoader): The data loader for the images to be tested.
            compare_to_dataloader (DataLoader): The data loader for the images to compare against.
            database (Database): The database instance for storing and retrieving embeddings.
            scoring_strategy (str): The strategy to use for scoring the distance between embeddings. Defaults to "nearest".
        """
        super().__init__(None, database)

        self._tested_embedding_field_name = tested_embedding_field_name
        self._compare_to_embedding_field_name = compare_to_embedding_field_name
        self._data_processors = [
            FacenetEmbbedderModule(
                tested_dataloader,
                database,
                embedding_field_name=self._tested_embedding_field_name,
            ),
            FacenetEmbbedderModule(
                compare_to_dataloader,
                database,
                embedding_field_name=self._compare_to_embedding_field_name,
            ),
        ]
        self._scorer = EmbeddingDistanceScorerModule(
            database,
            tested_field=self._tested_embedding_field_name,
            compare_to_field=self._compare_to_embedding_field_name,
            strategy=scoring_strategy,
            score_field_name=score_field_name,
            k_nearest=k_nearest,
        )
        self._n_closest_compare_to_to_consider = n_closest_compare_to_to_consider

    @overrides
    def _post_process(self):
        """
        Post-processes the embeddings by scoring the images using the configured scoring strategy.
        """
        self._scorer.score_images(
            n_closest_vectors=self._n_closest_compare_to_to_consider
        )
