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
    """

    def __init__(
        self,
        tested_dataloader: DataLoader,
        compare_to_dataloader: DataLoader,
        database: Database,
        scoring_strategies: List[str] = ["nearest"],
        score_field_name_prefix: str = "face_similarity_score",
        n_closest_compare_to_to_consider: int = 15,
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
            scoring_strategies (List[str]): The strategies to use for scoring the distance between embeddings.
                                            Options are "nearest", "maximum_distance" or "average_k_nearest". Defaults to ["nearest"].
            score_field_name_prefix (str): The prefix for the score field names. Defaults to "face_similarity_score". This will be followed by the strategy name.
            n_closest_compare_to_to_consider (int): The number of closest vectors to compare if k average_k_nearest strategy is used. Defaults to 15.
            k_nearest (int): k nearest vectors to compare if k average_k_nearest strategy is used. Defaults to 5.
            tested_embedding_field_name (str): The name of the field in the database to store the tested embeddings. Defaults to a random UUID.
            compare_to_embedding_field_name (str): The name of the field in the database to store the compare to embeddings. Defaults to a random UUID.
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
        self._scorers = [
            EmbeddingDistanceScorerModule(
                database,
                tested_field=self._tested_embedding_field_name,
                compare_to_field=self._compare_to_embedding_field_name,
                strategy=strategy,
                score_field_name=f"{score_field_name_prefix}_{strategy}",
                k_nearest=k_nearest,
            )
            for strategy in scoring_strategies
        ]
        self._n_closest_compare_to_to_consider = n_closest_compare_to_to_consider

    @overrides
    def _post_process(self):
        """
        Post-processes the embeddings by scoring the images using the configured scoring strategy.
        """
        for scorer in self._scorers:
            scorer.score_images(
                n_closest_vectors=self._n_closest_compare_to_to_consider
            )
