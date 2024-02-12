from typing import List
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipeline import TaggingPipeline
from pixelbrain.modules.embedders import FacenetEmbbedderModule
from pixelbrain.modules.embedding_dsitance_scorer import EmbeddingDistanceScorerModule
from overrides import overrides


class FaceSimilarityPipeline(TaggingPipeline):
    """
    A pipeline for matching faces by comparing embeddings generated from two different data loaders.
    
    Attributes:
        tested_dataloader (DataLoader): The data loader for the images to be tested.
        compare_to_dataloader (DataLoader): The data loader for the images to compare against.
        database (Database): The database instance for storing and retrieving embeddings.
        scoring_strategy (str): The strategy to use for scoring the distance between embeddings. Defaults to "nearest".
    """
    def __init__(
        self,
        tested_dataloader: DataLoader,
        compare_to_dataloader: DataLoader,
        database: Database,
        scoring_strategy: str = "nearest",
        score_field_name: str = "face_similarity_score",
        n_closest_compare_to_to_consider: int = 40
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

        self._data_processors = [
            FacenetEmbbedderModule(
                tested_dataloader,
                database,
                embedding_field_name="face_embedding_tested",
            ),
            FacenetEmbbedderModule(
                compare_to_dataloader,
                database,
                embedding_field_name="face_embedding_compare_to",
            ),
        ]
        self._scorer = EmbeddingDistanceScorerModule(
            database,
            tested_field="face_embedding_tested",
            compare_to_field="face_embedding_compare_to",
            strategy=scoring_strategy,
            score_field_name=score_field_name
        )
        self._n_closest_compare_to_to_consider = n_closest_compare_to_to_consider
    
    @overrides    
    def _post_process(self):
        """
        Post-processes the embeddings by scoring the images using the configured scoring strategy.
        """
        self._scorer.score_images(n_closest_vectors=self._n_closest_compare_to_to_consider)
