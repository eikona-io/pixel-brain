from pixelbrain.data_loader import DataLoader
from pixelbrain.pipelines.face_similarity_pipeline import FaceSimilarityPipeline
from pixelbrain.data_loaders.cloudinary_dataloader import CloudinaryDataLoader
from pixelbrain.database import Database
from typing import List
from pixelbrain.utils import get_logger


class FaceSimilartyScorer:
    """
    A class to match faces using images stored in Cloudinary or local directories.

    This class initializes the necessary components to match faces from two different sources,
    using a specified scoring strategy to evaluate the similarity between faces.

    Attributes:
        source_dir (str): The directory for the images to be tested.
        compare_to_dir (str): The directory for the images to compare against.
        source_type (str): The type of source, either 'cloudinary' or 'local'.
        database (Database): An optional database instance for storing results. If not provided, a new one is created.
        scoring_strategy (str): The strategy to use for scoring the distance between embeddings. Defaults to "nearest".
        k_nearest (int): k nearest vectors to compare if k average_k_nearest strategy is used
    """

    def __init__(
        self,
        source_dir: str,
        compare_to_dir: str,
        source_type: str = 'cloudinary',
        database: Database = None,
        scoring_strategy: str = "nearest",
        score_field_name: str = "face_similarity_score",
        n_closest_compare_to_to_consider: int = 40,
        k_nearest: int = 5
    ):
        self._database_created = False
        if not database:
            database = Database("cloudinary-face-similarty-scorer")
            self._database_created = True

        if source_type == 'cloudinary':
            self._tested_dataloader = CloudinaryDataLoader(
                source_dir, database
            )
            self._compare_to_dataloader = CloudinaryDataLoader(
                compare_to_dir, database
            )
        elif source_type == 'local':
            self._tested_dataloader = DataLoader(
                source_dir, database, load_images=True
            )
            self._compare_to_dataloader = DataLoader(
                compare_to_dir, database, load_images=True
            )
        else:
            raise ValueError("source_type must be either 'cloudinary' or 'local'")
        self._source_type = source_type

        self._database = database
        self._matcher = FaceSimilarityPipeline(
            self._tested_dataloader,
            self._compare_to_dataloader,
            database,
            scoring_strategy=scoring_strategy,
            score_field_name=score_field_name,
            n_closest_compare_to_to_consider=n_closest_compare_to_to_consider,
            k_nearest=k_nearest
        )
        self._scoring_field_name = score_field_name
        self._logger = get_logger("CloudinaryFaceSimilartyScorer")

    def process(self) -> List[str]:
        """Processes the images to match faces using the configured pipeline."""
        self._matcher.process()
        results_meta = self._database.find_images_with_value(
            self._scoring_field_name,
            value=None,
            sort_by=self._scoring_field_name,
            ascending=True,
        )
        if self._database_created:
            self._database.delete_db()
        
        return [result['_id'] for result in results_meta]