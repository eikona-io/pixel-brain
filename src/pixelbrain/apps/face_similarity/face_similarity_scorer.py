from pixelbrain.data_loader import DataLoader
from pixelbrain.pipelines.face_similarity_pipeline import FaceSimilarityPipeline
from pixelbrain.data_loaders.cloudinary_dataloader import CloudinaryDataLoader
from pixelbrain.data_loaders.s3_dataloader import S3DataLoader
from pixelbrain.database import Database
from typing import List
from pixelbrain.utils import get_logger
from uuid import uuid4


class FaceSimilartyScorer:
    """
    A class to match faces using images stored in Cloudinary or local directories.

    This class initializes the necessary components to match faces from two different sources,
    using a specified scoring strategy to evaluate the similarity between faces.

    Attributes:
        source_dir (str): The directory for the images to be tested.
        compare_to_dir (str): The directory for the images to compare against.
        source_type (str): The type of source, either 'cloudinary', 's3' or 'local'.
        database (Database): An optional database instance for storing results. If not provided, a new one is created.
        scoring_strategies (List[str]): The strategies to use for scoring the distance between embeddings.
            Options are "nearest", "maximum_distance" or "average_k_nearest". Defaults to ["nearest"].
        score_field_name_prefix (str): The prefix for the score field names. Defaults to "face_similarity_score". This will be followed by the strategy name.
        n_closest_compare_to_to_consider (int): The number of closest vectors to compare to. Defaults to 15.
        k_nearest (int): k nearest vectors to compare if k average_k_nearest strategy is used. Defaults to 5.
    """

    def __init__(
        self,
        source_dir: str,
        compare_to_dir: str,
        database: Database,
        source_type: str = "s3",
        scoring_strategies: List[str] = ["nearest"],
        score_field_name_prefix: str = "face_similarity_score",
        n_closest_compare_to_to_consider: int = 15,
        k_nearest: int = 5,
        s3_bucket: str = None,
    ):
        self._database_created = False
        if not database:
            database = Database()
            self._database_created = True

        if source_type == "cloudinary":
            self._tested_dataloader = CloudinaryDataLoader(source_dir, database)
            self._compare_to_dataloader = CloudinaryDataLoader(compare_to_dir, database)
        elif source_type == "local":
            self._tested_dataloader = DataLoader(source_dir, database, load_images=True)
            self._compare_to_dataloader = DataLoader(
                compare_to_dir, database, load_images=True
            )
        elif source_type == "s3":
            self._tested_dataloader = S3DataLoader(source_dir, s3_bucket ,database)
            self._compare_to_dataloader = S3DataLoader(compare_to_dir, s3_bucket, database)
        else:
            raise ValueError("source_type must be either 'cloudinary', 's3' or 'local'")
        self._source_type = source_type

        self._database = database
        # generate unique field names so we would not compare to vectors generated from another session
        self._tested_embedding_field_name = f"face_embedding_tested_{uuid4().hex[:16]}"
        self._compare_to_embedding_field_name = (
            f"face_embedding_compare_to_{uuid4().hex[:16]}"
        )
        self._matcher = FaceSimilarityPipeline(
            self._tested_dataloader,
            self._compare_to_dataloader,
            database,
            scoring_strategies=scoring_strategies,
            score_field_name_prefix=score_field_name_prefix,
            n_closest_compare_to_to_consider=n_closest_compare_to_to_consider,
            k_nearest=k_nearest,
            tested_embedding_field_name=self._tested_embedding_field_name,
            compare_to_embedding_field_name=self._compare_to_embedding_field_name,
        )
        self._scoring_field_name = score_field_name_prefix
        self._logger = get_logger("CloudinaryFaceSimilartyScorer")

    def process(self) -> List[str]:
        """Processes the images to match faces using the configured pipeline."""
        self._matcher.process()
