from pixelbrain.pipelines.face_similarity_pipeline import FaceSimilarityPipeline
from pixelbrain.data_loaders.cloudinary_dataloader import CloudinaryDataLoader
from pixelbrain.database import Database
from typing import List

class CloudinaryFaceSimilartyScorer:
    """
    A class to match faces using images stored in Cloudinary.

    This class initializes the necessary components to match faces from two different Cloudinary prefixes,
    using a specified scoring strategy to evaluate the similarity between faces.

    Attributes:
        cloudinary_tested_prefix (str): The Cloudinary prefix for the images to be tested.
        cloudinary_compare_to_prefix (str): The Cloudinary prefix for the images to compare against.
        database (Database): An optional database instance for storing results. If not provided, a new one is created.
        scoring_strategy (str): The strategy to use for scoring the distance between embeddings. Defaults to "nearest".
    """

    def __init__(
        self,
        cloudinary_tested_prefix: str,
        cloudinary_compare_to_prefix: str,
        database: Database = None,
        scoring_strategy: str = "nearest",
        score_field_name: str = "face_similarity_score",
        n_closest_compare_to_to_consider: int = 40
    ):
        self._database_created = False
        if not database:
            database = Database("cloudinary-face-similarty-scorer")
            self._database_created = True

        self._cloudinary_tested_dataloader = CloudinaryDataLoader(
            cloudinary_tested_prefix, database
        )
        self._cloudinary_compare_to_dataloader = CloudinaryDataLoader(
            cloudinary_compare_to_prefix, database
        )
        self._database = database
        self._matcher = FaceSimilarityPipeline(
            self._cloudinary_tested_dataloader,
            self._cloudinary_compare_to_dataloader,
            database,
            scoring_strategy=scoring_strategy,
            score_field_name=score_field_name,
            n_closest_compare_to_to_consider=n_closest_compare_to_to_consider
        )
        self._scoring_field_name = score_field_name

    def process(self) -> List[str]:
        """Processes the images to match faces using the configured pipeline."""
        self._matcher.process()
        results_meta = self._database.find_images_with_value(
            self._scoring_field_name,
            value=None,
            sort_by=self._scoring_field_name,
            ascending=True,
        )
        print(results_meta)
        print(self._database.get_all_images())
        # if self._database_created:
        #     self._database.delete_db()
        
        return [result['cloudinary_public_id'] for result in results_meta]

# i = CloudinaryFaceSimilartyScorer("user_images/generated_images/demo/1707663552586", "user_photos/1707663552586").process()
# print(i)

