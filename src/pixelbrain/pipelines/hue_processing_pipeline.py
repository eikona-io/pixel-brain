from pixelbrain.data_loader import DataLoader
from pixelbrain.pipeline import TaggingPipeline
from pixelbrain.modules.face_extractor import FaceExtractorModule
from pixelbrain.pipelines.identity_tagging_pipeline import IdentityTaggingPipeline
from pixelbrain.modules.upload_to_cloudinary import UploadToCloudinaryModule
from pixelbrain.utils import create_timestamp
from os.path import join
from os import makedirs


class HueProcessingPipeline(TaggingPipeline):
    """
    Attributes:
        extracted_faces_results_dir (str): Directory where the results of the face extraction are stored.
        dataloader (DataLoader): DataLoader instance to load data.
        upload_prefix (str): Prefix for the uploaded files in Cloudinary.
        increase_face_ratio (float): Ratio by which the face detection bounding box is increased.
        assigned_identity_field_name (str): Field name in the database where the assigned identity is stored.
    """

    def __init__(
        self,
        extracted_faces_results_dir: str,
        dataloader: DataLoader,
        upload_prefix: str,
        increase_face_ratio: int = 2.5,
        assigned_identity_field_name: str = "assigned_identity",
    ):
        """
        Initializes the HueProcessingPipeline with necessary parameters and setups the directory structure.

        Args:
            extracted_faces_results_dir (str): Base directory for storing extracted face results.
            dataloader (DataLoader): The DataLoader to use for loading data.
            upload_prefix (str): The prefix to use for uploaded files on Cloudinary.
            increase_face_ratio (float, optional): The ratio to increase the size of the face bounding box. Defaults to 2.5.
            assigned_identity_field_name (str, optional): The field name for the assigned identity in the database. Defaults to "assigned_identity".
        """
        super().__init__(extracted_faces_results_dir, None)

        # Create a timestamped subdirectory to store results
        extracted_faces_results_dir = join(
            extracted_faces_results_dir, create_timestamp()
        )
        makedirs(extracted_faces_results_dir, exist_ok=True)

        # Define the data processors used in the pipeline
        self._data_processors = [
            FaceExtractorModule(
                dataloader,
                dataloader._database,
                image_save_path=extracted_faces_results_dir,
                increase_face_ratio=increase_face_ratio,
            ),
            IdentityTaggingPipeline(
                extracted_faces_results_dir,
                dataloader._database,
                apply_people_detector=False,
                identity_field_name=assigned_identity_field_name,
            ),
            UploadToCloudinaryModule(
                extracted_faces_results_dir,
                dataloader._database,
                upload_prefix,
                filtering_field_name=assigned_identity_field_name,
            ),
        ]
