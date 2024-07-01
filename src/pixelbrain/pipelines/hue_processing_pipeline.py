from pixelbrain.data_loader import DataLoader, Database, FutureFirstNDataloaderFilter
from pixelbrain.pipeline import TaggingPipeline
from pixelbrain.modules.face_extractor import FaceExtractorModule
from pixelbrain.pipelines.identity_tagging_pipeline import IdentityTaggingPipeline
from pixelbrain.modules.people_identifier import MostCommonIdentityFilter
from pixelbrain.modules.upload import UploadToS3Module
from pixelbrain.data_loaders.s3_dataloader import S3DataLoader
from pixelbrain.utils import create_timestamp
from pixelbrain.modules.image_save import ImageSaveModule
from os.path import join
from os import makedirs
from uuid import uuid4


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
        augmented_images_dir: str,
        s3_bucket: str,
        dataloader: DataLoader,
        database: Database,
        upload_prefix: str,
        increase_face_ratio: int = 2.5,
        assigned_identity_field_name: str = "assigned_identity",
        min_nof_processed_images: int = 15,
        resize_output_size: int = None,
    ):
        """
        Initializes the HueProcessingPipeline with necessary parameters and setups the directory structure.

        Args:
            extracted_faces_results_dir (str): Base directory for storing extracted face results.
            dataloader (DataLoader): The DataLoader to use for loading data.
            upload_prefix (str): The prefix to use for uploaded files on Cloudinary.
            increase_face_ratio (float, optional): The ratio to increase the size of the face bounding box. Defaults to 2.5.
            assigned_identity_field_name (str, optional): The field name for the assigned identity in the database. Defaults to "assigned_identity".
            min_nof_processed_images (int, optional): The minimum number of images to be processed. Defaults to 15. Below that number of images, the images are augmented.
            resize_output_size (int, optional): The size to which the extracted faces will be resized. Defaults to None.
        """
        super().__init__(extracted_faces_results_dir, database=database)

        # Create a timestamped subdirectory to store results
        extracted_faces_results_dir = join(
            extracted_faces_results_dir, create_timestamp()
        )
        makedirs(extracted_faces_results_dir, exist_ok=True)
        augmented_images_dir = join(augmented_images_dir, create_timestamp())
        makedirs(augmented_images_dir, exist_ok=True)

        self._temp_identity_db = Database(database_id=uuid4().hex)
        upload_dataloader = DataLoader(
            extracted_faces_results_dir,
            database=self._temp_identity_db,
            load_images=False,
        )
        augmented_images_dataloader = S3DataLoader(
            upload_prefix,
            s3_bucket,
            self._database,
        )
        augmented_images_upload_dataloader = DataLoader(
            augmented_images_dir,
            database=self._database,
            load_images=False,
        )
        most_common_identity_filter = MostCommonIdentityFilter(
            assigned_identity_field_name
        )
        # we need to fill with augmented images up to min_nof_processed_images
        get_nof_augmented_images_lambda = lambda: max(
            0, min_nof_processed_images - most_common_identity_filter.get_size()
        )
        augmented_images_filter = FutureFirstNDataloaderFilter(
            get_nof_augmented_images_lambda
        )
        # Define the data processors used in the pipeline
        self._data_processors = [
            FaceExtractorModule(
                dataloader,
                self._database,
                image_save_path=extracted_faces_results_dir,
                increase_face_ratio=increase_face_ratio,
                resize_output_size=resize_output_size,
                store_in_db=False,  # Do not store extracted faces metadata in the database
            ),
            IdentityTaggingPipeline(
                extracted_faces_results_dir,
                self._temp_identity_db,
                apply_people_detector=False,
                identity_field_name=assigned_identity_field_name,
            ),
            UploadToS3Module(
                upload_dataloader,
                self._database,
                upload_prefix,
                s3_bucket,
                filters=most_common_identity_filter,
            ),
            ImageSaveModule(
                augmented_images_dataloader,
                self._temp_identity_db,
                augmented_images_dir,
                filters=augmented_images_filter,
            ),
            UploadToS3Module(
                augmented_images_upload_dataloader,
                self._database,
                upload_prefix,
                s3_bucket,
            ),
        ]
