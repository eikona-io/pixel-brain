from pixelbrain.modules.images_of_person_finder import ImagesOfPersonFinder
from pixelbrain.modules.upload import UploadToCloudinaryModule
from pixelbrain.pipeline import TaggingPipeline, Database, DataLoader
from pixelbrain.modules.ssim_duplicate_filter import SSIMDuplicateFilter
from typing import Union, List


class UploadImagesOfPersonPipeline(TaggingPipeline):
    """
    A pipeline to de-duplicate and upload images of a specific person to Cloudinary.

    Attributes:
        raw_images_path_or_paths (Union[str, List[str]]): Path or list of paths where raw images are stored.
        database (Database): Database instance for storing and retrieving image metadata.
        upload_prefix (str): Prefix for the uploaded images' paths in Cloudinary.
        person_image_path (str, optional): Path to the image of the person to find in other images.
                        If supplied, the pipeline will also apply the person-finder module to filter only the images of the person.
                        Otherwise, all (not duplicated) images will be uploaded to Cloudinary.
        distance_threshold (float, optional): Threshold for the distance metric in face comparison.
        max_results (int, optional): Maximum number of images to return.
        matched_person_field (str, optional): Database field name to store matched person results.
        ssim_threshold (float, optional): Threshold for structural similarity index measure.
        ssim_max_image_size (int, optional): Maximum size of images for SSIM processing.
        ssim_is_duplicate_field (str, optional): Database field name to store SSIM duplicate results.
    """

    def __init__(
        self,
        raw_images_path_or_paths: Union[str, List[str]],
        database: Database,
        upload_prefix: str,
        person_image_path: str = None,
        distance_threshold: float = 0.6,
        max_results: int = 30,
        matched_person_field: str = "matched_person",
        ssim_threshold: float = 0.5,
        ssim_max_image_size: int = 1000,
        ssim_is_duplicate_field: str = "is_duplicate",
    ):
        super().__init__(database=database)
        self._temp_db = Database()
        find_data_loader = DataLoader(raw_images_path_or_paths, database=self._temp_db)
        upload_dataloader = DataLoader(
            raw_images_path_or_paths, database=self._temp_db, load_images=False
        )
        self._matched_person_field = matched_person_field
        self._data_processors = (
            [
                ImagesOfPersonFinder(
                    self._temp_db,
                    find_data_loader,
                    person_image_path,
                    matched_person_field_name=self._matched_person_field,
                    distance_threshold=distance_threshold,
                    max_nof_images=max_results,
                )
            ]
            if person_image_path is not None
            else []  # only apply person-finder if person_image_path is supplied
        )
        self._data_processors.extend(
            [
                SSIMDuplicateFilter(
                    find_data_loader,
                    self._temp_db,
                    ssim_threshold=ssim_threshold,
                    max_image_size=ssim_max_image_size,
                    is_duplicate_field_name=ssim_is_duplicate_field,
                    filters=(
                        {self._matched_person_field: True}
                        if person_image_path
                        else None
                    ),
                ),
                UploadToCloudinaryModule(
                    upload_dataloader,
                    self._database,
                    upload_prefix=upload_prefix,
                    filters=(
                        {
                            self._matched_person_field: True,
                            ssim_is_duplicate_field: False,
                        }
                        if person_image_path
                        else {
                            ssim_is_duplicate_field: False,
                        }
                    ),
                ),
            ]
        )
