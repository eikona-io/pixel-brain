from pixelbrain.modules.images_of_person_finder import ImagesOfPersonFinder
from pixelbrain.modules.upload_to_cloudinary import UploadToCloudinaryModule
from pixelbrain.pipeline import TaggingPipeline, Database, DataLoader
from pixelbrain.modules.ssim_duplicate_filter import SSIMDuplicateFilter
from typing import Union, List


class FindImagesOfPersonPipeline(TaggingPipeline):
    def __init__(
        self,
        person_image_path: str,
        raw_images_path_or_paths: Union[str, List[str]],
        database: Database,
        upload_prefix: str,
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
        self._data_processors = [
            ImagesOfPersonFinder(
                self._temp_db,
                find_data_loader,
                person_image_path,
                matched_person_field_name=self._matched_person_field,
                distance_threshold=distance_threshold,
                max_nof_images=max_results,
            ),
            SSIMDuplicateFilter(
                find_data_loader,
                self._temp_db,
                ssim_threshold=ssim_threshold,
                max_image_size=ssim_max_image_size,
                is_duplicate_field_name=ssim_is_duplicate_field,
                filters={self._matched_person_field: True},
            ),
            UploadToCloudinaryModule(
                upload_dataloader,
                self._database,
                upload_prefix=upload_prefix,
                filters={
                    self._matched_person_field: True,
                    ssim_is_duplicate_field: False,
                },
            ),
        ]
