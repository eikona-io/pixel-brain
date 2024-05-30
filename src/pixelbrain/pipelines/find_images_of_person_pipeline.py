from pixelbrain.modules.images_of_person_finder import ImagesOfPersonFinder
from pixelbrain.modules.upload_to_cloudinary import UploadToCloudinaryModule
from pixelbrain.pipeline import TaggingPipeline, Database, DataLoader, DataLoaderFilter
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
    ):
        super().__init__(database=database)
        self._temp_db = Database()
        find_data_loader = DataLoader(raw_images_path_or_paths, database=self._temp_db)
        upload_dataloader = DataLoader(raw_images_path_or_paths, database=self._temp_db, load_images=False)
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
            UploadToCloudinaryModule(
                upload_dataloader,
                self._temp_db,
                upload_prefix=upload_prefix,
                filters={self._matched_person_field: True},
            ),
        ]
