from typing import List
from pixelbrain.data_loader import DataLoader, DataLoaderFilter
from pixelbrain.database import Database
from pixelbrain.pipelines.identity_tagging_pipeline import IdentityTaggingPipeline
from pixelbrain.modules.face_extractor import FaceExtractorModule
from pixelbrain.apps.training_marks.training_marks_assigner import TrainingMarksAssigner
from pixelbrain.pipeline import DataProcessor, TaggingPipeline


class FaceExtractionAutgmentationFilter(DataLoaderFilter):
    def __init__(self, identity_field_name: str = 'assigned_identity', max_nof_images: int = 3) -> None:
        super().__init__()
        self._identity_field_name = identity_field_name
        self._max_nof_images = max_nof_images
        
    def filter(self, database: Database, image_ids: List[str]) -> List[str]:
        identities_meta = database.find_images_with_value(self._identity_field_name)
        unique_identities = {meta[self._identity_field_name] for meta in identities_meta}
        
        output_image_ids = []
        for identity in unique_identities:
            # find images with personalization mark out of this identity
            personalization_images = database._db.images.find({'training_mark': 'personalization', self._identity_field_name: identity})
            # choose up to max_nof_images from personalization images of that id
            output_image_ids.extend([image['_id'] for image in personalization_images[:self._max_nof_images]])
        
        if not output_image_ids:
            raise ValueError("No faces were extracted. Maybe no marks were assigned?")
        output_image_ids = list(set(output_image_ids).intersection(set(image_ids)))
        return output_image_ids


class FashionTaggingPipeline(TaggingPipeline):
    """
    This class represents the fashion tagging pipeline. It is responsible for tagging metadata, 
    tagging identities, assigning training marks, and augmenting with face crops.
    
    The pipeline follows the order: metadata_tagger -> identity_tagging -> assign_training_marks -> augment_with_face_crops
    """
    def __init__(self, 
                 dataset_path: str, 
                 database: Database,
                 metadata_tagger: DataProcessor = None,
                 apply_people_dectector: bool = False,
                 max_personalization_images: int = 7,
                 testset_ratio: float = 0.1,
                 max_nof_face_augmentation_images: int = 3,
                 face_augmentation_save_path: str = None) -> None:
        """
        Initializes the FashionTaggingPipeline with the given parameters.
        
        :param dataset_path: The path to the dataset.
        :param database: The database to use.
        :param metadata_tagger: The metadata tagger to use as the first stage of the pipeline. If None, no metadata tagger is used.
        :param apply_people_dectector: Whether to apply the people detector and filter out non-humans before people identification.
        :param max_personalization_images: The maximum number of images from each identity to tag as personalization images.
                The rest will be used either for dresser model training or as a testset.
        :param testset_ratio: The ratio of identities to use as the test set (will not be used to train the dresses model).
        :param max_nof_face_augmentation_images: The maximum number of face augmentation images per identity (could be less due to lack of identity images).
        :param face_augmentation_save_path: The save path for face augmentation images.
        """
        self._database = database
        self._dataset_path = dataset_path
        
        face_augmenter_loader = DataLoader(self._dataset_path, self._database)
        face_augmenter_filter = FaceExtractionAutgmentationFilter(max_nof_images=max_nof_face_augmentation_images)
        # optional modules
        self._data_processors = [metadata_tagger] if metadata_tagger else []
        # constant modules
        self._data_processors.extend([
            IdentityTaggingPipeline(self._dataset_path, self._database, apply_people_detector=apply_people_dectector),
            TrainingMarksAssigner(self._database, testset_ratio=testset_ratio, max_nof_personalization_images=max_personalization_images),
            FaceExtractorModule(face_augmenter_loader, self._database, filters=face_augmenter_filter, image_save_path=face_augmentation_save_path)
        ])