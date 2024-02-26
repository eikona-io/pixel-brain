from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipeline import TaggingPipeline
from pixelbrain.modules.grounded_sam_detector import GroundedSAMDetectorModule
from pixelbrain.pipelines.identity_tagging_pipeline import IdentityTaggingPipeline
from pixelbrain.modules.upload_to_cloudinary import UploadToCloudinaryModule
from pixelbrain.apps.detect_gender.cloudinary_detect_gender_app import CloudinaryGenderDetector
from pixelbrain.utils import create_timestamp
from os.path import join
from os import makedirs
from uuid import uuid4


class HueProcessingPipeline(TaggingPipeline):
    def __init__(self,
                 local_results_dir: str,
                 dataloader: DataLoader,
                 user_id: str,
                 maximal_medium_ratio: float = 1,
                 maximal_wide_ratio: float = 1,
                 include_background: bool = False,
                 path_to_sam_checkpoint: str = None):
        super().__init__(local_results_dir, None)
        
        current_run_results_dir = join(local_results_dir, create_timestamp())
        makedirs(current_run_results_dir, exist_ok=True)
        identity_db = Database(database_id=uuid4().hex)

        self._data_processors = [
            GroundedSAMDetectorModule(dataloader, dataloader._database, 'person', 'foobar', results_dir=current_run_results_dir,
                                      maximal_medium_ratio=maximal_medium_ratio, maximal_wide_ratio=maximal_wide_ratio,
                                      include_background=include_background, path_to_checkpoint=path_to_sam_checkpoint),
            IdentityTaggingPipeline(current_run_results_dir, identity_db, apply_people_detector=False),
            UploadToCloudinaryModule(identity_db, user_id, filtering_field_name='assigned_identity'),
            CloudinaryGenderDetector(user_id, num_images=32)
        ]
