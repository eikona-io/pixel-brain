from pixelbrain.data_loader import DataLoader
from pixelbrain.pipeline import TaggingPipeline
from pixelbrain.modules.face_extractor import FaceExtractorModule
from pixelbrain.pipelines.identity_tagging_pipeline import IdentityTaggingPipeline
from pixelbrain.modules.upload_to_cloudinary import UploadToCloudinaryModule
from pixelbrain.utils import create_timestamp
from os.path import join
from os import makedirs


class HueProcessingPipeline(TaggingPipeline):
    def __init__(
        self,
        local_results_dir: str,
        dataloader: DataLoader,
        user_id: str,
        increase_face_ratio: int = 2.5,
        assigned_identity_field_name: str = "assigned_identity",
    ):
        super().__init__(local_results_dir, None)

        current_run_results_dir = join(local_results_dir, create_timestamp())
        makedirs(current_run_results_dir, exist_ok=True)

        self._data_processors = [
            FaceExtractorModule(
                dataloader,
                dataloader._database,
                image_save_path=current_run_results_dir,
                increase_face_ratio=increase_face_ratio,
            ),
            IdentityTaggingPipeline(
                current_run_results_dir,
                dataloader._database,
                apply_people_detector=False,
                identity_field_name=assigned_identity_field_name,
            ),
            UploadToCloudinaryModule(
                dataloader._database,
                user_id,
                filtering_field_name=assigned_identity_field_name,
            ),
        ]
