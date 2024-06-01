from skimage.metrics import structural_similarity as ssim
from typing import List, Dict
from pixelbrain.pipeline import PipelineModule, Database, DataLoader
import torch
from pixelbrain.pre_processors.ssim import SSIMPreprocessor


class SSIMDuplicateFilter(PipelineModule):
    def __init__(
        self,
        data_loader: DataLoader,
        database: Database,
        ssim_threshold: float = 0.45,
        max_image_size: int = 1024,
        is_duplicate_field_name: str = "is_duplicate",
        duplicate_of_field_name: str = "duplicate_of",
        filters: Dict[str, str] = None,
    ):
        super().__init__(
            data_loader, database, SSIMPreprocessor(max_image_size), filters=filters
        )
        self._ssim_threshold = ssim_threshold
        self._is_duplicate_field_name = is_duplicate_field_name
        self._duplicate_of_field_name = duplicate_of_field_name
        self._accepted_images = []
        self._accepted_image_ids = []

    def _process(
        self,
        image_ids: List[str],
        processed_image_batch: List[torch.Tensor],
    ):
        for image_id, processed_image in zip(image_ids, processed_image_batch):
            duplicate_of = self._is_duplicate(processed_image)
            if duplicate_of is None:
                self._accepted_images.append(processed_image)
                self._accepted_image_ids.append(image_id)
                self._database.store_field(
                    image_id, self._is_duplicate_field_name, False
                )
            else:
                self._database.store_field(
                    image_id, self._is_duplicate_field_name, True
                )
                self._database.store_field(
                    image_id, self._duplicate_of_field_name, duplicate_of
                )

    def _is_duplicate(self, new_image: torch.Tensor) -> str:
        for accepted_image, accepted_image_id in zip(
            self._accepted_images, self._accepted_image_ids
        ):
            if new_image.shape == accepted_image.shape:
                ssim_score = ssim(
                    new_image.numpy(), accepted_image.numpy(), channel_axis=0
                )
                if ssim_score >= self._ssim_threshold:
                    return accepted_image_id
        return None
