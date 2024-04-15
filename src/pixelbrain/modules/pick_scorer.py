from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.pipeline import PipelineModule
from pixelbrain.pre_processors.pil_image import PilImagePreprocessor
from typing import List, Dict
from transformers import AutoProcessor, AutoModel
import torch


class PickScorerModule(PipelineModule):
    """Compute pick scores for images: https://arxiv.org/abs/2305.01569"""
    def __init__(
        self,
        data: DataLoader,
        database: Database,
        prompt: str,
        pick_score_field_name: str = "pick_score",
        filters: Dict[str, str] = None,
    ):
        super().__init__(data, database, PilImagePreprocessor(), filters)

        self._pre_processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = (
            AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
        )
        self._tokenized_prompt = self._pre_processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        self._pick_score_fieild_name = pick_score_field_name

    def _process(self, image_ids: List[str], processed_image_batch):
        image_inputs = self._pre_processor(
            images=processed_image_batch,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        with torch.no_grad():
            image_embs = self._model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self._model.get_text_features(**self._tokenized_prompt)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            probs = torch.softmax(scores, dim=-1)

            scores = probs.cpu().tolist()

        for pick_score, image_id in zip(scores, image_ids):
            self._database.store_field(
                image_id, self._pick_score_fieild_name, pick_score
            )
