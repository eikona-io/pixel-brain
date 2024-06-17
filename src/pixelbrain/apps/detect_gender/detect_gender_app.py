from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from os.path import join
from os import environ
from pixelbrain.data_loaders.cloudinary_dataloader import CloudinaryDataLoader
from pixelbrain.data_loaders.s3_dataloader import S3DataLoader
from pixelbrain.database import Database
from uuid import uuid4
from typing import Literal, Optional

MAX_BATCH_SIZE = 32


class GenderDetectorApp:
    """
    A class to detect gender using images stored in Cloudinary or S3.
    For now we only consider the binary case.
    """

    def __init__(
        self,
        remote_prefix: str,
        num_images: int = 10,
        download_from_hf: bool = False,
        model_name: str = "gender-classification",
        source_type: Literal["cloudinary", "s3"] = "s3",
        bucket_name: Optional[str] = None,
    ):
        """
        Uses a HF model to detect the gender of a user based on their images as stored in the cloudinary processed folder.
        :param user_id: The user id to process
        :param num_images: Takes the average score of the first <num_images> images
        :param download_from_hf: If True, downloads the model from Hugging Face
        :param model_name: The name of the model to use. If from Hugging Face, it should be the full model name in the HF hub, else the name of the direcotry where the model is stored under $HOME/
        :param source_type: The source type to use. Either 'cloudinary' or 's3'
        :param bucket_name: The name of the bucket to use if source_type is 's3'
        """
        self.remote_prefix = remote_prefix
        self.num_images = num_images
        self.download_from_hf = download_from_hf
        self.model_name = model_name
        self.source_type = source_type
        self.bucket_name = bucket_name

    def process(self) -> float:
        """Processes the images to detect gender. Returns probability of being a female"""
        if self.download_from_hf:
            processor = AutoImageProcessor.from_pretrained(
                "rizvandwiki/gender-classification"
            )
            model = AutoModelForImageClassification.from_pretrained(
                "rizvandwiki/gender-classification"
            )
        else:
            local_model_path = join(environ.get("HOME"), self.model_name.split("/")[-1])
            processor = AutoImageProcessor.from_pretrained(local_model_path)
            model = AutoModelForImageClassification.from_pretrained(local_model_path)
            model.eval()

        local_temp_database = Database(database_id=uuid4().hex)

        if self.source_type == "cloudinary":
            dataloader = CloudinaryDataLoader(
                self.remote_prefix,
                local_temp_database,
                min(MAX_BATCH_SIZE, self.num_images),
            )
        elif self.source_type == "s3":
            dataloader = S3DataLoader(
                self.remote_prefix,
                self.bucket_name,
                local_temp_database,
            )
        else:
            raise ValueError(f"Invalid source type: {self.source_type}")

        if not len(dataloader):
            import os

            raise RuntimeError(
                f"There are no images in {self.remote_prefix}, defined cloudinary url: {os.getenv('CLOUDINARY_URL')}"
            )

        max_iterations = max(1, self.num_images // dataloader._batch_size)
        results_tensor_list = []
        for i, image_batch in enumerate(dataloader):
            inputs = processor(image_batch[1], return_tensors="pt")
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            results_tensor_list.append(probs)
            if i == max_iterations:
                break
        probability_to_be_female = (
            torch.cat(results_tensor_list, dim=0).mean(dim=0)[0].item()
        )
        return probability_to_be_female
