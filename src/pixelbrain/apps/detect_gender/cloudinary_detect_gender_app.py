from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from os.path import join
from os import environ
from pixelbrain.data_loaders.cloudinary_dataloader import CloudinaryDataLoader
from pixelbrain.database import Database
from uuid import uuid4


MAX_BATCH_SIZE = 32


class CloudinaryGenderDetector:
    """
    A class to detect gender using images stored in Cloudinary.
    For now we only consider the binary case.
    """

    def __init__(self, cloudinary_prefix: str, num_images: int = 10, download_from_hf: bool = False, model_name: str = 'gender-classification'):
        """
        Uses a HF model to detect the gender of a user based on their images as stored in the cloudinary processed folder.
        :param user_id: The user id to process
        :param num_images: Takes the average score of the first <num_images> images
        :param download_from_hf: If True, downloads the model from Hugging Face
        :param model_name: The name of the model to use. If from Hugging Face, it should be the full model name in the HF hub, else the name of the direcotry where the model is stored under $HOME/
        """
        self.cloudinary_prefix = cloudinary_prefix
        self.num_images = num_images
        self.download_from_hf = download_from_hf
        self.model_name = model_name

    def process(self) -> float:
        """Processes the images to detect gender. Returns probability of being a female"""
        if self.download_from_hf:            
            processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
            model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
        else:
            local_model_path = join(environ.get('HOME'), self.model_name.split('/')[-1])
            processor = AutoImageProcessor.from_pretrained(local_model_path)
            model = AutoModelForImageClassification.from_pretrained(local_model_path)
            model.eval()

        local_temp_database = Database(database_id=uuid4().hex)

        dataloader = CloudinaryDataLoader(self.cloudinary_prefix, local_temp_database, min(MAX_BATCH_SIZE, self.num_images))
        if not len(dataloader):
            import os
            raise RuntimeError(f"There are no images in {self.cloudinary_prefix}, defined cloudinary url: {os.getenv('CLOUDINARY_URL')}")


        max_iterations = max(1, self.num_images // dataloader._batch_size)
        results_tensor_list = []
        for i, image_batch in enumerate(dataloader):
            inputs = processor(image_batch[1], return_tensors="pt")
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            results_tensor_list.append(probs)
            if i == max_iterations:
                break
        probability_to_be_female = torch.cat(results_tensor_list, dim=0).mean(dim=0)[0].item()
        return probability_to_be_female
