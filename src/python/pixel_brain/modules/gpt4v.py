import base64
import requests
from pixel_brain.pipeline import PipelineModule
from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
from typing import List, Dict
import torch
import logging
from pixel_brain.utils import OPENAI_KEY, get_logger


logger = get_logger(__name__)

class Gpt4VModule(PipelineModule):
    """
    Module for processing images with GPT-4 Vision and storing the results in a database.
    """
    def __init__(self, data: DataLoader, 
                 database: Database, 
                 question: str, 
                 metadata_field_name: str,
                 filters: Dict[str, str] = None,
                 high_detail=False):
        """
        Initialize the Gpt4vModule.
        
        :param data: DataLoader object with data
        :param database: Database object for storing processed tags
        :param question: Question to ask GPT-4 Vision
        :param metadata_field_name: Field name to store the results in the database
        :param high_detail: flag whether to use high detail gpt4v (costs more tokens), default to False
        """
        super().__init__(data, database, None, filters)
        self._question = question
        self._metadata_field_name = metadata_field_name
        if OPENAI_KEY is None:
            raise RuntimeError("OPENAI_KEY must be set to use GPT4VModules")
        self._api_key = OPENAI_KEY
        self._headers = self._init_openai_headers()
        self._detail = "high" if high_detail else "low"

    def _init_openai_headers(self):
        """
        Initialize a GPT-4 Vision chat session.
        
        :return: Session object
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        return headers

    def _generate_payload(self, image):
        """
        Generate the payload for the GPT-4 Vision chat session.
        
        :param image: bytes image
        :return: Payload dictionary
        """
        
        base64_image = base64.b64encode(image.numpy()).decode('utf-8')
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": self._question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": f"{self._detail}"
                        }
                    }
                    ]
                }
            ]
        }
        return payload

    def _process(self, image_ids: List[str], processed_image_batch: List[torch.Tensor]):
        """
        Process the images with GPT-4 Vision and return the results.
        
        :param image_ids: List of image ids
        :param processed_image_batch: Batch of preprocessed images
        :return: List of image ids and GPT-4 Vision results
        """
        gpt_results = []
        for image in processed_image_batch:
            payload = self._generate_payload(image)
            while True:
                response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=self._headers)
                if response.status_code != 200:
                    logger.info(f"Got a {response.status_code} for gpt4v api, response: {response.json()}")
                else:
                    break
            result = response.json()['choices'][0]['message']['content']
            gpt_results.append(result)
        return image_ids, gpt_results

    def _post_process_answers(self, gpt_results: List[str]):
        """
        Post-process the GPT-4 Vision results.
        
        Optionally to be implemented by child classes
        :param gpt_results: List of GPT-4 Vision results
        :return: Post-processed results
        """
        return gpt_results

    def _post_process_batch(self, image_ids: List[str], gpt_results: List[str]):
        """
        Post-process the batch of GPT-4 Vision results and store them in the database.
        
        :param image_ids: List of image ids
        :param gpt_results: List of GPT-4 Vision results
        """
        gpt_results = self._post_process_answers(gpt_results)
        for image_id, result in zip(image_ids, gpt_results):
            self._database.store_field(image_id, self._metadata_field_name, result)


class GPT4VPeopleDetectorModule(Gpt4VModule):
    """
    This class is a module for detecting people in images using GPT-4 Vision.
    It inherits from the Gpt4VModule class and overrides the _post_process_answers method.
    """
    def __init__(self, data: DataLoader, database: Database, filters: Dict[str, str] = None):
        question = """
        ANSWER ONLY IN YES OR NO.
        
        Is there at least one person in this image?
        """
        metadata_field_name = "is_person"
        super().__init__(data, database, question, metadata_field_name, filters=filters)
    
    def _post_process_answers(self, gpt_results: List[str]):
        processed_answers = []
        
        for result in gpt_results:
            result = result.replace(".", "").lower()
            switcher = {
                "yes": "True",
                "no": "False"
            }
            processed_answers.append(switcher.get(result.lower(), "Unkown"))
        return processed_answers
