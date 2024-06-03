import base64
import requests
from pixelbrain.pipeline import PipelineModule
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from typing import List, Dict
import torch
import logging
from pixelbrain.utils import OPENAI_KEY, get_logger
import asyncio
import aiohttp  # Import aiohttp for asynchronous HTTP requests

logger = get_logger(__name__)


class Gpt4VModule(PipelineModule):
    """
    Module for processing images with GPT-4 Vision and storing the results in a database.
    """

    def __init__(
        self,
        data: DataLoader,
        database: Database,
        question: str,
        metadata_field_name: str,
        filters: Dict[str, str] = None,
        high_detail=False,
    ):
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
            "Authorization": f"Bearer {self._api_key}",
        }
        return headers

    def _generate_payload(self, image):
        """
        Generate the payload for the GPT-4 Vision chat session.

        :param image: bytes image
        :return: Payload dictionary
        """

        if isinstance(image, torch.Tensor):
            base64_image = base64.b64encode(image.numpy()).decode("utf-8")
            url = f"data:image/jpeg;base64,{base64_image}"
        else:
            if not image.startswith("http"):
                raise ValueError("Image must be a URL or a base64 encoded string")
            url = image
        payload = {
            "model": "gpt-4o-2024-05-13",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                                "detail": f"{self._detail}",
                            },
                        },
                    ],
                }
            ],
        }
        return payload

    async def _async_request(self, session, payload):
        """
        Helper function to perform asynchronous HTTP POST request.
        """
        while True:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=self._headers,
            ) as response:
                if response.status != 200:
                    response_json = await response.json()
                    logger.info(
                        f"Got a {response.status} for gpt4v api, response: {response_json}"
                    )
                else:
                    return await response.json()

    def _process(self, image_ids: List[str], processed_image_batch: List[torch.Tensor]):
        """
        Process the images with GPT-4 Vision and return the results synchronously using asynchronous internals.

        :param image_ids: List of image ids
        :param processed_image_batch: Batch of preprocessed images
        :return: List of image ids and GPT-4 Vision results
        """
        gpt_results = []

        async def process_images():
            async with aiohttp.ClientSession() as session:
                tasks = []
                for image in processed_image_batch:
                    payload = self._generate_payload(image)
                    task = asyncio.create_task(self._async_request(session, payload))
                    tasks.append(task)
                responses = await asyncio.gather(*tasks)
                for response in responses:
                    result = response["choices"][0]["message"]["content"]
                    gpt_results.append(result)

        # Run the asynchronous process_images function in the event loop
        asyncio.run(process_images())

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

    def __init__(
        self, data: DataLoader, database: Database, filters: Dict[str, str] = None
    ):
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
            switcher = {"yes": "True", "no": "False"}
            processed_answers.append(switcher.get(result.lower(), "Unkown"))
        return processed_answers


class GPT4VPerfectEyesModule(Gpt4VModule):
    """
    This class is a module for detecting perfectly generated eyes in images using GPT-4 Vision.
    """

    def __init__(
        self,
        data: DataLoader,
        database: Database,
        metadata_field_name: str = "has_perfect_eyes",
        filters: Dict[str, str] = None,
    ):
        question = """
        You are a helpful assistant that helps me pick images of people that do not have anything strange in their eyes, or have their eyes obscured in some ways.
        The images have been generated by a model and might have generation artifacts in the eyes.
        The eyes and pupils should have exactly the same size as well.
        The eyes should be perfectly visible.
        I also don't want images with closed eyes.
        They should have a consistent gaze!
        Be picky, I have lots of images.
        YOU CAN ONLY ANSWER:
        yes: if the image is good
        no: if its not
        """
        super().__init__(data, database, question, metadata_field_name, filters=filters)

    def _post_process_answers(self, gpt_results: List[str]):
        processed_answers = []

        for result in gpt_results:
            result = result.replace(".", "").lower()
            switcher = {"yes": True, "no": False}
            processed_answers.append(switcher.get(result.lower(), None))
        return processed_answers


class GPT4VNoGenerationArtifactsModule(Gpt4VModule):
    """
    This class is a module for detecting generation artifacts in images using GPT-4 Vision.
    """

    def __init__(
        self,
        data: DataLoader,
        database: Database,
        metadata_field_name: str = "has_no_generation_artifacts",
        filters: Dict[str, str] = None,
    ):
        question = """
        You are a helpful assistant that helps me pick images of people that look very good.
        I don't want images that look like they were generated by a model.
        I don't want images to look like CGI in any way.
        I don't want ANY generation artifacts.
        The image must be perfect.
        Be very picky, I have lots of images.
        YOU CAN ONLY ANSWER:
        yes: if the image is good
        no: if its not
        """
        super().__init__(data, database, question, metadata_field_name, filters=filters)

    def _post_process_answers(self, gpt_results: List[str]):
        processed_answers = []

        for result in gpt_results:
            result = result.replace(".", "").lower()
            switcher = {"yes": True, "no": False}
            processed_answers.append(switcher.get(result.lower(), None))
        return processed_answers
