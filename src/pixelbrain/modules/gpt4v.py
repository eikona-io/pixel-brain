import base64
from pixelbrain.pipeline import PipelineModule
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from typing import List, Dict
import torch
from pixelbrain.utils import OPENAI_KEY, get_logger
import asyncio
import aiohttp  # Import aiohttp for asynchronous HTTP requests
import tenacity

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
        max_retries=10,
        wait_exponential_multiplier=1,
        wait_exponential_max=10,
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
        self._max_retries = max_retries
        self._wait_exponential_multiplier = wait_exponential_multiplier
        self._wait_exponential_max = wait_exponential_max

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

    async def _async_gpt_process_and_store(self, session, payload, image_id):
        """
        Helper function to perform asynchronous HTTP POST requests.
        """

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(self._max_retries),
            wait=tenacity.wait_exponential(
                multiplier=self._wait_exponential_multiplier,
                max=self._wait_exponential_max,
            ),
        )
        async def make_request():
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=self._headers,
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    try:
                        json_data = await response.json()
                        if "error" in json_data:
                            raise RuntimeError(json_data["error"])
                        try:
                            result = json_data["choices"][0]["message"]["content"]
                            parsed_result = self._post_process_answers([result])[0]
                            if self._database.is_async():
                                await self._database.async_store_field(
                                    image_id, self._metadata_field_name, parsed_result
                                )
                            else:
                                self._database.store_field(
                                    image_id, self._metadata_field_name, parsed_result
                                )
                            return parsed_result
                        except Exception as e:
                            logger.error(f"Failed to extract JSON data: {e}")
                            raise
                    except Exception as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        raise
                else:
                    text_data = await response.text()
                    logger.info(f"Response is not JSON, received text data.")
                    return text_data

        try:
            return await make_request()
        except Exception as e:
            logger.error(f"All retries failed for image: {image_id}, error: {e}")
            return None

    def _process(self, image_ids: List[str], processed_image_batch: List[torch.Tensor]):
        """
        Process the images with GPT-4 Vision and return the results synchronously using asynchronous internals.

        :param image_ids: List of image ids
        :param processed_image_batch: Batch of preprocessed images
        :return: List of image ids and GPT-4 Vision results
        """
        gpt_results = []

        async def process_image_batch():
            async with aiohttp.ClientSession() as session:
                tasks = []
                for image_id, image in zip(image_ids, processed_image_batch):
                    payload = self._generate_payload(image)
                    task = asyncio.create_task(
                        self._async_gpt_process_and_store(session, payload, image_id)
                    )
                    tasks.append(task)
                responses = await asyncio.gather(*tasks)
                for response in responses:
                    gpt_results.append(response)

        self._run_in_event_loop(process_image_batch())
        return image_ids, gpt_results

    def _run_in_event_loop(self, func):
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(func)

    def _post_process_answers(self, gpt_results: List[str]):
        """
        Post-process the GPT-4 Vision results.

        Optionally to be implemented by child classes
        :param gpt_results: List of GPT-4 Vision results
        :return: Post-processed results
        """
        return gpt_results


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
