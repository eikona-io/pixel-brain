from abc import ABC, abstractmethod
from pixelbrain.database import Database
from pixelbrain.data_loader import DataLoader, DataLoaderFilter
import torch
from typing import List, Union, Dict, Optional
from tqdm import tqdm
from overrides import overrides
import multiprocessing
import sys


class Preprocessor(ABC):
    """
    Interface for preprocessing a batch of images for a certain model.
    """

    @abstractmethod
    def __call__(
        self, image_list: List[torch.Tensor]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Abstract method to be implemented by subclasses for preprocessing images.

        :param image_list: List of image tensors in batch
        :return: Preprocessed image tensor (batched)
        """
        pass


class DataProcessor(ABC):
    """This is an anbstract for any class that processes data and writes the metadata Database to inherit from."""

    @abstractmethod
    def process(self):
        pass

    async def async_process(self):
        """
        Optional Async version of the process method.
        """
        pass


class PipelineModule(DataProcessor):
    """
    Abstract base class for pipeline modules.
    """

    def __init__(
        self,
        data: DataLoader,
        database: Database,
        pre_processor: Preprocessor = None,
        filters: Union[Dict[str, str], DataLoaderFilter] = None,
    ):
        """
        Initialize the pipeline module.
        A PipelineModule is ephemeral in the sense that it's process() function can be called only once and then a new one must be constructed.

        :param data: DataLoader object with data
        :param database: Database object for storing processed tags
        :param pre_processor: Preprocessor object to preprocess the data, if None, preprocessing won't be done.
        :param batch_size: Size of the batch to be processed
        :param filters: (field_name, field_value) to apply on the dataloader before starting it processing (or a custom filter)
        """
        self._database = database
        self._pre_processor = pre_processor
        self._data = data
        self._filters = filters
        self._processed_called = False

    @overrides
    def process(self, *args, **kwargs):
        """
        Process the data and store tags.
        """

        if self._processed_called:
            raise RuntimeError(
                "A PipelineModule process() function can only be called once and then it should be discarded."
            )

        if self._filters is not None:
            self._apply_filters()
        for image_ids, image_batch in tqdm(self._data, desc=self.__class__.__name__):
            if self._pre_processor is not None:
                image_batch = self._pre_processor(image_batch)
            batch_results = self._process(image_ids, image_batch)
            if batch_results is not None:
                self._post_process_batch(*batch_results, *args, **kwargs)

        self._post_process(*args, **kwargs)
        self._processed_called = True

    @overrides
    async def async_process(self, *args, **kwargs):
        """
        Async process the data and store tags.
        """

        if self._processed_called:
            raise RuntimeError(
                "A PipelineModule process() function can only be called once and then it should be discarded."
            )

        if self._filters is not None:
            self._apply_filters()
        for image_ids, image_batch in tqdm(self._data, desc=self.__class__.__name__):
            if self._pre_processor is not None:
                image_batch = self._pre_processor(image_batch)
            batch_results = await self._async_process(image_ids, image_batch)
            if batch_results is not None:
                self._post_process_batch(*batch_results, *args, **kwargs)

        self._post_process(*args, **kwargs)
        self._processed_called = True

    @abstractmethod
    def _process(
        self,
        image_ids: List[str],
        processed_image_batch: Union[torch.Tensor, List[torch.Tensor], List[str]],
    ):
        """
        Abstract method to be implemented by subclasses for processing data.

        :param image_ids: Tuple of image ids
        :param processed_image_batch: Batch of preprocessed images
        """
        pass

    async def _async_process(
        self,
        image_ids: List[str],
        processed_image_batch: Union[torch.Tensor, List[torch.Tensor], List[str]],
    ):
        """
        Optional async version of the _process method.

        :param image_ids: Tuple of image ids
        :param processed_image_batch: Batch of preprocessed images
        """
        pass

    def _post_process(self):
        """
        Optional method for child classes to implement some post processing logic (after all images have ben processed)
        """
        pass

    def _post_process_batch(self, *args, **kwargs):
        """
        Optional method for child classes to implement some post processing logic for each batch of images
        """
        pass

    def _apply_filters(self):
        if isinstance(self._filters, DataLoaderFilter):
            self._data.custom_filter(self._filters)
        else:
            for field_id, field_value in self._filters.items():
                self._data.filter(field_id, field_value)


class TaggingPipeline(DataProcessor):
    """
    Class for tagging pipeline.
    """

    def __init__(
        self,
        images_path: str = None,
        database: Optional[Database] = None,
    ):
        """
        Initialize the tagging pipeline.

        :param images_path: path to images (could be remote S3 bucket)
        :param database: Database object to store data
        :param modules: List of modules in order to be processed. can be none and initialized by subclass.
        """
        self._images_path = images_path
        self._database = database
        self._data_processors: List[DataProcessor] = None

    @overrides
    def process(self):
        """
        Process the data using the defined modules and store the result in the database.
        """

        if self._data_processors is None:
            raise ValueError("_modules was not initialized for the pipeline")
        for module in self._data_processors:
            module.process()
        self._post_process()

    def _post_process(self):
        """
        Optional method for child classes to implement some post processing logic (after all images have ben processed)
        """
        pass


class ParallelProcessors:
    """
    Class to run multiple DataProcessors instances in parallel using multiprocessing.
    """

    def __init__(self, modules: List[DataProcessor]):
        """
        Initialize the ParallelPipelines class.

        :param modules: List of PipelineModule instances to be run in parallel.
        """
        self._modules = modules

    def _run_pipeline(self, module: DataProcessor, queue: multiprocessing.Queue):
        """
        Run a single pipeline and stream its logs to the queue.
        """
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamToQueue(queue)
        sys.stderr = StreamToQueue(queue)
        try:
            module.process()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def process(self):
        """
        Run all the TaggingPipeline instances in parallel and stream logs to the current console.
        """
        queue = multiprocessing.Queue()
        processes = []
        for module in self._modules:
            process = multiprocessing.Process(
                target=self._run_pipeline, args=(module, queue)
            )
            processes.append(process)
            process.start()

        # Start a process to read from the queue and print to the console
        log_reader = multiprocessing.Process(target=self._log_listener, args=(queue,))
        log_reader.start()

        for process in processes:
            process.join()

        # Signal the log reader to stop
        queue.put(None)
        log_reader.join()

    def _log_listener(self, queue: multiprocessing.Queue):
        """
        Listen to the log queue and print messages to the console.
        """
        while True:
            message = queue.get()
            if message is None:
                break
            print(message)


class StreamToQueue:
    """
    File-like stream object that redirects writes to a multiprocessing queue.
    """

    def __init__(self, queue: multiprocessing.Queue):
        self.queue = queue

    def write(self, message):
        if message.strip():
            self.queue.put(message)

    def flush(self):
        pass
