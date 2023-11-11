from abc import ABC, abstractmethod
from pixel_brain.database import Database
from pixel_brain.data_loader import DataLoader
import torch
from typing import List


class Preprocessor(ABC):
    """
    Interface for preprocessing a batch of images for a certain model.
    """

    @abstractmethod
    def __call__(self, image_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses for preprocessing images.
        
        :param image_list: List of image tensors in batch
        :return: Preprocessed image tensor (batched)
        """
        pass


class PipelineModule(ABC):
    """
    Abstract base class for pipeline modules.
    """
    def __init__(self, data: DataLoader, database: Database, pre_processor: Preprocessor, batch_size=1):
        """
        Initialize the pipeline module.
        
        :param data: DataLoader object with data
        :param database: Database object for storing processed tags
        :param pre_processor: Preprocessor object to preprocess the data
        :param batch_size: Size of the batch to be processed
        """
        self._database = database
        self._pre_processor = pre_processor
        self._batch_size = batch_size
        self._data = data

    def process(self):
        """
        Process the data and store tags.
        """
        self._data.set_batch_size(self._batch_size)
        for image_batch in self._data:
            processed_batch = self._pre_processor(image_batch)
            self._process(processed_batch)
        
        self._post_process()

    @abstractmethod
    def _process(self, data: DataLoader):
        """
        Abstract method to be implemented by subclasses for processing data.
        
        :param data: DataLoader object with data in batches
        """
        pass
    
    def _post_process(self):
        """
        Optional method for child classes to implement some post processing logic (after all images have ben processed)
        """
        pass

class TaggingPipeline():
    """
    Class for tagging pipeline.
    """
    def __init__(self, original_data: DataLoader, database: Database):
        """
        Initialize the tagging pipeline.
        
        :param original_data: DataLoader object with original data
        :param database: Database object to store data
        """
        self._original_data = original_data
        self._database = database
        

    @abstractmethod
    def _modules(self):
        """
        Abstract property to be implemented by subclasses for defining modules.
        """
        pass

    def process(self):
        """
        Process the data using the defined modules and store the result in the database.
        """

        for module in self._modules:
            module.process()
