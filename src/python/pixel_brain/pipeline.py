from abc import ABC, abstractmethod
from pixel_brain.database import Database
from pixel_brain.data_loader import DataLoader
import torch
from typing import List, Union


class Preprocessor(ABC):
    """
    Interface for preprocessing a batch of images for a certain model.
    """

    @abstractmethod
    def __call__(self, image_list: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
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
    def __init__(self, data: DataLoader, database: Database, pre_processor: Preprocessor = None):
        """
        Initialize the pipeline module.
        
        :param data: DataLoader object with data
        :param database: Database object for storing processed tags
        :param pre_processor: Preprocessor object to preprocess the data, if None, preprocessing won't be done.
        :param batch_size: Size of the batch to be processed
        """
        self._database = database
        self._pre_processor = pre_processor
        self._data = data

    def process(self):
        """
        Process the data and store tags.
        """
        for image_ids, image_batch in self._data:
            if self._pre_processor is not None:
                image_batch = self._pre_processor(image_batch)
            batch_results = self._process(image_ids, image_batch)
            if batch_results is not None:
                self._post_process_batch(*batch_results)
        
        self._post_process()

    @abstractmethod
    def _process(self, image_ids: List[str], processed_image_batch: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Abstract method to be implemented by subclasses for processing data.
        
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
