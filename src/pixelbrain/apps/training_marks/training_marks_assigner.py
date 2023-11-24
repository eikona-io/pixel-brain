from typing import List, Dict
from pixelbrain.database import Database
from tqdm import tqdm
from pixelbrain.pipeline import DataProcessor
from overrides import overrides


class TrainingMarksAssigner(DataProcessor):
    """
    This class is responsible for assigning a training_purpose field for each image.
    Training purpose can be one of:
    - personalization (dreambooth training)
    - dresser (virtual-tryon control-net)
    - testset - held out test set for which only a personalization model have been trained.
    """
    def __init__(self, 
                 database: Database,
                 testset_ratio: float = 0.1,
                 max_nof_personalization_images: int = 7,
                 identity_field_name='assigned_identity'):
        """
        Initializes the TrainingMarksAssigner with database and mode.
        Mode can be 'train' or 'test'.
        """
        self._database = database
        self._max_nof_personalization_images = max_nof_personalization_images
        self._identity_field_name = identity_field_name
        self._testset_ratio = testset_ratio

    @overrides
    def process(self):
        """
        Assigns training marks according to the formula:
        Out of each identity, we will assign min(max_nof_personalization_images, nof_identity_images -1) as personalization images.
        The rest from each identity will be assigned as dresser or testset images according to testset_ratio.
        This means that for every identity a personalization model (dreambooth) will be trained on a subset (worst case of 1) of the identity images.
        The rest of the images are reserved either for the training of the dresser model or for testing purposes.
        Anyway, the images that are not used for presonalization training can be used for testing of the personalization model.
        """
        
        identities_meta = self._database.find_images_with_value(self._identity_field_name)
        unique_identities = {meta[self._identity_field_name] for meta in identities_meta}
        nof_testset_identities = int(len(unique_identities) * self._testset_ratio)

        for identity_num, identity in enumerate(tqdm(unique_identities, desc='Assigning training marks')):
            # images with assigned identity must have a person in them
            images = self._database.find_images_with_value(self._identity_field_name, identity)
            nof_identity_images = len(images)
            for i, image in enumerate(images):
                if i < min(self._max_nof_personalization_images, nof_identity_images - 1): # leave at least one dresser image
                    training_mark = 'personalization'
                else:
                    if identity_num < nof_testset_identities:
                        # reserve identity for testset
                        training_mark = 'testset'
                    else:
                        training_mark = 'dresser'
                self._database.store_field(image['_id'], 'training_mark', training_mark)