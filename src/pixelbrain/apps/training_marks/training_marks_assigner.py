from typing import List, Dict
from pixelbrain.database import Database
from tqdm import tqdm


class TrainingMarksAssigner:
    """
    This class is responsible for assigning a training_purpose field for each image.
    Training purpose can be one of:
    - personalization (dreambooth training)
    - dresser (virtual-tryon control-net)
    - testset - held out test set for which only a personalization model have been trained.
    """
    def __init__(self, database: Database, 
                 mode: str,
                 max_nof_personalization_images: int = 7,
                 identity_field_name='assigned_identity'):
        """
        Initializes the TrainingMarksAssigner with database and mode.
        Mode can be 'train' or 'test'.
        """
        self._database = database
        assert mode in ['train', 'test'], "mode should be either train or test"
        self._mode = mode
        self._max_nof_personalization_images = max_nof_personalization_images
        self._identity_field_name = identity_field_name

    def assign_training_marks(self):
        """
        Assigns training marks according to the formula:
        Out of each identity, we will assign min(max_nof_personalization_images, nof_identity_images -1) as personalization images.
        The rest from each identity will be assigned as dresser images if ‘train’ parameter is used else it would be assigned as testset images.
        """
        
        identities_meta = self._database.find_images_with_value(self._identity_field_name)
        unique_identities = {meta[self._identity_field_name] for meta in identities_meta}
    
        for identity in tqdm(unique_identities, desc='Assigning identities'):
            images = self._database.find_images_with_value(self._identity_field_name, identity)
            nof_identity_images = len(images)
            for i, image in enumerate(images):
                if i < min(self._max_nof_personalization_images, nof_identity_images - 1): # leave at least one dresser image
                    training_mark = 'personalization'
                else:
                    if self._mode == 'train':
                        training_mark = 'dresser'
                    else:
                        training_mark = 'testset'
                self._database.store_field(image['_id'], 'training_mark', training_mark)