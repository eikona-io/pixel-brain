from typing import List
from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from tqdm import tqdm


class ZalandoDataLoader(DataLoader):
    """This class is a custom dataloader for tagging zalando dataset metadata"""
    def __init__(self, images_path, database: Database):
        super().__init__(images_path, database, 1, False, False)
    
    def __next__(self) -> List[str]:
        image_id, _ = super().__next__()
        image_id = image_id[0]
        # zalando dataset dir structure is:
        # dataset_dir:
        #   - train
        #       - image: people images
        #       - cloth: clothing images
        #   - test
        #       - image: people images
        #       - cloth: clothing images
        
        # so we will search whether 'cloth' is in the image_path and tag accordingly
        image_path = self._database.get_field(image_id, 'image_path')
        is_cloth = image_path.find('cloth') != -1
        self._database.store_field(image_id, 'is_person', str(not is_cloth))
        if is_cloth:
            # its all images of tops
            self._database.store_field(image_id, 'clothing_type', 'top')
        else:
            # its all images of females
            self._database.store_field(image_id, 'gender', 'female')
        
        # the basename of the file is the same for cloth and person image
        # i.e. .../image/00.jpg is wearing .../cloth/00.jpg outfit
        outfit_id = image_path.split('/')[-1].split('.')[0]
        self._database.store_field(image_id, 'outfit_id', outfit_id)


class ZalandoMetadataTagger:
    """This is a wrapper class for tagging zalando metadata"""
    def __init__(self, dataset_path: str, database: Database) -> None:
        """
        Initializes the ZalandoMetadataTagger with dataset path and database

        :param dataset_path: The path to the dataset
        :param database: The database object to use for storing metadata
        """
        self._database = database
        self._dataset_path = dataset_path

    def tag_metadata(self):
        """
        Tags the metadata for people and clothing images in the dataset
        """
        people_dataloader = ZalandoDataLoader(f'{self._dataset_path}/image', self._database)
        cloth_dataloader = ZalandoDataLoader(f'{self._dataset_path}/cloth', self._database)
        
        # tag
        for _ in tqdm(zip(people_dataloader, cloth_dataloader), desc="Tagging Zalando Metadata", total=len(people_dataloader)):
            pass