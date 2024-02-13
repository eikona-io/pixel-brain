from pixelbrain.pipeline import DataProcessor
from pixelbrain.data_loader import DataLoader
import cloudinary.uploader
from typing import Optional, List, Dict
from pixelbrain.database import Database
from overrides import overrides
import os
import torch


class UploadToCloudinaryModule(DataProcessor):
    """
    Uploads images from a dataloader to cloudinary.
    Can upload all images or by a given field.
    """
    def __init__(self,
                 database: Database,
                 user_id: str,
                 filtering_field_name: Optional[str] = None,
                 filtering_field_value: Optional[str] = None):
        """
        Uploads images from their paths in the DB to cloudinary.
        Uses some user defined logic to decide which images to upload.
        
        :param images_path: Path of images to upload
        :param database: The database that keeps state for all images
        :param user_id: The id of the user being processed
        :param metadata_field_name: The name of the metadata field to store the results
        :param filters: (field_name, field_value) to apply on the dataloader before starting it processing
        :param filtering_field_name: The name of the database field to filter by. If None then all images are uploaded
        :param filtering_field_value: The value of the database field to filter by. If None then the most common value is used
        """
        self._database = database
        self._user_id = user_id
        self._filtering_field_name = filtering_field_name
        self._filtering_field_value = filtering_field_value


    @overrides
    def process(self):
        """
        Process the data using grounded sam for detecting the specific string in the photos
        and store the results in the database under the specific metadata field.
        
        :param image_ids: List of image ids
        :param processed_image_batch: Batch of preprocessed images
        """
        # Get the image ids you want to upload
        if self._filtering_field_name is None:
            upload_image_paths = [d['_id'] for d in self._database.get_all_images()]
        else:
            if self._filtering_field_value is None:
                upload_image_paths = self._database.aggregate_on_field(self._filtering_field_name)[0]['_id_list']
            else:
                upload_image_paths = [d['_id'] for d in self._database.find_images_with_value(self._filtering_field_name , self._filtering_field_value)]

        uploaded_image_ids = []
        for image_path in upload_image_paths:
            remote_image_path = f"user_photos/{self._user_id}/processed/{os.path.basename(image_path)}"
            cloudinary.uploader.upload(image_path,
                                        public_id=remote_image_path,
                                        unique_filename = False, 
                                        overwrite=True)
            uploaded_image_ids.append(remote_image_path)
        print(f"User ID: {self._user_id}, Image URLs: {uploaded_image_ids}")
