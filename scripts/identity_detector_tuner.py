from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.embedders import FacenetEmbbedderModule
from pixelbrain.modules.people_identifier import PeopleIdentifierModule
from glob import glob
import os
import shutil
import random
import tempfile
from pixelbrain.utils import PIXELBRAIN_PATH
from tests.test_utils import DeleteDatabaseAfterTest
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
import functools


DATA_DIR = f"{PIXELBRAIN_PATH}/assets/tuning_data/people_identifier"


def create_dataloader(number_of_subjects, database, tempdir, add_objects=True):
    identity_dirs = glob(f"{DATA_DIR}/identity*")
    identities = [i for i in range(len(identity_dirs))]
    random.shuffle(identities)
    random_identities = identities[:number_of_subjects]
    
    for identity in random_identities:
        identity_dir = identity_dirs[identity]
        identity_images = glob(f"{identity_dir}/*")
        random.shuffle(identity_images)
        num_images_per_this_identity = int(len(identity_images) * random.randint(5,10) * 0.1)
        identity_images = identity_images[:num_images_per_this_identity]
        for idx, image_path in enumerate(identity_images):
            _, extension = os.path.splitext(image_path)
            shutil.copy(image_path, os.path.join(tempdir, f"identity_{identity}_{idx}{extension}"))
        if add_objects:
            object_files = glob(f"{DATA_DIR}/objects/*")
            for idx, object_photo_path in enumerate(object_files):
                _, extension = os.path.splitext(object_photo_path)
                shutil.copy(object_photo_path, os.path.join(tempdir, f"identity_object_{idx}{extension}"))
    dataloader = DataLoader(tempdir, database)
    return dataloader


def compare_results():
    pass


def identifier_run(number_of_subjects, comparison_algorithm, algorithm_params):
    database = Database(database_id=f"identity_identifier_run{int(time())}")
    with DeleteDatabaseAfterTest(database):
        with tempfile.TemporaryDirectory() as tempdir:
            facenet_dataloader = create_dataloader(number_of_subjects, database, tempdir)
            identifier_dataloader = facenet_dataloader.clone()
            
            facenet_embedder = FacenetEmbbedderModule(facenet_dataloader, database)
            facenet_embedder.process()

            people_identifier = PeopleIdentifierModule(
                identifier_dataloader, database, "face_embedding", strategy=comparison_algorithm, **algorithm_params
            )
            people_identifier.process()

            metadata = database.get_all_images()
                        
            df = pd.DataFrame(metadata)
            df['label'] = df['_id'].apply(lambda x: x.split('identity_')[1][0])
            df['identity'].fillna('o', inplace=True)
            labels = df['label'].unique()
            identities = df['identity'].unique()
            print(f"Detected {len(identities)} out of {len(labels)} existing labels (Including objects for both)")
            # Greedly assign the detected identity to the label with the most number of objects detected
            assignment_df = pd.DataFrame(index=labels, columns=['label', 'how_many_photos_of_label', 'ordered_detected_ids', 'assigned_id'])
            assignment_df['label'] = labels
            for label in labels:
                assignment_df.loc[label, 'how_many_photos_of_label'] = len(df[df['label'] == label])
                identity_cardinality_per_label = {}
                for identity in identities:
                    identity_cardinality = len(df[(df['identity'] == identity) & (df['label'] == label)])
                    if identity_cardinality == 0:
                        continue
                    identity_cardinality_per_label[identity] = identity_cardinality
                sorted_identities_for_label = sorted(identity_cardinality_per_label, key=lambda identity: identity_cardinality_per_label[identity])
                assignment_df.at[label, 'ordered_detected_ids'] = sorted_identities_for_label
                
            assignment_df.sort_values(by='how_many_photos_of_label', ascending=False, inplace=True)
            used_identities = []
            for idx, row in assignment_df.iterrows():
                sorted_identities = row['ordered_detected_ids']
                # TODO: Make sure the sorting order is correct here
                for candidate_id in sorted_identities[::-1]:
                    if candidate_id not in used_identities:
                        assignment_df.at[idx, 'assigned_id'] = candidate_id
                        used_identities.append(candidate_id)
                        break
            
            df['assigned_id'] = df['label'].apply(lambda label: assignment_df.loc[label]['assigned_id'])
            
            df['correct'] = np.where(
                (df['identity'].isna() & df['assigned_id'].isna()) | (df['identity'] == df['assigned_id']),
                True,
                False
            )
            accuracy = df['correct'].sum() / len(df)
            return accuracy


if __name__ == "__main__":
    # Silence tqdm
    original_init = tqdm.__init__
    def patched_init(self, *args, **kwargs):
        # Force disable to True
        kwargs['disable'] = True
        original_init(self, *args, **kwargs)
    tqdm.__init__ = patched_init
    
    # for min_samples in [2, 3, 4, 5, 6, 7]:
    for min_samples in [4, 5, 6, 7]:
        print("#" * 10, " Results for min_samples = ", min_samples, "#" * 10)
        # for num_subjects in [1, 2, 3, 4, 5]:
        for num_subjects in [3, 4, 5]:
            print("#" * 5, " Results for number of subjects = ", num_subjects, "#" * 5)
            accuracy_list = []
            for i in range(3):
                accuracy = identifier_run(num_subjects, "hdbscan", {"min_samples": min_samples, "min_cluster_size": min_samples})
                print(f"Run {i}, accuracy {accuracy}")
                accuracy_list.append(accuracy)
            
            print(f"Average accuracy: {sum(accuracy_list) / len(accuracy_list)}\n")
            