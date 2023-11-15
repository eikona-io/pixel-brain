from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
from pixel_brain.modules.embedders import FacenetEmbbedderModule
import numpy as np
import re
import matplotlib.pyplot as plt


def test_facenet_embedder():
    database = Database(database_id="facenet_embedder_test")
    data = DataLoader("assets/test_data", database)

    # Create an instance of FacenetEmbbedderModule
    facenet_embedder = FacenetEmbbedderModule(data, database)

    # Call the _process method
    facenet_embedder.process()
    metadata = database.get_all_images()
    num_faces = 0
    for image_meta in metadata:
        if "face_embedding" in image_meta:
            num_faces += 1
    assert num_faces == 30, "should be 30 faces"
    database.delete_db()


def calculate_distance(image_path1, image_path2, db: Database):
    vec1 = db.get_field(image_path1, "face_embedding")
    vec2 = db.get_field(image_path2, "face_embedding")
    distance = np.linalg.norm(vec1 - vec2) ** 2
    return distance

def calculate_self_distances(subject, db: Database):
    subject_1_path = f"assets/test_data/subjects/{subject}_1.jpeg"
    subject_2_path = f"assets/test_data/subjects/{subject}_2.jpeg"
    subject_3_path = f"assets/test_data/subjects/{subject}_3.jpeg"
    one_two_dist = calculate_distance(subject_1_path, subject_2_path, db)
    one_three_dist = calculate_distance(subject_1_path, subject_3_path, db)
    two_three_dist = calculate_distance(subject_2_path, subject_3_path, db)
    return [one_two_dist, one_three_dist, two_three_dist]


def calculate_closest_other_distance(subject, db: Database):
    subject_1_path = f"assets/test_data/subjects/{subject}_1.jpeg"
    vec = db.get_field(subject_1_path, "face_embedding")
    metadatas, dists = db.query_vector_field("face_embedding", vec, n_results=5)
    for metadata, distance in zip(metadatas, dists):
        id = metadata["_id"]
        match = re.search(r'/(\d+)_\d+.jpeg', id)
        subject_int = match.group(1)
        if str(subject_int) != subject: # closest non-subject dis
            return distance

def calulate_subjects_distances(db: Database):
    subjects = [f"{i}" for i in range(1, 9)]
    self_distances = {}
    closest_other_distances = {}

    for subject in subjects:
        subject_self_distances = calculate_self_distances(subject, db)
        subject_closest_other_distance = calculate_closest_other_distance(subject, db)
        self_distances[subject] = subject_self_distances
        closest_other_distances[subject] = subject_closest_other_distance

    return self_distances, closest_other_distances

def test_facenet_embedder_threshold(draw_plt=False):
    database = Database(database_id="facenet_embedder_test")
    data = DataLoader("assets/test_data/subjects", database)

    # Create an instance of FacenetEmbbedderModule
    facenet_embedder = FacenetEmbbedderModule(data, database)
    facenet_embedder.process()

    self_dists, other_dists = calulate_subjects_distances(database)

    SELF_THRESHOLD = 290
    database.delete_db()
    
    if draw_plt:
        draw_precision_recall_curve(self_dists, other_dists)
    for subject in self_dists:
        assert other_dists[subject] > SELF_THRESHOLD, f"Closest distance from other subject must be larger then {SELF_THRESHOLD}"

def draw_precision_recall_curve(self_dists, other_dists):
    """
    Function to draw a precision/recall curve for different distances from 150 to 400.
    Also plots the average self distance and other distance below the precision recall graph.

    :param self_dists: Dictionary with subjects as keys and list of self distances as values
    :param other_dists: Dictionary with subjects as keys and closest other distance as value
    """
    thresholds = np.arange(150, 401, 10)
    precision = []
    recall = []

    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        total_self_dists = []
        total_other_dists = [] 

        for subject in self_dists:
            for self_dist in self_dists[subject]:
                total_self_dists.append(self_dist)
                if self_dist < threshold:
                    tp += 1
                else:
                    fn += 1

            total_other_dists.append(other_dists[subject])
            if other_dists[subject] < threshold:
                fp += 1

        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
    
    plt.figure(figsize=(10, 7))

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')

    # Adding thresholds to the plot
    for i, threshold in enumerate(thresholds):
        plt.annotate(str(threshold), (recall[i], precision[i]))

    # Writing below the plot the avg value for self/other distances
    avg_self_dist = sum(total_self_dists) / len(total_self_dists)
    avg_other_dist = sum(total_other_dists) / len(total_other_dists)
    plt.figtext(0.5, 0.01, f"Average self distance: {avg_self_dist}, Average other distance: {avg_other_dist}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    plt.legend(['Precision-Recall', 'Thresholds'])

    plt.show()
    plt.savefig('precision_recall_curve.png')