from pixel_brain.data_loader import DataLoader
from pixel_brain.database import Database
from pixel_brain.modules.embedders import FacenetEmbbedderModule


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
    assert num_faces == 6, "should be 6 faces"
    database.delete_db()