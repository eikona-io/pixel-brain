import argparse

def main():
    parser = argparse.ArgumentParser(description='''
        This is an Fashion Dataset Tagging Pipeline App. It takes an image dataset and assignes metadata to images for training personalization and dresser models.
        The pipeline follows the order: metadata_tagger -> identity_tagging -> assign_training_marks -> augment_with_face_crops
        The results are stored in a MongoDB database with a specified ID.
        Optionally, the metadata database can be exported as a CSV file to a specified path.
    ''')
    parser.add_argument('--data_path', type=str, required=True, help='Path to image data')
    parser.add_argument('--database_id', type=str, default="identity_tagging_db", help='[Optional] Mongo Database ID')
    parser.add_argument('--apply_people_detector', action='store_true', default=False, help='[Optional] Apply people detector as the first stage of the pipeline to filter out non-person images (default to True)')
    parser.add_argument('--export', type=str, default=None, help="[Optional] Path to export matadata database as csv to")
    parser.add_argument('--testset_ratio', type=float, default=0.1, help="[Optional] Ration of identities to reserve for testset. Default is 0.1.")
    parser.add_argument('--rm', action='store_true', default=False, help="[Optional] Flag whether to delete metadata and vector database after the run (defaults to False)")
    parser.add_argument('--zalando', action='store_true', default=False, help="[Optional] Flag whether this is zalando dataset (will use ZalandoMetadataTagger)")

    args = parser.parse_args()

    print("\nCommand-line Arguments:")
    print(f"Data Path: \n\t\t\t{args.data_path}")
    print(f"Database ID: \n\t\t\t{args.database_id}")
    print(f"Apply People Detector: \n\t\t\t{args.apply_people_detector}")
    print(f"Testset ratio: \n\t\t\t{args.testset_ratio}")
    print(f"Zalando dataset: \n\t\t\t{args.zalando}")
    print(f"Export Path: \n\t\t\t{args.export}")
    print(f"Remove Database: \n\t\t\t{args.rm}\n")

    from pixelbrain.database import Database
    from pixelbrain.apps.fashion_tagging_pipeline.fashion_tagging_pipeline import FashionTaggingPipeline
    from pixelbrain.apps.zalando.zalando_dataloader import ZalandoMetadataTagger

    database = Database(database_id=args.database_id)
    metadata_tagger = ZalandoMetadataTagger(args.data_path, database) if args.zalando else None
    pipe = FashionTaggingPipeline(args.data_path,
                                  database,
                                  metadata_tagger=metadata_tagger,
                                  apply_people_dectector=args.apply_people_detector,
                                  testset_ratio=args.testset_ratio)
    pipe.process()
    if args.export:
        database.export_to_csv(args.export)
    if args.rm:
        database.delete_db()
    print("Fashion Tagging finished!")

if __name__ == "__main__":
    main()
