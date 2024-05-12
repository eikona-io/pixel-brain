import argparse

def main():
    parser = argparse.ArgumentParser(description='''
        This is an Identity Tagging Pipeline App. It takes an image dataset and assignes identities to images with the same person in it.
        The pipeline includes applying a people detector to filter out non-person images, and then applying an identifying strategy.
        The identifying strategy is using a vector database and can be either 'dbscan' or 'pairwise'.
        The results are stored in a MongoDB database with a specified ID.
        Optionally, the metadata database can be exported as a CSV file to a specified path.
    ''')
    parser.add_argument('--data_path', type=str, required=True, help='Path to image data')
    parser.add_argument('--database_id', type=str, default="identity_tagging_db", help='[Optional] Mongo Database ID')
    parser.add_argument('--identity_field_name', type=str, default='assigned_identity', help='[Optional] Identity field name for storing identities in database')
    parser.add_argument('--apply_people_detector', action='store_true', default=False, help='[Optional] Apply people detector as the first stage of the pipeline to filter out non-person images (default to True)')
    parser.add_argument('--identifying_strategy', type=str, default='dbscan', help="[Optional] Identifying strategy can be either 'dbscan' or 'pairwise' (defaults to dbscan)")
    parser.add_argument('--export', type=str, default=None, help="[Optional] Path to export matadata database as csv to")
    parser.add_argument('--rm', action='store_true', default=False, help="[Optional] Flag whether to delete metadata and vector database after the run (defaults to False)")

    from pixelbrain.database import Database
    from pixelbrain.pipelines.identity_tagging_pipeline import IdentityTaggingPipeline

    args = parser.parse_args()
    print("\nCommand-line Arguments:")
    print(f"Data Path: \n\t\t\t{args.data_path}")
    print(f"Database ID: \n\t\t\t{args.database_id}")
    print(f"Identity Field Name: \n\t\t\t{args.identity_field_name}")
    print(f"Apply People Detector: \n\t\t\t{args.apply_people_detector}")
    print(f"Identifying Strategy: \n\t\t\t{args.identifying_strategy}")
    print(f"Export Path: \n\t\t\t{args.export}")
    print(f"Remove Database: \n\t\t\t{args.rm}\n")

    database = Database(database_id=args.database_id)
    pipe = IdentityTaggingPipeline(args.data_path, 
                                   database,
                                   identity_field_name=args.identity_field_name,
                                   apply_people_detector=args.apply_people_detector,
                                   identifying_strategy=args.identifying_strategy)

    pipe.process()
    if args.export:
        database.export_to_csv(args.export)
    if args.rm:
        database.delete_db()
    print("Identity Tagging finished!")

if __name__ == "__main__":
    main()
