import argparse

def main():
    parser = argparse.ArgumentParser(description='''
        Hue data processing pipeline
    ''')
    parser.add_argument('--data_path', type=str, required=True, help='Path to image data')
    parser.add_argument('--database_id', type=str, default="identity_tagging_db", help='[Optional] Mongo Database ID')
    parser.add_argument('--export', type=str, default=None, help="[Optional] Path to export matadata database as csv to")
    parser.add_argument('--rm', action='store_true', default=True, help="[Optional] Flag whether to delete metadata and vector database after the run (defaults to False)")

    from pixelbrain.database import Database
    from pixelbrain.pipelines.face_similarity_pipeline import HueProcessingPipeline

    args = parser.parse_args()
    print("\nCommand-line Arguments:")
    print(f"Data Path: \n\t\t\t{args.data_path}")
    print(f"Database ID: \n\t\t\t{args.database_id}")
    print(f"Export Path: \n\t\t\t{args.export}")
    print(f"Remove Database: \n\t\t\t{args.rm}\n")

    database = Database(database_id=args.database_id)
    pipe = HueProcessingPipeline(args.data_path, 
                                 database)

    pipe.process()
    if args.export:
        database.export_to_csv(args.export)
    if args.rm:
        database.delete_db()
    print("Hue Processing finished!")

if __name__ == "__main__":
    main()
