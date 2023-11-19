import argparse

def main():
    parser = argparse.ArgumentParser(description='CLI tool for tagging zalando dataset metadata')
    
    parser.add_argument('--dataset_path', type=str, help='The path to zalando dataset', required=True)
    parser.add_argument('--read_csv', help='Read the database from a CSV file', default=None)
    parser.add_argument('--export', help='Save the database to a CSV file', default=None)
    parser.add_argument('--rm', help='Delete the MongoDB once the app is done', action='store_true')
    args = parser.parse_args()
    
    args = parser.parse_args()
    print("\nCommand-line Arguments:")
    print(f"Dataset Path: \n\t\t\t{args.dataset_path}")
    print(f"Read Path: \n\t\t\t{args.read_csv}")
    print(f"Export Path: \n\t\t\t{args.export}")
    print(f"Remove Database: \n\t\t\t{args.rm}\n")
    
    from pixelbrain.apps.zalando.zalando_dataloader import ZalandoMetadataTagger
    from pixelbrain.database import Database
    
    if args.read_csv:
        database = Database.create_from_csv(args.read_csv, 'zalando_db')
    else:
        database = Database('zalando_db')

    tagger = ZalandoMetadataTagger(args.dataset_path, database)
    tagger.tag_metadata()
    
    if args.export:
        database.export_to_csv(args.export)
    
    if args.rm:
        database.delete_db()
    print("Finished tagging metadata!")

if __name__ == "__main__":
    main()
