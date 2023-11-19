import argparse

def main():
    parser = argparse.ArgumentParser(description='CLI tool for filtering a database based on values')
    
    parser.add_argument('--filter', type=str, help='Filter field name to apply. If not provided will filter unidentified people.', default=None)
    parser.add_argument('--value', type=str, help='Filter field value to apply. if not provided will accept all values', default=None)
    parser.add_argument('--database_id', type=str, help='Mongo database id', default='filter_db')
    parser.add_argument('--read_csv', help='Read the database from a CSV file', default=None)
    parser.add_argument('--export', help='Save the database to a CSV file', default=None)
    parser.add_argument('--rm', help='Delete the MongoDB once the app is done', action='store_true')
    args = parser.parse_args()
    
    args = parser.parse_args()
    print("\nCommand-line Arguments:")
    if args.filter:
        print(f"Filter: \n\t\t\t{args.filter}")
        print(f"Filter Value: \n\t\t\t{args.value}")
    else:
        print(f"Filter: \n\t\t\tUnidentified People")
        
    print(f"Database Id: \n\t\t\t{args.database_id}")
    print(f"Read Path: \n\t\t\t{args.read_csv}")
    print(f"Export Path: \n\t\t\t{args.export}")
    print(f"Remove Database: \n\t\t\t{args.rm}\n")
    
    from pixelbrain.database import Database
    
    if args.read_csv:
        database = Database.create_from_csv(args.read_csv, database_id=args.database_id)
    else:
        database = Database(database_id=args.database_id)

    if args.filter:
        database.filter(args.filter, args.value)
    else:
        # I know it aint pretty ..
        database.filter_unidentified_people()
    
    if args.export:
        database.export_to_csv(args.export)
    
    if args.rm:
        database.delete_db()
    print("Finished filtering database!")

if __name__ == "__main__":
    main()
