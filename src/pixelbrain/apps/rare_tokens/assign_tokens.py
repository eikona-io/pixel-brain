import argparse


def main():
    parser = argparse.ArgumentParser(description='A CLI app for assigning rare tokens to each identity in a database')
    parser.add_argument('--database_id', help='The name of the database to interact with', default='rare_tokens_db')
    parser.add_argument('--tokenizer_type', help='The type of tokenizer to use', default='sd1.5')
    parser.add_argument('--identity_field_name', help='The name of the field to store the assigned identity', default='assigned_identity')
    parser.add_argument('--read_csv', help='Read the database from a CSV file', default=None)
    parser.add_argument('--export', help='Save the database to a CSV file', default=None)
    parser.add_argument('--rm', help='Delete the MongoDB once the app is done', action='store_true')
    
    args = parser.parse_args()
    print("\nCommand-line Arguments:")
    print(f"Database ID: \n\t\t\t{args.database_id}")
    print(f"Identity Field Name: \n\t\t\t{args.identity_field_name}")
    print(f"Read Path: \n\t\t\t{args.read_csv}")
    print(f"Export Path: \n\t\t\t{args.export}")
    print(f"Remove Database: \n\t\t\t{args.rm}\n")
    
    from pixelbrain.apps.rare_tokens.rare_tokens_assigner import RareTokenAssigner
    from pixelbrain.database import Database
    
    
    if args.read_csv:
        database = Database.create_from_csv(args.read_csv, database_id=args.database_id)
    else:
        database = Database(database_id=args.database_id)
    
    rare_token_assigner = RareTokenAssigner(database, tokenizer_type=args.tokenizer_type, identity_field_name=args.identity_field_name)
    
    rare_token_assigner.assign_tokens()
    
    if args.export:
        database.export_to_csv(args.export)
    
    if args.rm:
        database.delete_db()


if __name__ == '__main__':
    main()