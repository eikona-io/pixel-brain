from pixelbrain.apps.rare_tokens.rare_tokens_generator import StableDiffution15RareTokenGenerator
from pixelbrain.database import Database
from tqdm import tqdm


class RareTokenAssigner:
    def __init__(self, database: Database, tokenizer_type='sd1.5', identity_field_name='assigned_identity'):
        """
        Initialize the RareTokenAssigner class.

        :param database: The Database object to interact with the MongoDB database.
        :param tokenizer_type: The type of tokenizer to use. Default is 'sd1.5'.
        :param identity_field_name: The name of the field to store the assigned identity. Default is 'assigned_identity'.
        """
        self._database = database
        self._tokenizer_type = tokenizer_type
        self._identity_field_name = identity_field_name
        self._generator = None
        if self._tokenizer_type == 'sd1.5':
            self._generator = StableDiffution15RareTokenGenerator()
        else:
            raise ValueError(f"Unknown tokenizer type: {self._tokenizer_type}")

    def assign_tokens(self):
        """
        Assign rare tokens using the rare token generator and store them in the identity_field_name field in the database.
        """
        unique_identities = self._database._db.images.distinct(self._identity_field_name)

        for identity in tqdm(unique_identities, desc="Assigning rare tokens"):
            rare_token = next(self._generator)
            images_with_identity = self._database.find_images_with_value(self._identity_field_name, identity)
            for image_doc in images_with_identity:
                self._database.store_field(image_doc['_id'], 'rare_token', rare_token)



