def generate_aggregation_pipeline(field_name: str, sort_order: int = -1) -> str:
    """
    Returns a pipleine for aggregating the most common values of a field in the database.
    :param field_name: The field name to aggregate
    :param sort_order: The sort order of the aggregation. -1: descending, 1: ascending, 0: no sort
    """
    aggregation_pipeline = [
        {
            '$match': {
                field_name: {'$exists': True}
            }
        },
        {
            '$group': {
                '_id': f'${field_name}',
                'count': {'$sum': 1},
                '_id_list': {'$push': '$_id'}
            }
        },
    ]
    if sort_order != 0:
        aggregation_pipeline.append({
            '$sort': {
                'count': sort_order
            }
        })
    return aggregation_pipeline

