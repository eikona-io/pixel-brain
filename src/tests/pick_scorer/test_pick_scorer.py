from pixelbrain.data_loader import DataLoader
from pixelbrain.database import Database
from pixelbrain.modules.pick_scorer import PickScorerModule
from pixelbrain.utils import PIXELBRAIN_PATH
import pytest
from tests.test_utils import DeleteDatabaseAfterTest
import re


@pytest.mark.slow_suit
def pick_scorer_module_run(batch_size: int):
    database = Database()
    data = DataLoader(
        f"{PIXELBRAIN_PATH}/src/tests/pick_scorer/test_identity_1", database, batch_size=batch_size
    )

    prompt = "a man"
    module = PickScorerModule(data, database, prompt=prompt)
    with DeleteDatabaseAfterTest(database):
        module.process()
        metadata = database.get_all_images()
        scores = {image["image_path"]: image["pick_score"] for image in metadata}

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print("Sorted Scores:", sorted_scores)
        for image_idx, (image_path, score) in enumerate(sorted_scores, start=1):
            # the images name represent their expected order
            # in terms of their scores
            image_name = extract_name_from_path(image_path)
            print(f"Image ID: {image_name}, Score: {score}")
            assert image_idx == int(image_name)


def test_pick_scorer_module():
    pick_scorer_module_run(batch_size=1)

def test_pick_scorer_module_batch_size_4():
    pick_scorer_module_run(batch_size=4)


def extract_name_from_path(path):
    match = re.search(r"/([^/]+)\.png$", path)
    if match:
        return match.group(1)
    else:
        return None

test_pick_scorer_module_batch_size_4()