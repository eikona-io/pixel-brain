import unittest
import numpy as np
import pandas as pd
import xgboost as xgb
from pixelbrain.database import Database
from pixelbrain.database_processors.xgboost_processor import (
    XGBoostRegressorTrainer,
    XGBoostDatabaseProcessor,
)
from tempfile import TemporaryDirectory


class TestXGBoostDatabaseTrainer(unittest.TestCase):
    def setUp(self):
        # Create a simple in-memory database
        self.database = Database()
        self.data_field_names = ["feature1", "feature2"]
        self.metric_field_name = "target"

        # Insert some dummy data into the database
        self.database.add_image("1", "1")
        self.database.add_image("2", "2")
        self.database.add_image("3", "3")
        self.database.add_image("4", "4")
        self.database.store_field("1", "feature1", 1.0)
        self.database.store_field("1", "feature2", 2.0)
        self.database.store_field("1", "target", 3.0)
        self.database.store_field("2", "feature1", 2.0)
        self.database.store_field("2", "feature2", 3.0)
        self.database.store_field("2", "target", 4.0)
        self.database.store_field("3", "feature1", 3.0)
        self.database.store_field("3", "feature2", 4.0)
        self.database.store_field("3", "target", 5.0)
        self.database.store_field("4", "feature1", 4.0)
        self.database.store_field("4", "feature2", 5.0)
        self.database.store_field("4", "target", 6.0)
        df = pd.DataFrame(self.database.get_all_images())
        self.trainer = XGBoostRegressorTrainer(
            df,
            data_field_names=self.data_field_names,
            metric_field_name=self.metric_field_name,
            test_split=0.5,
            nof_cv_folds=2
        )

    def test_fit(self):
        self.trainer.fit()
        self.assertIsNotNone(self.trainer._model)


class TestXGBoostDatabaseProcessor(unittest.TestCase):
    def setUp(self):
        # Create a simple in-memory database
        self.database = Database()
        self.data_field_names = ["feature1", "feature2"]

        self.database.add_image("1", "1")
        self.database.add_image("2", "2")
        self.database.add_image("3", "3")
        self.database.add_image("4", "4")
        # Insert some dummy data into the database
        self.database.store_field("1", "feature1", 1.0)
        self.database.store_field("1", "feature2", 2.0)
        self.database.store_field("1", "target", 3.0)
        self.database.store_field("2", "feature1", 2.0)
        self.database.store_field("2", "feature2", 3.0)
        self.database.store_field("2", "target", 4.0)
        self.database.store_field("3", "feature1", 3.0)
        self.database.store_field("3", "feature2", 4.0)
        self.database.store_field("3", "target", 5.0)
        self.database.store_field("4", "feature1", 4.0)
        self.database.store_field("4", "feature2", 5.0)
        self.database.store_field("4", "target", 6.0)

        df = pd.DataFrame(self.database.get_all_images())
        # Train a simple model and save it
        trainer = XGBoostRegressorTrainer(
            df,
            data_field_names=self.data_field_names,
            metric_field_name="target",
            test_split=0.5,
            nof_cv_folds=2
        )
        with TemporaryDirectory() as tmpdir:
            trainer.fit(save_model_path=f"{tmpdir}/test_model.xgb")

            self.processor = XGBoostDatabaseProcessor(
                database=self.database,
                data_field_names=self.data_field_names,
                model_path=f"{tmpdir}/test_model.xgb",
                prediction_field_name="xgb_score",
            )

    def test_process(self):
        self.processor.process()
        filters = {"feature1": None, "feature2": None, "target": None}
        for record in self.database.find_images_with_filters(filters):
            self.assertIn("xgb_score", record)


if __name__ == "__main__":
    unittest.main()