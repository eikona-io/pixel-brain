import unittest
import numpy as np
import pandas as pd
import xgboost as xgb
from pixelbrain.database import Database
from pixelbrain.database_processors.xgboost_processor import (
    XGBoostDatabaseTrainer,
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
        self.database.store_field("1", "feature1", 1.0)
        self.database.store_field("1", "feature2", 2.0)
        self.database.store_field("1", "target", 3.0)
        self.database.store_field("2", "feature1", 2.0)
        self.database.store_field("2", "feature2", 3.0)
        self.database.store_field("2", "target", 4.0)

        self.trainer = XGBoostDatabaseTrainer(
            database=self.database,
            data_field_names=self.data_field_names,
            metric_field_name=self.metric_field_name,
            validation_split=0.5,
            xgb_max_depth=3,
            xgb_eta=0.1,
            xgb_verbosity=0,
            xgb_num_boost_round=10,
            xgb_early_stopping_rounds=5,
        )

    def test_fit(self):
        self.trainer.fit()
        self.assertIsNotNone(self.trainer._model)
        self.assertIsInstance(self.trainer._model, xgb.Booster)


class TestXGBoostDatabaseProcessor(unittest.TestCase):
    def setUp(self):
        # Create a simple in-memory database
        self.database = Database()
        self.data_field_names = ["feature1", "feature2"]

        self.database.add_image("1", "1")
        self.database.add_image("2", "2")
        # Insert some dummy data into the database
        self.database.store_field("1", "feature1", 1.0)
        self.database.store_field("1", "feature2", 2.0)
        self.database.store_field("1", "target", 3.0)
        self.database.store_field("2", "feature1", 2.0)
        self.database.store_field("2", "feature2", 3.0)
        self.database.store_field("2", "target", 4.0)

        # Train a simple model and save it
        trainer = XGBoostDatabaseTrainer(
            database=self.database,
            data_field_names=self.data_field_names,
            metric_field_name="target",
            validation_split=0.5,
            xgb_max_depth=3,
            xgb_eta=0.1,
            xgb_verbosity=0,
            xgb_num_boost_round=10,
            xgb_early_stopping_rounds=5,
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
        for record in self.database.find_images_with_fields(self.data_field_names):
            self.assertIn("xgb_score", record)


if __name__ == "__main__":
    unittest.main()
