import xgboost as xgb
import numpy as np
from typing import List, Dict, Any
from pixelbrain.database import Database
from pixelbrain.pipeline import DataProcessor
import pandas as pd


class XGBoostDatabaseTrainer:
    """
    A class to train an XGBoost model using data from a database.

    Attributes:
        database (Database): The database instance to fetch data from.
        data_field_names (List[str]): List of field names to be used as features.
        metric_field_name (str): The field name to be used as the target variable.
        validation_split (float): The proportion of data to be used for validation.
        xgb_params (Dict[str, Any]): Parameters for the XGBoost model.
        model (xgb.Booster): The trained XGBoost model.
        num_boost_round (int): Number of boosting rounds.
        early_stopping_rounds (int): Rounds for early stopping.
    """

    def __init__(
        self,
        database: Database,
        data_field_names: List[str],
        metric_field_name: str,
        validation_split: float = 0.2,
        xgb_max_depth: int = 6,
        xgb_eta: float = 0.3,
        xgb_verbosity: int = 1,
        xgb_num_boost_round: int = 100,
        xgb_early_stopping_rounds: int = 10,
    ):
        """
        Initializes the XGBoostDatabaseTrainer with the given parameters.

        Args:
            database (Database): The database instance to fetch data from.
            data_field_names (List[str]): List of field names to be used as features.
            metric_field_name (str): The field name to be used as the target variable.
            validation_split (float): The proportion of data to be used for validation.
            xgb_max_depth (int): Maximum depth of the XGBoost trees.
            xgb_eta (float): Learning rate for XGBoost.
            xgb_verbosity (int): Verbosity level for XGBoost.
            xgb_num_boost_round (int): Number of boosting rounds.
            xgb_early_stopping_rounds (int): Rounds for early stopping.
        """
        self._database = database
        self._data_field_names = data_field_names
        self._metric_field_name = metric_field_name
        self._validation_split = validation_split
        self._xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": xgb_max_depth,
            "eta": xgb_eta,
            "verbosity": xgb_verbosity,
        }
        self._model = None
        self._num_boost_round = xgb_num_boost_round
        self._early_stopping_rounds = xgb_early_stopping_rounds

    def fit(self, save_model_path: str = None):
        """
        Trains the XGBoost model and optionally saves it to a file.

        Args:
            save_model_path (str, optional): Path to save the trained model.
        """
        data = self._load_data()
        if not data:
            raise ValueError("No data found in the database for the specified fields.")

        X, y = self._prepare_data(data)

        X_train, X_val, y_train, y_val = self._split_data(X, y)

        self._model = self._train_model(X_train, y_train, X_val, y_val)

        if save_model_path:
            self._model.save_model(save_model_path)

        self._run_validation_experiment(X_val, y_val)

    def _load_data(self):
        """
        Loads data from the database.

        Returns:
            List[Dict[str, Any]]: List of records from the database.
        """
        field_names = self._data_field_names + [self._metric_field_name]
        images = self._database.find_images_with_fields(field_names)
        return images

    def _prepare_data(self, data: List[Dict[str, Any]]):
        """
        Prepares the data for training.

        Args:
            data (List[Dict[str, Any]]): List of records from the database.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target variable arrays.
        """
        df = pd.DataFrame(data)
        X = df[self._data_field_names].values
        y = df[self._metric_field_name].values
        return X, y

    def _split_data(self, X: np.ndarray, y: np.ndarray):
        """
        Splits the data into training and validation sets.

        Args:
            X (np.ndarray): Feature array.
            y (np.ndarray): Target variable array.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and validation sets.
        """
        split_index = int(len(X) * (1 - self._validation_split))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        return X_train, X_val, y_train, y_val

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """
        Trains the XGBoost model.

        Args:
            X_train (np.ndarray): Training feature array.
            y_train (np.ndarray): Training target variable array.
            X_val (np.ndarray): Validation feature array.
            y_val (np.ndarray): Validation target variable array.

        Returns:
            xgb.Booster: The trained XGBoost model.
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, "train"), (dval, "eval")]
        model = xgb.train(
            self._xgb_params,
            dtrain,
            num_boost_round=self._num_boost_round,
            evals=evals,
            early_stopping_rounds=self._early_stopping_rounds,
        )
        return model

    def _run_validation_experiment(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Runs validation on the trained model and prints the RMSE.

        Args:
            X_val (np.ndarray): Validation feature array.
            y_val (np.ndarray): Validation target variable array.
        """
        dval = xgb.DMatrix(X_val)
        predictions = self._model.predict(dval)
        rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
        print(f"Validation RMSE: {rmse}")


class XGBoostDatabaseProcessor(DataProcessor):
    """
    A class to process data using a pre-trained XGBoost model and store predictions in the database.

    Attributes:
        database (Database): The database instance to fetch data from.
        data_field_names (List[str]): List of field names to be used as features.
        model_path (str): Path to the pre-trained model.
        prediction_field_name (str): The field name to store predictions.
        model (xgb.Booster): The loaded XGBoost model.
    """

    def __init__(
        self,
        database: Database,
        data_field_names: List[str],
        model_path: str,
        prediction_field_name: str = "xgb_score",
    ):
        """
        Initializes the XGBoostDatabaseProcessor with the given parameters.

        Args:
            database (Database): The database instance to fetch data from.
            data_field_names (List[str]): List of field names to be used as features.
            model_path (str): Path to the pre-trained model.
            prediction_field_name (str, optional): The field name to store predictions.
        """
        self._database = database
        self._data_field_names = data_field_names
        self._model_path = model_path
        self._prediction_field_name = prediction_field_name
        self._model = xgb.Booster()
        self._model.load_model(self._model_path)

    def process(self):
        """
        Processes the data and stores predictions in the database.
        """
        data = self._load_data()
        if not data:
            raise ValueError("No data found in the database for the specified fields.")

        X = self._prepare_data(data)

        predictions = self.predict(X)

        for record, prediction in zip(data, predictions):
            self._database.store_field(
                record["_id"], self._prediction_field_name, float(prediction)
            )

    def _load_data(self):
        """
        Loads data from the database.

        Returns:
            List[Dict[str, Any]]: List of records from the database.
        """
        images = self._database.find_images_with_fields(self._data_field_names)
        return images

    def _prepare_data(self, data: List[Dict[str, Any]]):
        """
        Prepares the data for prediction.

        Args:
            data (List[Dict[str, Any]]): List of records from the database.

        Returns:
            np.ndarray: Feature array.
        """
        df = pd.DataFrame(data)
        X = df[self._data_field_names].values
        return X

    def predict(self, data: np.ndarray):
        """
        Predicts using the pre-trained model.

        Args:
            data (np.ndarray): Feature array.

        Returns:
            np.ndarray: Predictions.
        """
        if not self._model:
            raise RuntimeError("Model has not been loaded yet. Call process() first.")
        dtest = xgb.DMatrix(data)
        return self._model.predict(dtest)
