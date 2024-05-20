import xgboost as xgb
import numpy as np
from typing import List, Dict, Any
from pixelbrain.database import Database
from pixelbrain.pipeline import DataProcessor
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


class XGBoostDatabaseTrainer:
    """
    A class to train an XGBoost model using data from a database.

    Attributes:
        database (Database): The database instance to fetch data from.
        data_field_names (List[str]): List of field names to be used as features.
        metric_field_name (str): The field name to be used as the target variable.
        validation_split (float): The proportion of data to be used for validation.
        xgb_params (Dict[str, Any]): Parameters for the XGBoost model.
        model (XGBRegressor): The trained XGBoost model.
    """

    def __init__(
        self,
        database: Database,
        data_field_names: List[str],
        metric_field_name: str,
        test_split: float = 0.1,
        param_grid: Dict[str, List[Any]] = None,
        nof_cv_folds: int = 5,
    ):
        """
        Initializes the XGBoostDatabaseTrainer with the given parameters.

        Args:
            database (Database): The database instance to fetch data from.
            data_field_names (List[str]): List of field names to be used as features.
            metric_field_name (str): The field name to be used as the target variable.
            test_split (float): The proportion of data to be used for testing.
            param_grid (Dict[str, List[Any]], optional): Grid of parameters for GridSearchCV.
            nof_cv_folds (int, optional): Number of folds for cross-validation.
        """
        self._database = database
        self._data_field_names = data_field_names
        self._metric_field_name = metric_field_name
        self._test_split = test_split
        self._param_grid = (
            param_grid
            if param_grid
            else {
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.3],
                "n_estimators": [100, 200, 300],
                "subsample": [0.5, 0.7, 0.9],
                "colsample_bytree": [0.6, 0.8],
                "colsample_bylevel": [0.6, 0.8],
                "colsample_bynode": [0.8, 1],
                "gamma": [0],
                "min_child_weight": [1],
                # "reg_alpha": [0, 0.01],
                # "reg_lambda": [1, 1.1],
            }
        )
        self._model = None
        self._nof_cv_folds = nof_cv_folds

    def fit(self, save_model_path: str = None):
        """
        Trains the XGBoost model using GridSearchCV and optionally saves it to a file.

        Args:
            save_model_path (str, optional): Path to save the trained model.
        """
        data = self._load_data()
        if not data:
            raise ValueError("No data found in the database for the specified fields.")

        X, y = self._prepare_data(data)

        X_train, X_test, y_train, y_test = self._split_data(X, y)

        self._model = self._train_model(X_train, y_train)

        if save_model_path:
            self._model.save_model(save_model_path)

        self._run_testing_experiment(X_test, y_test)

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
        Splits the data into training and test sets.

        Args:
            X (np.ndarray): Feature array.
            y (np.ndarray): Target variable array.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and test sets.
        """
        split_index = int(len(X) * (1 - self._test_split))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        return X_train, X_test, y_train, y_test

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        """
        Trains the XGBoost model using GridSearchCV.

        Args:
            X_train (np.ndarray): Training feature array.
            y_train (np.ndarray): Training target variable array.

        Returns:
            XGBRegressor: The trained XGBoost model.
        """
        xgb_model = XGBRegressor(objective="reg:squarederror")
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=self._param_grid,
            scoring="neg_mean_squared_error",
            cv=self._nof_cv_folds,
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        return best_model

    def _run_testing_experiment(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Runs testing on the trained model and prints the RMSE.

        Args:
            X_test (np.ndarray): Test feature array.
            y_test (np.ndarray): Test target variable array.
        """
        predictions = self._model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Testing RMSE: {rmse}")
        return rmse


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

        X, raw_data = self.get_prepared_data(return_raw_data=True)

        predictions = self.predict(X)

        for record, prediction in zip(raw_data, predictions):
            self._database.store_field(
                record["_id"], self._prediction_field_name, float(prediction)
            )

    def get_prepared_data(self, return_raw_data: bool = False):
        """
        Loads data from the database and prepares it for prediction.

        Args:
            return_raw_data (bool, optional): Whether to return the raw data or the prepared data.

        Returns:
            np.ndarray: Prepared data.
        """
        data = self._load_data()
        if not data:
            raise ValueError("No data found in the database for the specified fields.")
        prep_data = self._prepare_data(data)
        if return_raw_data:
            return data, prep_data
        return prep_data

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
