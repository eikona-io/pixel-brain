import xgboost as xgb
import numpy as np
from typing import List, Dict, Any
from pixelbrain.database import Database
from pixelbrain.pipeline import DataProcessor
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from overrides import overrides
from pixelbrain.database_processors.xgboost_ranker_estimator import (
    XgboostRankerEstimator,
    mean_grouped_ndcg_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBRegressor


class XGBoostTrainer(ABC):
    """
    A class to train an XGBoost model using data from a pandas DataFrame.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        data_field_names: List[str],
        metric_field_name: str,
        test_split: float = 0.1,
        param_grid: Dict[str, List[Any]] = None,
        nof_cv_folds: int = 5,
    ):
        """
        Initializes the XGBoostRegressorTrainer with the given parameters.

        Args:
            data_frame (pd.DataFrame): The DataFrame containing the data.
            data_field_names (List[str]): List of field names to be used as features.
            metric_field_name (str): The field name to be used as the target variable.
            test_split (float): The proportion of data to be used for testing.
            param_grid (Dict[str, List[Any]], optional): Grid of parameters for GridSearchCV.
            nof_cv_folds (int, optional): Number of folds for cross-validation.
        """
        self._data_frame = data_frame
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
                "gamma": [0],
                "min_child_weight": [1],
            }
        )
        self._model: BaseEstimator = None
        self._trained_model: BaseEstimator = None
        self._nof_cv_folds = nof_cv_folds
        self._scorer = None

    def fit(
        self,
        save_model_path: str = None,
    ):
        """
        Trains the XGBoost model using GridSearchCV and optionally saves it to a file.

        Args:
            save_model_path (str, optional): Path to save the trained model.
        """
        X, y = self._prepare_data(self._data_frame)

        X_train, X_test, y_train, y_test = self._split_data(X, y)

        self._trained_model, best_params = self._train_model(X_train, y_train)

        test_metric = self._run_testing_experiment(X_test, y_test)

        if save_model_path:
            self._trained_model.save_model(save_model_path)

        return test_metric, best_params

    def _prepare_data(self, data: pd.DataFrame):
        """
        Prepares the data for training.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target variable arrays.
        """
        X = data[self._data_field_names]
        y = data[self._metric_field_name]
        return X, y

    @abstractmethod
    def _split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Splits the data into training and test sets.

        Args:
            X (np.ndarray): Feature array.
            y (np.ndarray): Target variable array.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and test sets.
        """
        pass

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        """
        Trains the XGBoost model using GridSearchCV.

        Args:
            X_train (np.ndarray): Training feature array.
            y_train (np.ndarray): Training target variable array.

        Returns:
            XGBRegressor: The trained XGBoost model.
        """
        assert self._model, "Model is not initialized by subclass"
        grid_search = GridSearchCV(
            estimator=self._model,
            param_grid=self._param_grid,
            scoring=make_scorer(self._scorer, greater_is_better=True),
            cv=self._get_cv_folds(X_train, y_train),
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        return best_model, best_params

    def _get_cv_folds(self, X: pd.DataFrame, y: pd.Series):
        """
        Returns the CV folds for the given data.
        """
        pass

    def _run_testing_experiment(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        Runs testing on the trained model and returns the score on the test set.

        Args:
            X_test (np.ndarray): Test feature array.
            y_test (np.ndarray): Test target variable array.
        """
        predictions = self._trained_model.predict(X_test)

        metric = self._scorer(y_test, predictions)
        return metric


class XGBoostRegressorTrainer(XGBoostTrainer):
    """
    A class to train an XGBoost Regressor model using data from a pandas DataFrame.
    """

    def __init__(
        self,
        *args,
        mse_weights_func: callable = None,
        **kwargs,
    ):
        """
        Initializes the XGBoostRegressorTrainer with the given parameters.

        Args:
            data_frame (pd.DataFrame): The DataFrame containing the data.
            data_field_names (List[str]): List of field names to be used as features.
            metric_field_name (str): The field name to be used as the target variable.
            test_split (float): The proportion of data to be used for testing.
            param_grid (Dict[str, List[Any]], optional): Grid of parameters for GridSearchCV.
            nof_cv_folds (int, optional): Number of folds for cross-validation.
            mse_weights_func (callable, optional): A function that takes a target value and returns a weight for the MSE calculation.
                                                    weights_func(y_true: np.array) -> np.array
        """
        super().__init__(*args, **kwargs)
        self._mse_weights_func = mse_weights_func
        self._scorer = self._make_weighted_mse_scorer(mse_weights_func)
        self._model = XGBRegressor(
            objective=self._make_weighted_mse_objective(mse_weights_func)
        )

    @staticmethod
    def _make_weighted_mse_scorer(mse_weights_func: callable):
        def _weighted_mse_scorer(y_true, y_pred):
            weights = (
                mse_weights_func(y_true) if mse_weights_func else np.ones(len(y_true))
            )
            return np.mean(weights * (y_true - y_pred) ** 2)

        return _weighted_mse_scorer

    @staticmethod
    def _make_weighted_mse_objective(mse_weights_func: callable):
        def _weighted_mse_objective(y_pred, y_true):
            weights = (
                mse_weights_func(y_true) if mse_weights_func else np.ones(len(y_true))
            )
            residual = y_true - y_pred
            grad = weights * residual
            hess = weights
            return grad, hess

        return _weighted_mse_objective

    @overrides
    def _get_cv_folds(self, X: pd.DataFrame, y: pd.Series):
        """
        Returns the CV folds for the given data.
        """
        return self._nof_cv_folds

    @overrides
    def _split_data(self, X: pd.DataFrame, y: pd.Series):
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


class XGBoostRankerTrainer(XGBoostTrainer):
    """
    A class to train an XGBoost Ranker model using data from a pandas DataFrame.
    """

    def __init__(
        self,
        *args,
        group_by_field_name: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._scorer = mean_grouped_ndcg_score
        self._model = XgboostRankerEstimator(
            group_by_field=group_by_field_name,
        )
        self._group_by_field_name = group_by_field_name

    @overrides
    def _get_cv_folds(self, X: pd.DataFrame, y: pd.Series):
        gkf = GroupKFold(n_splits=self._nof_cv_folds)
        groups = X[self._group_by_field_name].values
        return gkf.split(X, y, groups=groups)

    @overrides
    def _split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Splits the data into training and test sets.

        Args:
            X (np.ndarray): Feature array.
            y (np.ndarray): Target variable array.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and test sets.
        """

        grouped_df = self._data_frame.sort_values(
            by=[self._group_by_field_name, self._metric_field_name],
            ascending=[True, False],
        )
        groups = grouped_df[self._group_by_field_name].values
        gss = GroupShuffleSplit(
            test_size=self._test_split, n_splits=1, random_state=22
        ).split(X, y, groups=groups)
        train_idx, test_idx = next(gss)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        return X_train, X_test, y_train, y_test


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
        filters: Dict[str, Any] = None,
        prediction_field_name: str = "xgb_score",
        
    ):
        """
        Initializes the XGBoostDatabaseProcessor with the given parameters.

        Args:
            database (Database): The database instance to fetch data from.
            data_field_names (List[str]): List of field names to be used as features.
            model_path (str): Path to the pre-trained model.
            filters (Dict[str, Any], optional): Filters to apply to the database.
            prediction_field_name (str, optional): The field name to store predictions.
        """
        self._database = database
        self._data_field_names = data_field_names
        self._model_path = model_path
        self._prediction_field_name = prediction_field_name
        self._model = xgb.Booster()
        self._model.load_model(self._model_path)
        self._filters = filters

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
            return prep_data, data
        return prep_data

    def _load_data(self):
        """
        Loads data from the database.

        Returns:
            List[Dict[str, Any]]: List of records from the database.
        """
        filters = {field_name: None for field_name in self._data_field_names}
        if self._filters is not None:
            filters.update(self._filters)
        images = self._database.find_images_with_filters(filters)
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
