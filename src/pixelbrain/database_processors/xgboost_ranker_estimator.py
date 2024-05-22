from sklearn.base import BaseEstimator
from xgboost import XGBRanker
import pandas as pd
from sklearn.metrics import ndcg_score, make_scorer


class XgboostRankerEstimator(BaseEstimator):
    def __init__(
        self,
        group_by_field,
        objective="rank:ndcg",
        max_depth=6,
        learning_rate=0.3,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        gamma=0,
        min_child_weight=1,
        n_jobs=-1,
        score_field="score",
    ):
        """
        Wrapper for XGBRanker model that can be used in a exposes scikit-learn interface.
        """
        self.objective = objective
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.n_jobs = n_jobs
        self.group_by_field = group_by_field
        self.score_field = score_field

    def fit(self, X, y=None):
        """
        Fits the XGBoostRanker model to the training data.

        Parameters:
        -----------
        X: pandas.DataFrame
            Training input data, where the last column should be the query ID.
        y: numpy.array
            Training target labels.

        Returns:
        --------
        self: object
            Returns the instance itself.
        """

        self.model = XGBRanker(
            objective=self.objective,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
        )

        group_sizes = X.groupby(self.group_by_field).size().to_numpy()
        self.model.fit(
            X.drop(columns=[self.group_by_field]), y, group=group_sizes, verbose=True
        )

        return self

    def predict(self, X, y=None, return_groups=True):
        """
        Generates predictions for the input data X.

        Parameters:
        -----------
        X: pandas.DataFrame
            Input data for which predictions should be generated, where the last column should be the query ID.
        y: numpy.array
            Training target labels.

        Returns:
        --------
        df: pandas.DataFrame
            Dataframe containing the query ID and predicted scores for each record in X.
        """

        qids = X[self.group_by_field].values
        preds = self.model.predict(X.drop(columns=[self.group_by_field]))
        if return_groups:
            df = pd.DataFrame(
                list(zip(qids, preds)), columns=[self.group_by_field, self.score_field]
            )
            return df
        else:
            return preds


def mean_grouped_ndcg_score(y_true, y_pred, k=None):
    """
    Calculates the mean NDCG score for each group in y_pred and average them.
    """
    ndcg_scores = []
    group_ids = y_pred[:, 0]
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    for group_id in group_ids.unique():
        group_mask = group_ids == group_id
        y_true_masked = y_true[group_mask]
        y_pred_masked = y_pred[group_mask]
        ndcg = ndcg_score([y_true_masked], [y_pred_masked[:, 1]], k=k)
        ndcg_scores.append(ndcg)

    average_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    return average_ndcg


def make_mean_grouped_ndcg_score(k=None):
    """
    Returns a function that calculates the mean NDCG score for each group in y_pred and average them.
    """
    return make_scorer(mean_grouped_ndcg_score, greater_is_better=True, k=k)
