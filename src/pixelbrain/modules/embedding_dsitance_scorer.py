from typing import List, Dict
import numpy as np
from pixelbrain.database import Database
from abc import ABC, abstractmethod


class ScoringStrategy(ABC):
    """
    Abstract base class for scoring strategies.
    Each strategy should implement a method to score a vector from the tested field
    against vectors from the compare_to field.
    """

    @abstractmethod
    def score(
        self, tested_vector: np.ndarray, compare_to_vectors: List[np.ndarray]
    ) -> float:
        pass


class NearestScoringStrategy(ScoringStrategy):
    """
    Scores vectors based on the nearest distance from a vector in the compare_to field.
    """

    def score(
        self, tested_vector: np.ndarray, compare_to_vectors: List[np.ndarray]
    ) -> float:
        distances = [np.linalg.norm(tested_vector - v) for v in compare_to_vectors]
        return min(distances) if distances else float("inf")


class AverageKNearestStrategy(ScoringStrategy):
    """
    Scores vectors based on the average distance of the k nearest vectors from the compare_to field.
    If there are fewer than k vectors available, it averages over whatever is available.
    """

    def __init__(self, k: int):
        self.k = k

    def score(
        self, tested_vector: np.ndarray, compare_to_vectors: List[np.ndarray]
    ) -> float:
        distances = np.array(
            [np.linalg.norm(tested_vector - v) for v in compare_to_vectors]
        )
        k = min(
            self.k, len(distances)
        )  # Use min to handle cases with fewer than k vectors
        nearest_indices = np.argsort(distances)[:k]
        average_vector = np.mean(
            [compare_to_vectors[i] for i in nearest_indices], axis=0
        )
        return np.linalg.norm(tested_vector - average_vector)


class MaximumDistanceStrategy(ScoringStrategy):
    """
    Scores vectors based on the maximum distance to any vector in the compare_to field.
    This strategy highlights the most dissimilar vector pair.
    """

    def score(
        self, tested_vector: np.ndarray, compare_to_vectors: List[np.ndarray]
    ) -> float:
        distances = [np.linalg.norm(tested_vector - v) for v in compare_to_vectors]
        return max(distances) if distances else float("inf")


class EmbeddingDistanceScorerModule:
    """
    Module for scoring vectors from a tested field against vectors from a compare_to field
    in a database using a specified scoring strategy.

    Attributes:
        database (Database): The database containing the images.
        tested_field (str): The name of the field containing the vectors to be tested.
        compare_to_field (str): The name of the field containing the vectors to compare against.
        scoring_strategy (ScoringStrategy): The strategy to use for scoring.
        k_nearest: (int): k nearest vectors to consider if average_k_nearest strategy is used
    """

    def __init__(
        self,
        database: Database,
        tested_field: str,
        compare_to_field: str,
        strategy: str,
        score_field_name="distance-score",
        k_nearest: int = 5,
    ):
        self.database = database
        self.tested_field = tested_field
        self.compare_to_field = compare_to_field
        self.score_field_name = score_field_name
        # Initialize scoring strategy based on the strategy string
        if strategy == "nearest":
            self.scoring_strategy = NearestScoringStrategy()
        elif strategy == "average_k_nearest":
            self.scoring_strategy = AverageKNearestStrategy(k=k_nearest)
        elif strategy == "maximum_distance":
            self.scoring_strategy = MaximumDistanceStrategy()
        else:
            raise ValueError(f"Unknown scoring strategy: {strategy}")

    def score_images(self, n_closest_vectors: int = 40):
        """
        Scores all images in the database that have the tested field,
        by comparing them only to the closest n vectors in the compare_to field.

        Args:
            n_closest_vectors (int): The number of closest vectors to compare against.

        Returns:
            A dictionary mapping image IDs to their scores.
        """
        tested_images = self.database.find_images_with_value(self.tested_field)

        for image in tested_images:
            tested_vector = self.database.get_field(image["_id"], self.tested_field)
            if not isinstance(tested_vector, np.ndarray):
                raise RuntimeError("No valid vector found in the tested field.")
            # Query the database for the closest n vectors
            closest_meta, _ = self.database.query_vector_field(
                self.compare_to_field, tested_vector, n_results=n_closest_vectors
            )
            closest_vectors = [
                self.database.get_field(image["_id"], self.compare_to_field)
                for image in closest_meta
            ]
            if len(closest_vectors) == 0 or not isinstance(
                closest_vectors[0], np.ndarray
            ):
                raise RuntimeError("No valid vectors found in the compare_to field.")

            score = self.scoring_strategy.score(tested_vector, closest_vectors)
            self.database.store_field(image["_id"], self.score_field_name, score)
