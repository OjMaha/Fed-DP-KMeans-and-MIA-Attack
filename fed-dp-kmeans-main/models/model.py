from dataclasses import dataclass
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Tuple, Callable, Optional


from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import ModelHyperParamsType, ModelHyperParams
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.model.base import EvaluatableModel, ModelType
from pfl.stats import StatisticsType, MappedVectorStatistics

from utils import kmeans_cost, compute_num_correct, awasthisheffet_kmeans

# =================================================================================================
# Paper Connection: This file defines the server-side model classes. These classes are not
# machine learning models in the traditional sense (like a neural network). Instead, they are
# state containers that hold the server's knowledge at each stage of the federated algorithms.
# They define how the server should update its state based on the aggregated statistics
# received from clients.
# =================================================================================================


class KMeansModel(EvaluatableModel):
    """
    A base class for any model that represents a set of k-means cluster centers.
    It primarily provides a standardized `evaluate` method.
    """
    def __init__(self, centers: np.array):
        super().__init__()
        self.centers = centers

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['KMeansModel', Metrics]:
        """ This method is meant to be overridden by subclasses. """
        pass

    def evaluate(
            self,
            dataset: AbstractDatasetType,
            name_formatting_fn: Callable[[str], StringMetricName],
            eval_params: Optional[ModelHyperParamsType] = None) -> Metrics:
        """
        Calculates standard clustering metrics (k-means cost and accuracy if labels are available)
        for a given dataset against the model's current centers.
        """
        X, Y = (dataset.raw_data if len(dataset.raw_data) == 2
                else (dataset.raw_data[0], None))

        # Assign each point to the nearest center.
        dist_matrix = pairwise_distances(X, self.centers)
        Y_pred = np.argmin(dist_matrix, axis=1)
        
        # Calculate the k-means cost (sum of squared distances).
        cost = kmeans_cost(X, self.centers, y_pred=Y_pred)
        num_correct = 0
        if Y is not None:
            # If true labels are provided, calculate clustering accuracy.
            num_correct = compute_num_correct(Y_pred, Y)

        cost_metric = Weighted(cost, len(X))
        accuracy_metric = Weighted(num_correct, len(X))
        metrics = Metrics([(name_formatting_fn('kmeans-cost'), cost_metric),
                           (name_formatting_fn('kmeans-accuracy'), accuracy_metric)
                           ])
        return metrics


@dataclass(frozen=True)
class LloydsModelHyperParams(ModelHyperParams):
    K: int


class LloydsModel(KMeansModel):
    """
    The server-side model for the `FederatedLloyds` algorithm.
    Its main responsibility is to update the cluster centers.
    """
    def __init__(self, centers: np.array):
        super().__init__(centers)

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['LloydsModel', Metrics]:
        """
        Updates the cluster centers based on the aggregated statistics from clients.
        Paper Connection: This implements the server-side update (line 10 of Algorithm 2).
        It computes `nu_r^t = m_hat_r / n_hat_r`.
        """
        # Case 1: Clients sent means and contribution counts (for client-level DP).
        if 'mean_points_per_component' in statistics.keys():
            # The new centers are the aggregated means divided by the aggregated contribution counts.
            self.centers = statistics['mean_points_per_component'] / statistics['contributed_components'].reshape(-1, 1)
        # Case 2: Clients sent sums of points and counts of points (for data-point-level DP).
        else:
            # Avoid division by zero for empty clusters.
            mask = statistics['num_points_per_component'] == 0
            statistics['num_points_per_component'][mask] = 1
            # The new centers are the aggregated sums divided by the aggregated counts.
            self.centers = (statistics['sum_points_per_component']
                            / statistics['num_points_per_component'].reshape(-1, 1))
        return self, Metrics()


@dataclass(frozen=True)
class FedClusterInitModelHyperParams(ModelHyperParams):
    K: int


class FedClusterInitModel(EvaluatableModel):
    """
    The server-side model for the `FedDP-Init` algorithm. This is a more complex state
    container as it needs to hold intermediate results across the three stages of initialization.
    """
    def __init__(self):
        super().__init__()
        self.accumulated_statistics = {}  # A dictionary to store aggregated results.
        # --- State for Step 1 ---
        self.singular_vectors = None  # The projection matrix (Π).
        # --- State for Step 2 ---
        self.proj_X_server = None  # The server's own data, projected.
        self.proj_server_point_centers = None  # The intermediate centers found in the projected space (ξ_r).
        # --- State for Step 3 ---
        self.initial_centers = None  # The final, high-dimensional initial centers (ν_r).

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['FedClusterInitModel', Metrics]:
        """
        This is a generic update function used by all three stages of FedDP-Init.
        It simply accumulates the statistics received from clients into a dictionary.
        """
        for statistic_name, statistic_val in statistics.items():
            try:
                self.accumulated_statistics[statistic_name] += statistic_val
            except KeyError:
                self.accumulated_statistics[statistic_name] = statistic_val
        return self, Metrics()

    def compute_centers(self):
        """
        Performs the final server-side computation for Step 3 of `FedDP-Init`.
        Paper Connection: Implements line 24 of Algorithm 1 (`nu_r = m_hat_r / n_hat_r`).
        """
        if 'sum_points_per_component' in self.accumulated_statistics.keys():
            point_sums = self.accumulated_statistics['sum_points_per_component']
            point_counts = self.accumulated_statistics['num_points_per_component']

            # --- ADD THIS SAFETY CHECK ---
            mask = point_counts == 0
            point_counts[mask] = 1 # Prevent division by zero
            
            self.initial_centers = point_sums / point_counts.reshape(-1, 1)
        else:
            # The "mean of means" case for client-level DP.
            self.initial_centers = self.accumulated_statistics['mean_points_per_component'] / self.accumulated_statistics['contributed_components'].reshape(-1, 1)

    def evaluate(
            self,
            dataset: AbstractDatasetType,
            name_formatting_fn: Callable[[str], StringMetricName],
            eval_params: Optional[ModelHyperParamsType] = None) -> Metrics:
        """
        Evaluates the quality of the final initial centers.
        """
        if self.initial_centers is None:
            return Metrics()
        
        # This re-uses the same evaluation logic as the base KMeansModel.
        temp_model = KMeansModel(self.initial_centers)
        return temp_model.evaluate(dataset, name_formatting_fn, eval_params)


@dataclass(frozen=True)
class KFedModelHyperParams(ModelHyperParams):
    K: int
    K_client: int


class KFedModel(KMeansModel):
    """
    The server-side model for the `k-FED` baseline algorithm.
    """
    def __init__(self, centers, K, K_client):
        super().__init__(centers)
        self.client_local_centers = [] # A list to store all local centers from all clients.
        self.K = K
        self.K_client = K_client

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['KFedModel', Metrics]:
        """
        Receives the large block of local centers and parses them into a list.
        """
        local_client_centers = statistics['local_client_centers']
        # This loop extracts each client's set of local centers from the aggregated block.
        for client_idx in range(len(local_client_centers) // self.K_client):
            self.client_local_centers.append(
                local_client_centers[client_idx * self.K_client: (client_idx + 1) * self.K_client])
        return self, Metrics()

    def initialize_kfed_centers(self):
        """
        This method is not used in the provided code's flow but represents an alternative
        way to perform the second-level clustering in k-FED, similar to k-means++.
        The current implementation in `algorithms/kfed.py` uses `sklearn.cluster.KMeans` instead.
        """
        assert len(self.client_local_centers)
        i = np.random.choice(range(len(self.client_local_centers)))
        initial_centers = self.client_local_centers.pop(i)
        remaining_centers = np.vstack(self.client_local_centers)
        dist_to_initial_centers = np.min(pairwise_distances(remaining_centers, initial_centers), axis=1)
        M = list(initial_centers)
        while len(M) < self.K:
            assert len(dist_to_initial_centers) == len(remaining_centers)
            i = np.argmax(dist_to_initial_centers)
            M.append(remaining_centers[i])
            dists_to_new_point = np.sqrt(np.sum((remaining_centers - remaining_centers[i])**2, axis=1))
            dist_to_initial_centers = np.min(np.array([dist_to_initial_centers, dists_to_new_point]), axis=0)
        self.centers = np.vstack(M)