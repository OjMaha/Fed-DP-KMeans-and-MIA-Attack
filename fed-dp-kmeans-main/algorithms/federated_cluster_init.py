from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional

from pfl.algorithm.base import FederatedAlgorithm
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import Dataset, AbstractDatasetType
from pfl.hyperparam.base import AlgorithmHyperParamsType, ModelHyperParamsType, AlgorithmHyperParams
from pfl.metrics import Metrics, Weighted, MetricName, MetricValue, StringMetricName
from pfl.model.base import ModelType
from pfl.stats import MappedVectorStatistics, StatisticsType

from utils import project_subspace

# =================================================================================================
# Paper Connection: This entire file implements the core novel contribution of the paper:
# the `FedDP-Init` algorithm described in Section 3.1 and Algorithm 1.
# The implementation is broken into three distinct classes, each corresponding to a major step
# in the algorithm, which are then run sequentially in `run.py`.
# =================================================================================================


@dataclass(frozen=True)
class FederatedClusterInitHyperParams(AlgorithmHyperParams):
    """
    This dataclass holds all the hyperparameters required for the full FedDP-Init process.
    It consolidates parameters needed for all three steps.
    """
    K: int
    center_init_send_sums_and_counts: bool
    server_dataset: Dataset
    num_iterations_svd: int
    num_iterations_weighting: int
    num_iterations_center_init: int
    multiplicative_margin: float
    minimum_server_point_weight: float
    train_cohort_size: int
    val_cohort_size: Optional[int]
    datapoint_privacy: Optional[bool] = False
    outer_product_data_clipping_bound: Optional[float] = 1.


# =================================================================================================
# Step 1: Private PCA - Finding the Data Subspace
# Paper Connection: Implements Step 1 of Algorithm 1.
# =================================================================================================
class FederatedOuterProduct(FederatedAlgorithm):
    """
    This class handles the first stage of `FedDP-Init`: computing the top-k singular
    vectors of the combined client data in a federated and differentially private manner.
    """
    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[AlgorithmHyperParamsType,
                                                  ModelHyperParamsType],
            aggregate_metrics: Metrics, model: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:
        """
        Server-side function: Takes the aggregated (and noised) outer product matrix
        and applies it to the central model.
        """
        # The PFL framework calls `apply_model_update` which will simply add the
        # received 'outer_product' matrix to the model's accumulated statistics.
        return model.apply_model_update(statistics)

    def simulate_one_user(
        self, model: ModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[AlgorithmHyperParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        """
        Client-side function: Each client computes the outer product of its local data.
        Paper Connection: Corresponds to line 4 of Algorithm 1.
        """
        if central_context.population == Population.TRAIN:
            statistics = MappedVectorStatistics()
            metrics = Metrics()
            algo_params = central_context.algorithm_params
            X = user_dataset.raw_data[0]
            
            # This logic is structured to run for a specific number of iterations,
            # but for Step 1 it's typically just one round.
            if central_context.current_central_iteration < algo_params.num_iterations_svd:
                # --- Data Clipping for Data-Point Level Privacy ---
                # Paper Connection: As mentioned in Section 5.1, to enforce a known sensitivity for
                # data-point level privacy, the L2 norm of each data point is clipped.
                clipped_X = X
                if algo_params.datapoint_privacy:
                    norms = np.sqrt((X**2).sum(axis=1))
                    clipping_mask = norms > algo_params.outer_product_data_clipping_bound
                    clipped_X[clipping_mask] = (X[clipping_mask] / norms[clipping_mask].reshape(-1, 1)) * algo_params.outer_product_data_clipping_bound
                    clipping_metrics = Weighted(np.sum(clipping_mask), len(X))
                    metrics[StringMetricName('Fraction of clipped points')] = clipping_metrics

                # The `assert` is likely for debugging to ensure clipping isn't happening when not intended.
                # In a real run with privacy, this would likely be removed or handled differently.
                assert np.all(clipped_X == X)

                # Compute the outer product matrix (X^T * X) and return it as a statistic to the server.
                statistics['outer_product'] = clipped_X.transpose().dot(clipped_X)

            return statistics, metrics
        else:
            # This branch is for evaluation, not part of the training algorithm.
            return MappedVectorStatistics(), model.evaluate(user_dataset, name_formatting_fn)

    def get_next_central_contexts(
        self,
        model: ModelType,
        iteration: int,
        algorithm_params: AlgorithmHyperParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext, ...]], ModelType, Metrics]:
        """
        Server-side function: After the federated round is complete, this function is called.
        It performs the server-side computation for Step 1.
        Paper Connection: Corresponds to lines 6 and 7 of Algorithm 1.
        """
        X_server = algorithm_params.server_dataset.raw_data[0]
        # Once the required number of iterations are done (usually 1), the server proceeds.
        if iteration == algorithm_params.num_iterations_svd:
            # 1. Get the aggregated (and privacy-noised) outer product matrix.
            outer_product = model.accumulated_statistics['outer_product']
            
            # 2. Compute the top-k eigenvectors of this matrix. `eigh` is used for symmetric matrices.
            # This gives the principal components (singular vectors).
            _, V = eigh(outer_product,
                        subset_by_index=(len(outer_product) - algorithm_params.K, len(outer_product) - 1))
            model.singular_vectors = V.T  # Store the projection matrix in the model.

            # 3. Project the server's own data onto this learned subspace.
            model.proj_X_server = project_subspace(model.singular_vectors, X_server)

            # Returning None signals the end of this algorithm stage.
            return None, model, Metrics()

        # If more iterations are needed, create a context for the next round.
        context = CentralContext(
                current_central_iteration=iteration,
                do_evaluation=False,
                cohort_size=algorithm_params.train_cohort_size,
                population=Population.TRAIN,
                model_train_params=model_train_params.static_clone(),
                model_eval_params=model_eval_params.static_clone(),
                algorithm_params=algorithm_params.static_clone(),
                seed=self._get_seed())

        return tuple([context]), model, Metrics()


# =================================================================================================
# Step 2: Server Point Weighting - Creating the Proxy Map
# Paper Connection: Implements Step 2 of Algorithm 1.
# =================================================================================================
class FederatedServerPointWeighting(FederatedAlgorithm):
    """
    This class handles the second stage of `FedDP-Init`: using the client data to
    compute importance weights for each of the server's data points.
    """
    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[AlgorithmHyperParamsType,
                                                  ModelHyperParamsType],
            aggregate_metrics: Metrics, model: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:
        """
        Server-side function: Aggregates the weights from the clients.
        """
        return model.apply_model_update(statistics)

    def simulate_one_user(
        self, model: ModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[AlgorithmHyperParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        """
        Client-side function: Each client determines how many of its points are closest
        to each of the server's projected points.
        Paper Connection: Corresponds to lines 11-13 of Algorithm 1.
        """
        if central_context.population == Population.TRAIN:
            statistics = MappedVectorStatistics()
            X = user_dataset.raw_data[0]
            
            # 1. Get the projection matrix (V) and the server's projected data from the model.
            V = model.singular_vectors
            proj_X = project_subspace(V, X) # Project the client's own data.
            
            # 2. Compute distances between each client point and each server point in the projected space.
            dist_matrix = pairwise_distances(proj_X, model.proj_X_server)
            
            # 3. For each client point, find the index of the closest server point.
            closest_server_points = np.argmin(dist_matrix, axis=1)
            
            # 4. Count how many client points were assigned to each server point. This is the weight.
            server_point_weights = np.zeros(len(model.proj_X_server))
            for i in range(len(server_point_weights)):
                server_point_weights[i] = sum(closest_server_points == i)

            # Return these counts as statistics to be aggregated by the server.
            statistics['server_point_weights'] = server_point_weights
            return statistics, Metrics()
        else:
            return MappedVectorStatistics(), model.evaluate(user_dataset, name_formatting_fn)

    def get_next_central_contexts(
        self,
        model: ModelType,
        iteration: int,
        algorithm_params: AlgorithmHyperParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext, ...]], ModelType, Metrics]:
        """
        Server-side function: After aggregating the weights, the server clusters its own
        weighted data to find the initial centers in the projected space.
        Paper Connection: Corresponds to lines 15-17 of Algorithm 1.
        """
        if iteration == algorithm_params.num_iterations_weighting:
            # 1. Get the aggregated (and privacy-noised) server point weights.
            server_point_weights = model.accumulated_statistics['server_point_weights']
            
            # 2. (Optional) Filter out server points that are not representative of client data.
            server_point_mask = server_point_weights >= algorithm_params.minimum_server_point_weight
            
            # 3. Run weighted k-means on the server's own projected data.
            # This is a key step: it's a non-private operation performed entirely on the server
            # using the proxy information gathered from clients.
            server_point_clustering = KMeans(n_clusters=algorithm_params.K)
            server_point_clustering.fit(
                model.proj_X_server[server_point_mask],
                sample_weight=server_point_weights[server_point_mask]
            )
            # Store the resulting projected centers in the model for the next step.
            model.proj_server_point_centers = server_point_clustering.cluster_centers_

            return None, model, Metrics()

        context = CentralContext(
            current_central_iteration=iteration,
            do_evaluation=False,
            cohort_size=algorithm_params.train_cohort_size,
            population=Population.TRAIN,
            model_train_params=model_train_params.static_clone(),
            model_eval_params=model_eval_params.static_clone(),
            algorithm_params=algorithm_params.static_clone(),
            seed=self._get_seed())

        return tuple([context]), model, Metrics()


# =================================================================================================
# Step 3: Center Initialization - Translating Back to Original Space
# Paper Connection: Implements Step 3 of Algorithm 1.
# =================================================================================================
class FederatedInitFromProjectedCenters(FederatedAlgorithm):
    """
    This class handles the final stage of `FedDP-Init`: using the projected centers found
    in Step 2 to compute the final, high-dimensional initial centers for the main algorithm.
    """
    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[AlgorithmHyperParamsType,
                                                  ModelHyperParamsType],
            aggregate_metrics: Metrics, model: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:
        """Server-side aggregation of sums and counts (or means)."""
        return model.apply_model_update(statistics)

    def simulate_one_user(
        self, model: ModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[AlgorithmHyperParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        """
        Client-side function: Each client assigns its points to the projected centers and
        computes the sum and count of points for each center.
        Paper Connection: Corresponds to lines 19-21 of Algorithm 1.
        """
        algo_params = central_context.algorithm_params
        X = user_dataset.raw_data[0]
        if central_context.population == Population.TRAIN:
            statistics = MappedVectorStatistics()
            
            # 1. Project client data and compute distances to the projected centers from Step 2.
            V = model.singular_vectors
            proj_X = project_subspace(V, X)
            dist_matrix = pairwise_distances(proj_X, model.proj_server_point_centers)

            # This part implements the "1/3 proximity" rule from Awasthi & Sheffet (2012),
            # which is a heuristic to only consider points that are confidently assigned to a cluster.
            center_assignments = -np.ones(len(proj_X), dtype=int)
            smallest_two_distances = np.partition(dist_matrix, 1, axis=1)[:, :2]
            assignment_mask = (smallest_two_distances[:, 0] <=
                               (algo_params.multiplicative_margin * smallest_two_distances[:, 1]))
            center_assignments[assignment_mask] = np.argmin(dist_matrix, axis=1)[assignment_mask]

            # 2. For each new cluster, compute the sum and count of the original high-dimensional points.
            point_sums = []
            point_counts = []
            for k in range(algo_params.K):
                kth_mask = center_assignments == k
                point_sums.append(X[kth_mask].sum(axis=0))
                point_counts.append(kth_mask.sum())

            point_sums = np.vstack(point_sums)
            point_counts = np.hstack(point_counts)
            
            # 3. Return either the sums and counts directly, or the means and contribution flags,
            # depending on the privacy strategy for client-level DP (see Section 5.2).
            if algo_params.center_init_send_sums_and_counts:
                statistics['sum_points_per_component'] = point_sums
                statistics['num_points_per_component'] = point_counts
            else:
                statistics['contributed_components'] = (point_counts > 0).astype(int)
                point_counts[point_counts == 0] = 1 # Avoid division by zero.
                statistics['mean_points_per_component'] = point_sums / point_counts.reshape(-1, 1)

            return statistics, Metrics()
        else:
            return MappedVectorStatistics(), model.evaluate(user_dataset, name_formatting_fn)

    def get_next_central_contexts(
        self,
        model: ModelType,
        iteration: int,
        algorithm_params: AlgorithmHyperParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext, ...]], ModelType, Metrics]:
        """
        Server-side function: After aggregating the final sums and counts, the server computes
        the final initial centers.
        Paper Connection: Corresponds to lines 23-24 of Algorithm 1.
        """
        if iteration == algorithm_params.num_iterations_center_init:
            # The `compute_centers` method handles the division of the aggregated (and noised)
            # sums by the counts to get the final mean for each initial cluster center.
            model.compute_centers()
            return None, model, Metrics()

        context = CentralContext(
            current_central_iteration=iteration,
            do_evaluation=False,
            cohort_size=algorithm_params.train_cohort_size,
            population=Population.TRAIN,
            model_train_params=model_train_params.static_clone(),
            model_eval_params=model_eval_params.static_clone(),
            algorithm_params=algorithm_params.static_clone(),
            seed=self._get_seed())

        return tuple([context]), model, Metrics()