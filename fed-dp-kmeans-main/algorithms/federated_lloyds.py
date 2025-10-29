from dataclasses import dataclass
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional

from pfl.algorithm.base import FederatedAlgorithm
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import Dataset
from pfl.hyperparam.base import AlgorithmHyperParams
from pfl.metrics import Metrics, Weighted, MetricName
from pfl.stats import MappedVectorStatistics

from utils import compute_num_correct, kmeans_cost
from models import LloydsModel, LloydsModelHyperParams

import os, argparse

# =================================================================================================
# Paper Connection: This file implements the `FedDP-Lloyds` algorithm, which is the
# iterative refinement part of the overall FedDP-KMeans method. It corresponds to
# Section 3.2 and Algorithm 2 in the paper.
# =================================================================================================


@dataclass(frozen=True)
class FederatedLloydsHyperParams(AlgorithmHyperParams):
    """
    This dataclass holds all the hyperparameters required for the Federated Lloyds algorithm.
    """
    K: int
    send_sums_and_counts: int  # Whether clients send sums/counts or means.
    central_num_iterations: int  # The total number of communication rounds (T in the paper).
    evaluation_frequency: int
    train_cohort_size: int
    val_cohort_size: Optional[int]


class FederatedLloyds(FederatedAlgorithm):
    """
    This class implements the federated and differentially private version of Lloyd's algorithm for k-means.
    It takes an initial set of centers and iteratively refines them.
    """
    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[FederatedLloydsHyperParams,
                                                  LloydsModelHyperParams],
            aggregate_metrics: Metrics, model: LloydsModel,
            statistics: MappedVectorStatistics) -> Tuple[LloydsModel, Metrics]:
        """
        Server-side function: This is called after the server receives the aggregated statistics from clients.
        It updates the central model's cluster centers.
        Paper Connection: Corresponds to lines 9-10 of Algorithm 2.
        """
        # The `apply_model_update` method in `LloydsModel` will perform the division of the
        # aggregated (and noised) sums by the aggregated (and noised) counts to compute the new centers.
        return model.apply_model_update(statistics)

    def simulate_one_user(
        self, model: LloydsModel, user_dataset: Dataset,
        central_context: CentralContext[FederatedLloydsHyperParams,
                                        LloydsModelHyperParams]
    ) -> Tuple[Optional[MappedVectorStatistics], Metrics]:
        """
        Client-side function: This function simulates the work done by a single client in one round of the algorithm.
        Paper Connection: Corresponds to lines 3-7 of Algorithm 2.
        """
        algo_params = central_context.algorithm_params
        if central_context.population == Population.TRAIN:
            X, Y = (user_dataset.raw_data if len(user_dataset.raw_data) == 2
                    else (user_dataset.raw_data[0], None))
            
            # 1. Assign each local data point to the closest global cluster center from the previous round.
            # Paper Connection: Corresponds to line 5 of Algorithm 2 (`S_r^j`).
            dist_matrix = pairwise_distances(X, model.centers)
            Y_pred = np.argmin(dist_matrix, axis=1)

            # 2. For each of the K global clusters, compute the sum of the points assigned to it and the count of points.
            # Paper Connection: Corresponds to line 6 of Algorithm 2 (`m_r^j` and `n_r^j`).
            sum_points_per_component = []
            num_points_per_component = []
            for k in range(algo_params.K):
                component_mask = Y_pred == k
                sum_of_points = np.sum(X[component_mask], axis=0)
                sum_points_per_component.append(sum_of_points)
                num_points_per_component.append(sum(component_mask))

            sum_points_per_component = np.array(sum_points_per_component)
            num_points_per_component = np.array(num_points_per_component)

            # 3. Prepare the statistics to be sent to the server.
            # Depending on the privacy strategy (especially for client-level DP), the client
            # might send means and a contribution flag instead of raw sums and counts.
            statistics = MappedVectorStatistics()
            if algo_params.send_sums_and_counts:
                statistics['sum_points_per_component'] = sum_points_per_component
                statistics['num_points_per_component'] = num_points_per_component
            else:
                # This 'else' block corresponds to the "mean of means" approach described for
                # client-level privacy in Section 5.2 and Appendix G.4.
                statistics['contributed_components'] = (num_points_per_component > 0).astype(int)
                num_points_per_component[num_points_per_component == 0] = 1  # Avoid division by zero
                statistics['mean_points_per_component'] = (sum_points_per_component /
                                                           num_points_per_component.reshape(-1, 1))
            
            # --- Local Evaluation Metrics ---
            # These are computed locally for monitoring but are not part of the algorithm's state.
            cost = kmeans_cost(X, model.centers, y_pred=Y_pred)
            num_correct = 0
            if Y is not None:
                num_correct = compute_num_correct(Y_pred, Y)

            cost_metric = Weighted(cost, len(X))
            accuracy_metric = Weighted(num_correct, len(X))
            metrics = Metrics([(MetricName('kmeans-cost', central_context.population), cost_metric),
                               (MetricName('kmeans-accuracy', central_context.population), accuracy_metric)])

            return statistics, metrics
        else:
            # This branch is for evaluation on a validation set, not part of the training algorithm.
            return None, model.evaluate(user_dataset,
                                         lambda s: MetricName(s, central_context.population))

    def get_next_central_contexts(
        self,
        model: LloydsModel,
        iteration: int,
        algorithm_params: FederatedLloydsHyperParams,
        model_train_params: LloydsModelHyperParams,
        model_eval_params: Optional[LloydsModelHyperParams] = None,
    ) -> Tuple[Optional[Tuple[CentralContext, ...]], LloydsModel,
               Metrics]:
        """
        Server-side function: This orchestrates the federated rounds. It creates the context
        for the next training iteration.
        """
        # Stop the algorithm after the specified number of iterations.
        if iteration == algorithm_params.central_num_iterations:
            return None, model, Metrics()

        do_evaluation = iteration % algorithm_params.evaluation_frequency == 0

        # Create the context for the next training round.
        configs = [
            CentralContext(current_central_iteration=iteration,
                           do_evaluation=do_evaluation,
                           cohort_size=algorithm_params.train_cohort_size,
                           population=Population.TRAIN,
                           model_train_params=model_train_params.static_clone(),
                           model_eval_params=model_eval_params.static_clone(),
                           algorithm_params=algorithm_params.static_clone(),
                           seed=self._get_seed())
        ]

        # Optionally create a context for an evaluation round.
        if do_evaluation and algorithm_params.val_cohort_size is not None:
            configs.append(
                CentralContext(
                    current_central_iteration=iteration,
                    do_evaluation=do_evaluation,
                    cohort_size=algorithm_params.val_cohort_size,
                    population=Population.VAL,
                    algorithm_params=algorithm_params.static_clone(),
                    model_train_params=model_train_params.static_clone(),
                    model_eval_params=model_eval_params.static_clone(),
                    seed=self._get_seed()))
        return tuple(configs), model, Metrics()