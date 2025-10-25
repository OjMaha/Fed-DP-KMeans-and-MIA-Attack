from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional

from pfl.algorithm.base import FederatedAlgorithm
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import AlgorithmHyperParamsType, ModelHyperParamsType, AlgorithmHyperParams
from pfl.metrics import Metrics, MetricName
from pfl.model.base import ModelType
from pfl.stats import MappedVectorStatistics, StatisticsType

from models import KFedModel
from utils import awasthisheffet_kmeans

# =================================================================================================
# Paper Connection: This file implements the `k-FED` algorithm, which is a non-private
# federated k-means baseline used for comparison. It is mentioned as one of the key
# baselines in Section 5 of the paper. This algorithm is from Dennis et al. (2021).
# =================================================================================================

@dataclass(frozen=True)
class KFedHyperParams(AlgorithmHyperParams):
    """
    Hyperparameters for the k-FED algorithm.
    """
    K: int  # The global number of clusters.
    K_client: int  # The number of local clusters each client computes.
    userid_to_idx: dict  # A mapping to place client centers in the correct part of a large matrix.
    multiplicative_margin: float
    train_cohort_size: int
    val_cohort_size: Optional[int]


class KFed(FederatedAlgorithm):
    """
    Implements the k-FED algorithm. The core idea is:
    1. Each client runs k-means locally to find `K_client` centers.
    2. All local centers from all clients are sent to the server.
    3. The server clusters this collection of local centers into `K` global clusters.
    This is a one-shot algorithm (it only requires one round of communication).
    """

    def __init__(self):
        super().__init__()

    def process_aggregated_statistics(
            self, central_context: CentralContext[AlgorithmHyperParamsType,
                                                  ModelHyperParamsType],
            aggregate_metrics: Metrics, model: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:
        """
        Server-side function: Takes the aggregated block of all local client centers.
        """
        return model.apply_model_update(statistics)

    def simulate_one_user(
        self, model: ModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[AlgorithmHyperParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        def name_formatting_fn(s: str):
            return MetricName(s, central_context.population)
        """
        Client-side function: Each client runs a local clustering algorithm.
        """
        if central_context.population == Population.TRAIN:
            statistics = MappedVectorStatistics()
            algo_params = central_context.algorithm_params
            K_client = algo_params.K_client
            X, _ = user_dataset.raw_data

            # The paper this baseline is from (Dennis et al., 2021) uses a specific local
            # clustering method from Awasthi & Sheffet (2012). This is implemented here.
            # Some checks are performed to handle cases with few unique points.

            dists = pairwise_distances(X, X)
            np.fill_diagonal(dists, 1)
            num_unique_points = np.isclose(dists, 0).sum() / 2

            if num_unique_points > K_client:
                try:
                    # Run the Awasthi & Sheffet k-means locally.
                    _, client_centers = awasthisheffet_kmeans(X, K_client, max_iter=100,
                                                              random_svd=False, mult_margin=algo_params.multiplicative_margin)
                except ValueError:
                    # Fallback if the local clustering fails.
                    client_centers = np.vstack([X for _ in range(K_client // len(X) + 1)])[:K_client]

            else:
                # If there are fewer unique points than K_client, just repeat the points.
                client_centers = np.vstack([X for _ in range(K_client // len(X) + 1)])[:K_client]

            # The client places its local centers into a specific slot in a large matrix.
            # This is a way to gather all centers from all clients into one block on the server.
            idx = algo_params.userid_to_idx[user_dataset.user_id]
            if (idx + 1) % 1000 == 0:
                print(f'Running user {idx+1}.')
            statistics['local_client_centers'] = np.zeros(
                (algo_params.train_cohort_size * K_client, client_centers.shape[1]))

            statistics['local_client_centers'][idx * K_client: (idx + 1) * K_client] = client_centers

            return statistics, Metrics()

        else:
            return MappedVectorStatistics(), model.evaluate(user_dataset, name_formatting_fn)

    def get_next_central_contexts(
        self,
        model: KFedModel,
        iteration: int,
        algorithm_params: AlgorithmHyperParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[
            AlgorithmHyperParamsType, ModelHyperParamsType], ...]], ModelType,
               Metrics]:
        """
        Server-side function: After receiving all local centers, the server performs the final clustering step.
        """
        # This algorithm only runs for one federated round.

        if iteration == 1:
            # 1. The model now holds the block containing all local centers from all clients.
            # 2. Run a single step of Lloyd's algorithm (or full k-means) on this set of *centers*.
            model.initialize_kfed_centers()
            lloyds_single_step = KMeans(n_clusters=algorithm_params.K, init=model.centers, max_iter=1)
            lloyds_single_step.fit(np.vstack(model.client_local_centers))
            # 3. The centers of this second-level clustering become the final global centers.
            model.centers = lloyds_single_step.cluster_centers_
            return None, model, Metrics()

        # Create the context for the one and only training round.
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