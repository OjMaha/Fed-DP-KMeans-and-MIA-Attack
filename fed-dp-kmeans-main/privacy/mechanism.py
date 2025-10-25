import numpy as np
from typing import Optional, Tuple, List

from pfl.hyperparam import get_param_value, HyperParamClsOrFloat
from pfl.internal.ops import get_ops
from pfl.internal.ops.numpy_ops import NumpySeedScope
from pfl.metrics import StringMetricName, Metrics, Weighted
from pfl.privacy import GaussianMechanism, CentrallyApplicablePrivacyMechanism
from pfl.stats import TrainingStatistics, MappedVectorStatistics
from pfl.privacy.privacy_snr import SNRMetric

# =================================================================================================
# Paper Connection: This file defines custom privacy mechanisms used by the algorithms.
# These classes build on top of the PFL framework's privacy tools to handle the specific
# needs of the FedDP-KMeans algorithm, such as applying noise to symmetric matrices
# and composing multiple mechanisms for a single federated round. This relates to the
# DP mechanisms (Gaussian, Laplace) discussed in Section 2 and 3 of the paper.
# =================================================================================================


def add_symmetric_gaussian_noise(tensors: List[np.ndarray], stddev: float,
                       seed: Optional[int]) -> List[np.ndarray]:
    """
    Adds zero-mean Gaussian noise to a list of SQUARE matrices while preserving their symmetry.
    This is necessary for Step 1 of FedDP-Init where the outer product matrix is symmetric.
    """
    with NumpySeedScope(seed):
        data_with_noise = []
        for v in tensors:
            if v.shape[0] != v.shape[1]:
                raise ValueError('Matrix must be square.')
            # Generate noise for the upper triangle of the matrix.
            idxs = np.triu_indices(v.shape[0], 1)
            symmetric_noise = np.random.normal(loc=0, scale=stddev, size=v.shape)
            # Copy the upper triangle noise to the lower triangle to maintain symmetry.
            symmetric_noise[(idxs[1], idxs[0])] = symmetric_noise[idxs]
            data_with_noise.append(v + symmetric_noise)
    return data_with_noise


class SymmetricGaussianMechanism(GaussianMechanism):
    """
    A custom Gaussian mechanism that uses the symmetric noise function.
    Paper Connection: This is the specific mechanism used for Step 1 of FedDP-Init to
    privatize the aggregated outer product matrix (`hat(PP^T)` in Algorithm 1).
    """
    def __init__(self, clipping_bound: HyperParamClsOrFloat,
                 relative_noise_stddev: float):
        super().__init__(clipping_bound, relative_noise_stddev)

    def add_noise(
            self,
            statistics: TrainingStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """ Applies the symmetric noise and calculates signal-to-noise ratio metrics. """
        noise_stddev = self._clipping_bound * self._relative_noise_stddev
        data_with_noise = statistics.apply(add_symmetric_gaussian_noise,
                                           stddev=noise_stddev,
                                           seed=seed)
        # The rest of the function calculates metrics for logging and analysis.
        num_dimensions = statistics.num_parameters
        _metadata, weights = statistics.get_weights()
        signal_norm = get_ops().global_norm(weights, order=2)
        squared_error = num_dimensions * (noise_stddev ** 2)
        metrics = Metrics([(name_formatting_fn('DP noise std. dev.'),
                            Weighted.from_unweighted(noise_stddev)),
                           (name_formatting_fn('signal-to-DP-noise ratio'),
                            SNRMetric(signal_norm, squared_error))])
        return data_with_noise, metrics


def split_statistics(statistics: MappedVectorStatistics, key_list: List[Tuple]):
    """ Helper function to split a dictionary of statistics into multiple smaller dictionaries. """
    sub_statistics_list = [MappedVectorStatistics() for _ in range(len(key_list))]
    for sub_statistics, keys in zip(sub_statistics_list, key_list):
        for key in keys:
            sub_statistics[key] = statistics[key]
    return sub_statistics_list


def combine_statistics(statistics_list: List[MappedVectorStatistics]):
    """ Helper function to combine multiple dictionaries of statistics into one. """
    combined_statistics = MappedVectorStatistics()
    for stats in statistics_list:
        for stat_name, stat_val in stats.items():
            combined_statistics[stat_name] = stat_val
    return combined_statistics


def rename_metrics(identifier: str, metrics: Metrics):
    """ Helper function to add a prefix to metric names for clarity in logs. """
    renamed_metrics = Metrics()
    for name, val in metrics:
        new_name = StringMetricName(identifier + str(name))
        renamed_metrics[new_name] = val
    return renamed_metrics


class MultipleMechanisms(CentrallyApplicablePrivacyMechanism):
    """
    A powerful wrapper class that allows applying different privacy mechanisms to
    different parts of the client update in a single federated round.
    Paper Connection: This is used in Step 3 of FedDP-Init and in FedDP-Lloyds. For example,
    it allows applying a Gaussian mechanism to the `sum_points` vector and a Laplace
    mechanism to the `num_points` vector simultaneously, as described in the paper.
    """
    def __init__(self, mechanisms: List[CentrallyApplicablePrivacyMechanism],
                 statistic_splitting_keys: List[Tuple]):
        super().__init__()
        self.mechanisms = mechanisms
        self.statistic_splitting_keys = statistic_splitting_keys

    def add_noise(
            self,
            statistics: MappedVectorStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """
        Splits the incoming statistics, applies the corresponding mechanism to each part,
        and then recombines them.
        """
        split_statistics_list = split_statistics(statistics, self.statistic_splitting_keys)

        all_noised_statistics = []
        all_metrics = Metrics()
        for i, (sub_statistics, mechanism) in enumerate(zip(split_statistics_list, self.mechanisms)):
            # Apply the appropriate mechanism to each part of the statistics.
            noised_statistics, metrics = mechanism.add_noise(sub_statistics, cohort_size, seed=seed)
            all_noised_statistics.append(noised_statistics)
            all_metrics |= rename_metrics(f'Mechanism {i+1}: ', metrics)

        # Recombine the noised parts into a single statistics object.
        recombined_noised_statistics = combine_statistics(all_noised_statistics)
        return recombined_noised_statistics, all_metrics

    def constrain_sensitivity(
            self,
            statistics: MappedVectorStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """
        Applies the sensitivity constraint (clipping) for each mechanism to its
        corresponding part of the statistics. This happens on the client-side
        before aggregation.
        """
        split_statistics_list = split_statistics(statistics, self.statistic_splitting_keys)

        all_clipped_statistics = []
        all_metrics = Metrics()
        for i, (sub_statistics, mechanism) in enumerate(zip(split_statistics_list, self.mechanisms)):
            clipped_statistics, metrics = mechanism.constrain_sensitivity(sub_statistics, seed=seed)
            all_clipped_statistics.append(clipped_statistics)
            all_metrics |= rename_metrics(f'Mechanism {i+1}: ', metrics)

        recombined_clipped_statistics = combine_statistics(all_clipped_statistics)
        return recombined_clipped_statistics, all_metrics