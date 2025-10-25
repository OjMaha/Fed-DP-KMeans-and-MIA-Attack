from typing import Optional, Tuple

from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.privacy import LaplaceMechanism, GaussianMechanism
from pfl.stats import TrainingStatistics

from .mechanism import SymmetricGaussianMechanism

# =================================================================================================
# Paper Connection: This file defines specialized privacy mechanism classes for handling
# "data-point-level" privacy, as discussed in Section 5.1.
# The key difference from standard PFL mechanisms is how sensitivity is enforced.
# In client-level DP, the server clips the *aggregated* client updates.
# In data-point-level DP, each client must clip its *own* update based on its local data,
# but this implementation simplifies that by having the client clip its raw data *before*
# even computing the update (see `algorithms/federated_cluster_init.py`).
# Therefore, these classes are mostly placeholders that inherit from the standard mechanisms
# but override the `constrain_sensitivity` method to do nothing, because the sensitivity
# is already being handled at the data-level.
# =================================================================================================


class DataPrivacyGaussianMechanism(GaussianMechanism):
    """
    A Gaussian mechanism for data-point level privacy.
    The `constrain_sensitivity` method is a no-op because clipping is applied
    to the raw data points directly, not the statistic.
    """
    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        return statistics, Metrics()


class DataPrivacySymmetricGaussianMechanism(SymmetricGaussianMechanism):
    """
    A symmetric Gaussian mechanism for data-point level privacy.
    Used for Step 1 of FedDP-Init in the data-point privacy setting.
    """
    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        return statistics, Metrics()


class DataPrivacyLaplaceMechanism(LaplaceMechanism):
    """
    A Laplace mechanism for data-point level privacy.
    Used for Step 2 of FedDP-Init and for noising counts in other steps.
    """
    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        return statistics, Metrics()