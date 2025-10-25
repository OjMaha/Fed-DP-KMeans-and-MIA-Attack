import argparse

import numpy as np
from dp_accounting.pld import privacy_loss_distribution
from typing import List

from pfl.hyperparam import get_param_value
from pfl.privacy import (CentrallyAppliedPrivacyMechanism, GaussianMechanism,
                         LaplaceMechanism, PLDPrivacyAccountant, NoPrivacy, PrivacyMechanism)

from privacy import (MultipleMechanisms, SymmetricGaussianMechanism,
                     DataPrivacyGaussianMechanism, DataPrivacySymmetricGaussianMechanism, DataPrivacyLaplaceMechanism)

# =================================================================================================
# Paper Connection: This file acts as a factory for creating and composing privacy mechanisms
# based on the command-line arguments. It also contains the crucial logic for calculating the
# final privacy budget (epsilon) using Google's `dp_accounting` library, as mentioned in
# Section 5 of the paper.
# =================================================================================================


def get_mechanism(args: argparse.Namespace, mechanism_name):
    """
    A factory function that constructs the appropriate privacy mechanism object
    based on the provided arguments.
    """
    # --- Mechanism for FedDP-Init Step 1 (Private PCA) ---
    if mechanism_name == 'outer_product':
        # Choose the right class based on whether we need data-point or client-level privacy.
        mechanism_cls = DataPrivacySymmetricGaussianMechanism if args.datapoint_privacy else SymmetricGaussianMechanism
        mechanism = CentrallyAppliedPrivacyMechanism(
            mechanism_cls.construct_single_iteration(
                clipping_bound=args.outer_product_clipping_bound,
                epsilon=args.outer_product_epsilon,
                delta=args.outer_product_delta)
        )

    # --- Mechanism for FedDP-Init Step 2 (Point Weighting) ---
    elif mechanism_name == 'point_weighting':
        mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
        mechanism = CentrallyAppliedPrivacyMechanism(
            mechanism_cls(args.weighting_clipping_bound, args.weighting_epsilon)
        )

    # --- Mechanism for FedDP-Init Step 3 (Center Initialization) ---
    elif mechanism_name == 'center_init':
        # Case 1: Clients send sums and counts.
        if args.center_init_send_sums_and_counts:
            # Create a Gaussian mechanism for the `sum_points`.
            sum_points_mechanism_cls = DataPrivacyGaussianMechanism if args.datapoint_privacy else GaussianMechanism
            sum_points_privacy = sum_points_mechanism_cls.construct_single_iteration(
                clipping_bound=args.center_init_clipping_bound,
                epsilon=args.center_init_gaussian_epsilon * args.center_init_epsilon_split,
                delta=args.center_init_delta)

            # Create a Laplace mechanism for the `num_points`.
            num_points_mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
            num_points_privacy = num_points_mechanism_cls(
                args.center_init_laplace_clipping_bound, args.center_init_gaussian_epsilon * (1 - args.center_init_epsilon_split)
            )
            # Combine them using the MultipleMechanisms wrapper.
            mechanism = CentrallyAppliedPrivacyMechanism(MultipleMechanisms(
                [sum_points_privacy, num_points_privacy],
                [('sum_points_per_component',), ('num_points_per_component',)]))
        # Case 2: Clients send means and contribution flags.
        else:
            # Create a Gaussian mechanism for the `mean_points`.
            mechanism_cls = DataPrivacyGaussianMechanism if args.datapoint_privacy else GaussianMechanism
            mean_points_privacy = mechanism_cls.construct_single_iteration(
                clipping_bound=args.center_init_clipping_bound,
                epsilon=args.center_init_gaussian_epsilon,
                delta=args.center_init_delta)

            # Create a Laplace mechanism for the `contributed_components` flags.
            components_mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
            contributed_components_privacy = components_mechanism_cls(args.center_init_contributed_components_clipping_bound,
                                                              args.center_init_contributed_components_epsilon)
            # Combine them.
            mechanism = CentrallyAppliedPrivacyMechanism(MultipleMechanisms(
                [mean_points_privacy, contributed_components_privacy],
                [('mean_points_per_component',), ('contributed_components',)]))

    # --- Mechanism for FedDP-Lloyds ---
    elif mechanism_name == 'fedlloyds':
        if args.send_sums_and_counts:
            # This uses a PLDPrivacyAccountant to calculate the noise required for multiple compositions (iterations).
            sampling_probability = args.fedlloyds_cohort_size / args.num_train_clients
            sum_points_accountant = PLDPrivacyAccountant(
                num_compositions=args.fedlloyds_num_iterations,
                sampling_probability=sampling_probability,
                mechanism='gaussian',
                epsilon=args.fedlloyds_epsilon * args.fedlloyds_epsilon_split,
                delta=args.fedlloyds_delta)

            num_points_accountant = PLDPrivacyAccountant(
                num_compositions=args.fedlloyds_num_iterations,
                sampling_probability=sampling_probability,
                mechanism='laplace',
                epsilon=args.fedlloyds_epsilon * (1 - args.fedlloyds_epsilon_split),
                delta=args.fedlloyds_delta)

            # Create the mechanisms using the noise levels calculated by the accountants.
            sum_points_mechanism_cls = DataPrivacyGaussianMechanism if args.datapoint_privacy else GaussianMechanism
            sum_points_privacy = sum_points_mechanism_cls.from_privacy_accountant(
                accountant=sum_points_accountant, clipping_bound=args.fedlloyds_clipping_bound)

            num_points_mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
            laplace_noise_param = num_points_accountant.cohort_noise_parameter
            num_points_privacy = num_points_mechanism_cls(args.fedlloyds_laplace_clipping_bound,
                                                  1 / laplace_noise_param)
            # Combine them.
            mechanism = CentrallyAppliedPrivacyMechanism(MultipleMechanisms(
                [sum_points_privacy, num_points_privacy],
                [('sum_points_per_component',), ('num_points_per_component',)]))
            
        else:
            sampling_probability = args.fedlloyds_cohort_size / args.num_train_clients
            fedlloyds_accountant = PLDPrivacyAccountant(
                num_compositions=args.fedlloyds_num_iterations,
                sampling_probability=sampling_probability,
                mechanism='gaussian',
                epsilon=args.fedlloyds_epsilon,
                delta=args.fedlloyds_delta)

            mechanism_cls = DataPrivacyGaussianMechanism if args.datapoint_privacy else GaussianMechanism
            fedlloyds_gaussian_noise_mechanism = mechanism_cls.from_privacy_accountant(
                accountant=fedlloyds_accountant, clipping_bound=args.fedlloyds_clipping_bound)

            components_mechanism_cls = DataPrivacyLaplaceMechanism if args.datapoint_privacy else LaplaceMechanism
            contributed_components_privacy = components_mechanism_cls(args.fedlloyds_contributed_components_clipping_bound, args.fedlloyds_contributed_components_epsilon)

            mechanism = CentrallyAppliedPrivacyMechanism(MultipleMechanisms(
                [fedlloyds_gaussian_noise_mechanism, contributed_components_privacy],
                [('mean_points_per_component',), ('contributed_components',)]))

    elif mechanism_name == 'no_privacy':
        mechanism = CentrallyAppliedPrivacyMechanism(NoPrivacy())

    else:
        raise ValueError('Mechanism name not recognized.')

    return mechanism


def compute_privacy_accounting(mechanisms: List[PrivacyMechanism], target_delta: float,
                               num_compositions: List[int] = None, sampling_probs: List[float] = None):
    """
    Calculates the total epsilon for a sequence of composed privacy mechanisms.
    Paper Connection: This function implements the "strong composition" mentioned in Section 5,
    using Google's `dp_accounting` library to get a tight bound on the total privacy cost.
    """
    if len(mechanisms) == 0:
        return 0

    if num_compositions is None:
        num_compositions = [1] * len(mechanisms)
    if sampling_probs is None:
        sampling_probs = [1] * len(mechanisms)

    # Unpack MultipleMechanisms into individual sub-mechanisms for accounting.
    unpacked_mechanisms = []
    unpacked_num_compositions = []
    unpacked_sampling_probs = []
    for mechanism, n, p in zip(mechanisms, num_compositions, sampling_probs):
        if type(mechanism).__name__ == "MultipleMechanisms":
            for sub_mechanism in mechanism.mechanisms:
                unpacked_mechanisms.append(sub_mechanism)
                unpacked_num_compositions.append(n)
                unpacked_sampling_probs.append(p)
        else:
            unpacked_mechanisms.append(mechanism)
            unpacked_num_compositions.append(n)
            unpacked_sampling_probs.append(p)

    # Convert each mechanism into a `privacy_loss_distribution` object.
    dp_accounting_mechanisms = []
    for mechanism, n, p in zip(unpacked_mechanisms, unpacked_num_compositions, unpacked_sampling_probs):
        if type(mechanism).__name__ in ['GaussianMechanism', 'SymmetricGaussianMechanism',
                                        'DataPrivacySymmetricGaussianMechanism', 'DataPrivacyGaussianMechanism']:
            dp_accounting_mechanism = privacy_loss_distribution.from_gaussian_mechanism(
                mechanism.relative_noise_stddev,
                value_discretization_interval=1e-3,
                sampling_prob=p
            )
        elif type(mechanism).__name__ in ['LaplaceMechanism', 'DataPrivacyLaplaceMechanism']:
            noise_scale = get_param_value(mechanism._clipping_bound) / mechanism._epsilon
            dp_accounting_mechanism = privacy_loss_distribution.from_laplace_mechanism(
                noise_scale,
                sensitivity=get_param_value(mechanism._clipping_bound),
                value_discretization_interval=1e-3,
                sampling_prob=p
            )
    
        elif type(mechanism).__name__ == 'NoPrivacy':
            return np.inf
        else:
            raise ValueError('Mechanism must be Gaussian or Laplace.')

        # Apply self-composition for the number of iterations.
        composed_dp_accounting_mechanism = dp_accounting_mechanism.self_compose(n)
        dp_accounting_mechanisms.append(composed_dp_accounting_mechanism)

    # Compose the mechanisms from all stages of the algorithm together.
    full_composed_mechanism = dp_accounting_mechanisms[0]
    for mechanism in dp_accounting_mechanisms[1:]:
        full_composed_mechanism = full_composed_mechanism.compose(mechanism)

    # Calculate the final epsilon for the given target delta.
    epsilon = full_composed_mechanism.get_epsilon_for_delta(target_delta)
    return epsilon