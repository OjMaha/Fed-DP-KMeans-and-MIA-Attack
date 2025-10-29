import argparse
from utils import str2bool


def add_algorithms_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    This is the main function for adding all algorithm-specific arguments to the
    top-level parser in `run.py`. It serves as an entry point that calls
    other, more specific argument-adding functions.
    """
    # --- General Algorithm Choices ---
    parser.add_argument("--K", type=int, default=10,
                        help="The total number of global clusters to find.")
    parser.add_argument("--initialization_algorithm", type=str,
                        choices=['FederatedClusterInit', 'FederatedClusterInitExact',
                                 'ServerKMeans++', 'KFed', 'SpherePacking', 'ServerLloyds'],
                        default='FederatedClusterInitExact',
                        help="The method to use for generating the initial cluster centers. "
                             "'FederatedClusterInit' and 'FederatedClusterInitExact' correspond to FedDP-Init.")
    parser.add_argument("--clustering_algorithm", type=str,
                        choices=['FederatedLloyds', 'None'],
                        default='FederatedLloyds',
                        help="The iterative algorithm to run after initialization. 'FederatedLloyds' corresponds to FedDP-Lloyds.")

    # --- General Privacy Arguments ---
    parser.add_argument("--overall_target_delta", type=float, default=1e-6,
                        help="The delta parameter for the overall (epsilon, delta)-DP guarantee.")
    parser.add_argument("--datapoint_privacy", type=str2bool, default=False,
                        help="If True, provides data-point level privacy. If False, provides client-level privacy.")

    # --- Add arguments for each specific algorithm ---
    parser = add_fedclusterinit_arguments(parser)
    parser = add_fedlloyds_arguments(parser)
    parser = add_kfed_arguments(parser)

    return parser


def add_fedclusterinit_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds all hyperparameters related to the FedDP-Init algorithm.
    Paper Connection: These parameters control the behavior of Algorithm 1 and its three main steps.
    """
    # --- Communication Round Controls ---
    parser.add_argument("--num_iterations_svd", type=int, default=1,
                        help="Number of communication rounds for Step 1 (Private PCA).")
    parser.add_argument("--num_iterations_weighting", type=int, default=1,
                        help="Number of communication rounds for Step 2 (Server Point Weighting).")
    parser.add_argument("--num_iterations_center_init", type=int, default=1,
                        help="Number of communication rounds for Step 3 (Center Initialization).")
    parser.add_argument("--cohort_fraction", type=float, default=1,
                        help="Fraction of clients to sample in each round (for privacy amplification by subsampling).")
    parser.add_argument("--multiplicative_margin", type=float, default=1,
                        help="Margin used in Step 3 for assigning points to initial centers.")
    parser.add_argument("--minimum_server_point_weight", type=float, default=0,
                        help="Threshold for discarding server points that are not representative of any client data.")

    # --- Privacy Arguments for Step 1 (Private PCA) ---
    parser.add_argument("--outer_product_privacy", type=str2bool, default=True,
                        help="Whether to apply DP to the outer product aggregation.")
    parser.add_argument("--outer_product_clipping_bound", type=float, default=1510,
                        help="L2 norm clipping bound for client data in Step 1.")
    parser.add_argument("--outer_product_epsilon", type=float, default=1,
                        help="Epsilon for the Gaussian mechanism in Step 1.")
    parser.add_argument("--outer_product_delta", type=float, default=1e-6,
                        help="Delta for the Gaussian mechanism in Step 1.")

    # --- Privacy Arguments for Step 2 (Server Point Weighting) ---
    parser.add_argument("--point_weighting_privacy", type=str2bool, default=True,
                        help="Whether to apply DP to the point weight aggregation.")
    parser.add_argument("--weighting_epsilon", type=float, default=1,
                        help="Epsilon for the Laplace mechanism in Step 2.")
    parser.add_argument("--weighting_clipping_bound", type=float, default=50,
                        help="L1 sensitivity (clipping bound) for the point counts in Step 2.")

    # --- Privacy Arguments for Step 3 (Center Initialization) ---
    parser.add_argument('--center_init_send_sums_and_counts', type=str2bool, default=False,
                        help="If True, clients send sums and counts. If False, they send means and contribution flags.")
    parser.add_argument("--center_init_privacy", type=str2bool, default=True,
                        help="Whether to apply DP to the final center calculation.")
    parser.add_argument("--center_init_clipping_bound", type=float, default=110,
                        help="Clipping bound for sum/mean of points in Step 3.")
    parser.add_argument("--center_init_laplace_clipping_bound", type=float, default=50,
                        help="Clipping bound for point counts in Step 3.")
    parser.add_argument("--center_init_gaussian_epsilon", type=float, default=1,
                        help="Total epsilon budget for the Gaussian mechanism part of Step 3.")
    parser.add_argument("--center_init_delta", type=float, default=1e-6,
                        help="Delta for the Gaussian mechanism in Step 3.")
    parser.add_argument("--center_init_epsilon_split", type=float, default=0.5,
                        help="Fraction of epsilon for sum of points vs. count of points in Step 3.")
    # These two are for the alternative "mean sending" strategy.
    parser.add_argument("--center_init_contributed_components_epsilon", type=float, default=10)
    parser.add_argument("--center_init_contributed_components_clipping_bound", type=float, default=30)

    parser.add_argument("--initialization_target_delta", type=float, default=1e-6,
                        help="The target delta for the entire initialization phase, used for privacy accounting.")

    return parser


def add_fedlloyds_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds all hyperparameters related to the FedDP-Lloyds algorithm.
    Paper Connection: These parameters control the behavior of Algorithm 2.
    """
    parser.add_argument('--send_sums_and_counts', type=str2bool, default=True,
                        help="If True, clients send sums and counts. If False, they send means.")
    parser.add_argument("--fedlloyds_cohort_size", type=int, default=100,
                        help="Number of clients to sample per round.")
    parser.add_argument("--fedlloyds_num_iterations", type=int, default=15,
                        help="Total number of iterative refinement rounds to run.")

    # --- Privacy Arguments for FedDP-Lloyds ---
    parser.add_argument("--fedlloyds_privacy", type=str2bool, default=True,
                        help="Whether to apply DP to the Lloyds updates.")
    parser.add_argument("--fedlloyds_epsilon", type=float, default=1,
                        help="Total epsilon budget for all iterations of FedDP-Lloyds.")
    parser.add_argument("--fedlloyds_epsilon_split", type=float, default=0.5,
                        help='Fraction of epsilon for sum of points vs. count of points.')
    parser.add_argument("--fedlloyds_delta", type=float, default=1e-6,
                        help="Total delta budget for all iterations of FedDP-Lloyds.")
    parser.add_argument("--fedlloyds_clipping_bound", type=float, default=85,
                        help="Clipping bound for sum/mean of points.")
    parser.add_argument("--fedlloyds_laplace_clipping_bound", type=float, default=50,
                        help="Clipping bound for point counts.")
    # These two are for the alternative "mean sending" strategy.
    parser.add_argument("--fedlloyds_contributed_components_epsilon", type=float, default=0.2)
    parser.add_argument("--fedlloyds_contributed_components_clipping_bound", type=float, default=10)

    # --- Reconstruction Attack Data Saving Arguments ---
    parser.add_argument("--save_reconstruction_data", type=str2bool, default=False,
                        help="If True, save intermediate data for reconstruction attack simulation.")
    parser.add_argument("--recon_target_client_id", type=str, default=None,
                        help="Client ID to target for saving reconstruction data.")
    parser.add_argument("--recon_target_iteration", type=int, default=0,
                        help="Iteration number (0-indexed) to save reconstruction data for.")

    return parser


def add_kfed_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds hyperparameters for the k-FED baseline algorithm.
    Paper Connection: This relates to the `k-FED` baseline described in Section 5.
    """
    parser.add_argument('--K_client', type=int, default=10,
                        help="Number of local clusters each client finds in the k-FED algorithm.")

    return parser
