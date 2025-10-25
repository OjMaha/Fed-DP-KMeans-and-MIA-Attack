import argparse
import numpy as np
import os
import random
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise_distances
from typing import List, Tuple

from pfl.data.dataset import Dataset
from pfl.metrics import StringMetricName
from pfl.model.base import EvaluatableModel
from pfl.privacy.privacy_mechanism import PrivacyMechanism

from privacy import compute_privacy_accounting

# =================================================================================================
# Paper Connection: This file contains various helper functions used across the project.
# These include functions for evaluation, implementing baseline algorithms, and other
# mathematical operations that are not part of the core federated logic.
# =================================================================================================


def set_seed(seed):
    """
    Sets the random seed for both numpy and Python's built-in random module.
    This is crucial for ensuring the reproducibility of experiments.
    """
    np.random.seed(seed)
    random.seed(seed)


def shorten_name(arg_name):
    """
    A utility function, likely for creating shorter names for logging or file paths,
    though it is not actively used in the provided version of the codebase.
    It creates an acronym from an underscore-separated string (e.g., 'outer_product_privacy' -> 'opp').
    """
    return ''.join([s[0] for s in arg_name.split('_')])


def make_results_path(privacy_type:str, dataset: str):
    """
    Creates the directory structure for saving experiment results (e.g., the final pickle file).
    It organizes results by privacy level (data-point vs client) and by dataset name.
    """
    path = os.path.join('results', privacy_type, dataset)
    os.makedirs(path, exist_ok=True)
    return path


def str2bool(v):
    """
    A helper function to allow boolean arguments to be parsed from the command line.
    It correctly interprets strings like 'true', 'yes', '1' as True and 'false', 'no', '0' as False.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def project_subspace(V: np.array, U: np.array):
    """
    Projects each row vector in matrix U onto the linear subspace spanned by the row vectors of matrix V.
    Paper Connection: This is a direct implementation of the projection operation that is fundamental
    to the FedDP-Init algorithm (Section 3.1). The matrix V represents the projection matrix Π,
    which is computed from the top-k singular vectors in Step 1. This function is then used in
    all three steps of FedDP-Init to project both client and server data.
    """
    return (V.T @ ((V @ U.T) / np.sum(V * V, axis=1, keepdims=True))).T


def compute_num_correct(y_pred: np.array, y_true: np.array):
    """
    Computes clustering accuracy, which is a non-trivial metric for an unsupervised task.
    Since cluster labels are arbitrary (e.g., predicted cluster '0' might correspond to true
    label '3'), this function first finds the best possible matching between the predicted
    cluster indices and the true labels and then calculates the accuracy based on this optimal mapping.
    It does this by finding the majority true label within each predicted cluster.
    """
    num_correct = 0
    # Iterate through each unique predicted cluster label.
    for j in np.unique(y_pred):
        # Get all the true labels for data points assigned to this predicted cluster.
        true_labels_in_cluster = y_true[y_pred == j]
        if len(true_labels_in_cluster) > 0:
            # Find the most frequent true label in this cluster.
            counts = np.bincount(true_labels_in_cluster)
            majority_label_count = counts.max()
            # The number of "correct" points for this cluster is the count of the majority true label.
            num_correct += majority_label_count
    return num_correct


def kmeans_cost(P, centers, y_pred=None):
    """
    Calculates the standard k-means cost, which is the sum of squared Euclidean distances
    from each data point to its assigned cluster center.
    Paper Connection: This metric is used for the y-axis in all the result plots (e.g., Figure 1, 2).
    """
    if y_pred is None:
        # If assignments aren't pre-computed, find the closest center for each point.
        dist_matrix = pairwise_distances(P, centers)
        y_pred = np.argmin(dist_matrix, axis=1)

    # Calculate the sum of squared distances.
    return ((P - centers[y_pred]) ** 2).sum()


def awasthisheffet_kmeans(P: np.array, k: int, max_iter: int = 10, random_svd: bool = True, mult_margin: float=0.9):
    """
    Implements the centralized k-means clustering algorithm from the paper "Improved spectral-norm
    bounds for clustering" by Awasthi & Sheffet (2012).
    Paper Connection: This specific algorithm is used as the local clustering method within the
    `k-FED` baseline algorithm (from Dennis et al., 2021), which is one of the key baselines
    compared against in Section 5 of the main paper.
    """
    # Part 1a: Project the data P onto the subspace spanned by its top k singular vectors (PCA).
    if random_svd:
        _, _, V = randomized_svd(P, n_components=k, random_state=None)
    else:
        _, _, V = svd(P)
        V = V[:k]
    P_proj = project_subspace(V, P)

    # Part 1b: Cluster the projected points using standard k-means.
    projected_kmeans = KMeans(n_clusters=k, n_init='auto').fit(P_proj)
    projected_centers = projected_kmeans.cluster_centers_

    # Part 2: Use the clustering in the projected space to get a good set of initial centers
    # in the original, high-dimensional space. This uses a proximity condition to select points
    # that are confidently assigned to a cluster.
    dist_matrix = pairwise_distances(P_proj, projected_centers)
    center_assignments = -1 * np.ones(len(P_proj), dtype=int)
    smallest_two_distances = np.partition(dist_matrix, 1, axis=1)[:, :2]
    assignment_mask = smallest_two_distances[:, 0] <= (mult_margin * smallest_two_distances[:, 1])
    center_assignments[assignment_mask] = np.argmin(dist_matrix, axis=1)[assignment_mask]
    initial_centers = []
    for j in range(k):
        # The initial centers are the means of the confidently assigned points.
        initial_centers.append(np.mean(P[center_assignments == j], axis=0))
    initial_centers = np.vstack(initial_centers)

    # Part 3: Run standard Lloyd's algorithm for a few iterations using these good initial centers.
    original_space_kmeans = KMeans(n_clusters=k, init=initial_centers, max_iter=max_iter, n_init='auto').fit(P)

    return original_space_kmeans.labels_, original_space_kmeans.cluster_centers_


def post_evaluation(results_dict: dict, args: argparse.Namespace, model: EvaluatableModel,
                    data: Tuple[Dataset], when: str, executed_privacy_mechanisms: List[PrivacyMechanism],
                    num_compositions: List[int], sampling_probs: List[float], verbose: bool):
    """
    A centralized function to run evaluation, compute the final privacy cost for a sequence
    of operations, and store all results in a dictionary for later analysis and plotting.
    """
    # Evaluate the model on both the training and validation client sets.
    metrics_train = model.evaluate(data[0], lambda s: StringMetricName(s))
    metrics_val = model.evaluate(data[1], lambda s: StringMetricName(s))

    train_cost = metrics_train['kmeans-cost'].overall_value
    train_acc = metrics_train['kmeans-accuracy'].overall_value

    val_cost = metrics_val['kmeans-cost'].overall_value
    val_acc = metrics_val['kmeans-accuracy'].overall_value

    if verbose:
        # Print results to the console during the run.
        print()
        print(f'Evaluating {when}...')
        print(f"Train Cost {train_cost:.4f}, Train Accuracy {train_acc:.4f}")
        print(f"Val Cost {val_cost:.4f}, Val Accuracy {val_acc:.4f}")
        print()

    # --- Privacy Accounting ---
    # Paper Connection: This is the practical implementation of the "strong composition"
    # mentioned in Section 5. It uses the `dp-accounting` library to get a tight
    # calculation of the total privacy cost (epsilon) after composing all the private
    # steps that have been executed so far.
    if when != 'Optimal':
        # The target delta can be different for the initialization phase vs. the whole algorithm.
        if when == 'Initialization':
            target_delta = args.initialization_target_delta
        elif when == 'Clustering':
            target_delta = args.overall_target_delta
        else:
            raise ValueError('When not recognized.')

        # Call the privacy accountant to compute the final epsilon.
        accountant_epsilon = compute_privacy_accounting(
            executed_privacy_mechanisms,
            target_delta,
            num_compositions=num_compositions,
            sampling_probs=sampling_probs
        )
        privacy_cost = (accountant_epsilon, target_delta)
        if verbose:
            print(f'{when} is ({accountant_epsilon}, {target_delta})-DP')
            print()
    else:
        # The "Optimal" baseline is non-private.
        privacy_cost = (np.inf, 0)

    # Store all the computed metrics and the privacy cost in the results dictionary.
    results_dict[when] = {
        'Train client cost': train_cost,
        'Train client accuracy': train_acc,
        'Val client cost': val_cost,
        'Val client accuracy': val_acc,
        'Privacy cost': privacy_cost,
    }


def generate_sphere_packing_centroids(a, r, d, k, num_trys):
    """
    Helper function for `kmeans_initialise_sphere_packing`. It attempts to sample `k` points
    that are all at least `2*a` distance from each other within a hypercube of radius `r`.
    """
    i = 0
    centroids = []
    while len(centroids) < k:
        new_point = np.random.uniform(-r, r, size=d)
        # Check distance to the corners of the hypercube.
        closest_corner = np.sign(new_point) * r
        dist_to_corner = np.sqrt(np.sum((new_point - closest_corner)**2))
        if dist_to_corner > a:
            if not centroids:
                centroids.append(new_point)
            else:
                # Check distance to already sampled centroids.
                centroids_numpy = np.array(centroids)
                dist_to_centroids = np.min(np.sqrt(np.sum((centroids_numpy - new_point)**2, axis=1)))
                if dist_to_centroids > 2 * a:
                    centroids.append(new_point)
        i += 1
        if i == num_trys:
            break
    return len(centroids) == k, centroids


def kmeans_initialise_sphere_packing(r, d, k, num_trys, tol, max_iters_binary_search):
    """
    Paper Connection: Implements the "Sphere Packing" baseline initialization from Su et al. (2017),
    as described in the baselines of Section 5. It is a data-independent method that tries to find
    an initial set of `k` centers that are maximally spaced apart from each other.
    It works by using a binary search to find the largest possible separation distance `a` for
    which `k` points can be successfully sampled.
    """
    rad_low, rad_high = 0, r * np.sqrt(d)
    iter_count = 0
    success = False
    # Binary search for the largest feasible separation radius.
    while (rad_high - rad_low > tol) and not success:
        rad_mid = (rad_low + rad_high) / 2
        success, centroids = generate_sphere_packing_centroids(rad_mid, r, d, k, num_trys)
        if success:
            rad_low = rad_mid
        else:
            rad_high = rad_mid

        iter_count += 1
        if iter_count == max_iters_binary_search:
            print(f'Binary search failed to converge after {max_iters_binary_search} iterations.')
            break

    return np.array(centroids)