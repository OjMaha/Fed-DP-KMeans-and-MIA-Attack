# --- GOAL: Reconstruct a single data point's value ---
# --- METHOD:
# --- 1. Train global model *without* target point (using --exclude_datapoint) -> centers_baseline
# --- 2. Simulate client update *with* target point (using centers_baseline) -> noisy_update_in
# --- 3. Simulate client update *without* target point (using centers_baseline) -> noisy_update_out
# --- 4. Reconstruct point from difference: noisy_update_in - noisy_update_out
# --- 5. Calculate error: || true_point - reconstructed_point ||^2
# --- Uses Data-Point Privacy settings ---

import argparse
import subprocess
import os
import numpy as np
import yaml
import random
from sklearn.metrics import pairwise_distances
import sys
import math

# Import necessary functions
from data import make_data, set_data_args, add_data_arguments
from utils import kmeans_cost, add_utils_arguments, set_seed
from utils.argument_parsing import maybe_inject_arguments_from_config
from pfl.data.sampling import get_user_sampler
from pfl.data.federated_dataset import FederatedDataset
from pfl.stats import MappedVectorStatistics
from privacy.utils import get_mechanism
from algorithms import add_algorithms_arguments

# --- create_recon_configs_datapoint
def create_recon_configs_datapoint(base_config_path='../configs/gaussians_data_privacy.yaml'):
    # (ensures DP settings)
    try:
        with open(base_config_path, 'r') as f: base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Base config file '{base_config_path}' not found. Using defaults.")
        base_config = {'dataset': 'GaussianMixtureUniform', 'K': 10, 'dim': 100,'num_train_clients': 100, 'samples_per_client': 1000,'samples_per_mixture_server': 20, 'num_uniform_server': 100,'initialization_algorithm': 'FederatedClusterInitExact', 'clustering_algorithm': 'FederatedLloyds','minimum_server_point_weight': 5, 'fedlloyds_num_iterations': 1,'datapoint_privacy': True, 'outer_product_epsilon': 1, 'weighting_epsilon': 1,'center_init_gaussian_epsilon': 1, 'center_init_epsilon_split': 0.5,'fedlloyds_epsilon': 1, 'fedlloyds_epsilon_split': 0.5,'outer_product_clipping_bound': 11, 'weighting_clipping_bound': 1,'center_init_clipping_bound': 11, 'center_init_laplace_clipping_bound': 1,'fedlloyds_clipping_bound': 11, 'fedlloyds_laplace_clipping_bound': 1,'overall_target_delta': 1e-6, 'fedlloyds_delta': 1e-6, 'send_sums_and_counts': True}
    except Exception as e:
        print(f"Error loading base config '{base_config_path}': {e}. Using defaults.")
        base_config = {'dataset': 'GaussianMixtureUniform', 'K': 10, 'dim': 100, 'num_train_clients': 100, 'samples_per_client': 1000,'samples_per_mixture_server': 20, 'num_uniform_server': 100,'initialization_algorithm': 'FederatedClusterInitExact','clustering_algorithm': 'FederatedLloyds', 'minimum_server_point_weight': 5, 'fedlloyds_num_iterations': 1,'datapoint_privacy': True, 'outer_product_epsilon': 1, 'weighting_epsilon': 1, 'center_init_gaussian_epsilon': 1,'center_init_epsilon_split': 0.5, 'fedlloyds_epsilon': 1, 'fedlloyds_epsilon_split': 0.5,'outer_product_clipping_bound': 11, 'weighting_clipping_bound': 1, 'center_init_clipping_bound': 11,'center_init_laplace_clipping_bound': 1, 'fedlloyds_clipping_bound': 11, 'fedlloyds_laplace_clipping_bound': 1,'overall_target_delta': 1e-6, 'fedlloyds_delta': 1e-6, 'send_sums_and_counts': True}
    base_config.setdefault('fedlloyds_num_iterations', 1); base_config.setdefault('fedlloyds_cohort_size', base_config.get('num_train_clients', 100))
    base_config.setdefault('num_train_clients', 100); base_config.setdefault('send_sums_and_counts', True)
    base_config.setdefault('datapoint_privacy', True); base_config.setdefault('fedlloyds_epsilon', 1)
    base_config.setdefault('fedlloyds_epsilon_split', 0.5); default_delta = base_config.get('overall_target_delta', 1e-6)
    base_config.setdefault('overall_target_delta', default_delta); base_config.setdefault('fedlloyds_delta', default_delta)
    base_config.setdefault('fedlloyds_clipping_bound', 11); base_config.setdefault('fedlloyds_laplace_clipping_bound', 1)
    config_non_private = base_config.copy(); config_non_private.update({'datapoint_privacy': True, 'outer_product_privacy': False, 'point_weighting_privacy': False,'center_init_privacy': False, 'fedlloyds_privacy': False,'fedlloyds_num_iterations': 1})
    config_non_private_fname = 'configs/gaussian_datapoint_non_private.yaml'
    with open(config_non_private_fname, 'w') as f: yaml.dump(config_non_private, f, sort_keys=False)
    config_private = base_config.copy(); config_private.update({'datapoint_privacy': True, 'outer_product_privacy': True, 'point_weighting_privacy': True,'center_init_privacy': True, 'fedlloyds_privacy': True,'fedlloyds_num_iterations': 1})
    config_private.setdefault('fedlloyds_clipping_bound', 11); config_private.setdefault('fedlloyds_laplace_clipping_bound', 1)
    config_private.setdefault('fedlloyds_delta', config_private.get('overall_target_delta', 1e-6))
    config_private_fname = 'configs/gaussian_datapoint_private.yaml'
    with open(config_private_fname, 'w') as f: yaml.dump(config_private, f, sort_keys=False)
    print("Single-point reconstruction attack config files created.")
    return config_non_private_fname, config_private_fname

# --- get_target_data ---
def get_target_data(target_client_id_str, all_train_clients):
    user_sampler = get_user_sampler('minimize_reuse', [target_client_id_str])
    target_dataset = FederatedDataset(all_train_clients.make_dataset_fn, user_sampler)
    (user_dataset, _) = next(target_dataset.get_cohort(1))
    if hasattr(user_dataset.raw_data[0], 'numpy'): return user_dataset.raw_data[0].numpy()
    return user_dataset.raw_data[0]

# --- run_training_get_centers (Runs run.py, gets centers) ---
def run_training_get_centers(config_file, exclude_client_id_str=None, exclude_datapoint_str=None, seed=None):
    # (supports both exclude flags)
    cmd = ['python', '../run.py', '--args_config', config_file]
    if exclude_client_id_str: cmd.extend(['--exclude_client_id_str', exclude_client_id_str])
    if exclude_datapoint_str: cmd.extend(['--exclude_datapoint', exclude_datapoint_str])
    if seed is not None: cmd.extend(['--seed', str(seed)])
    print(f"\nRunning command: {' '.join(cmd)}")
    try: subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e: print(f"Error running training: {e}"); return None
    except FileNotFoundError: print("Error: 'python' command not found."); return None
    center_file = 'final_centers.npy'
    if not os.path.exists(center_file): print(f"Error: Center file '{center_file}' not found."); return None
    return center_file

# --- load_config_as_namespace (Loads config + defaults) ---
def load_config_as_namespace(config_file):
    # (ensures DP defaults)
    with open(config_file, 'r') as f: config_dict = yaml.safe_load(f)
    config_dict.setdefault('num_train_clients', 100); config_dict.setdefault('send_sums_and_counts', True)
    config_dict.setdefault('center_init_send_sums_and_counts', False); config_dict.setdefault('datapoint_privacy', True)
    default_delta = config_dict.get('overall_target_delta', 1e-6); config_dict.setdefault('overall_target_delta', default_delta)
    config_dict.setdefault('fedlloyds_num_iterations', 1); config_dict.setdefault('fedlloyds_cohort_size', config_dict.get('num_train_clients', 100))
    config_dict.setdefault('fedlloyds_epsilon', 1.0); config_dict.setdefault('fedlloyds_epsilon_split', 0.5)
    config_dict.setdefault('fedlloyds_delta', default_delta); config_dict.setdefault('fedlloyds_clipping_bound', 11)
    config_dict.setdefault('fedlloyds_laplace_clipping_bound', 1)
    config_dict.setdefault('outer_product_epsilon', 1.0); config_dict.setdefault('outer_product_delta', default_delta)
    config_dict.setdefault('outer_product_clipping_bound', 11); config_dict.setdefault('weighting_epsilon', 1.0)
    config_dict.setdefault('weighting_clipping_bound', 1); config_dict.setdefault('center_init_gaussian_epsilon', 1.0)
    config_dict.setdefault('center_init_delta', default_delta); config_dict.setdefault('center_init_epsilon_split', 0.5)
    config_dict.setdefault('center_init_clipping_bound', 11); config_dict.setdefault('center_init_laplace_clipping_bound', 1)
    return argparse.Namespace(**config_dict)

# --- simulate_client_contribution (Simulates noisy update) ---
def simulate_client_contribution(client_data, global_centers, config_namespace, seed=None):
    K = global_centers.shape[0]; dim = global_centers.shape[1]
    if client_data.shape[0] == 0:
        raw_sums = np.zeros((K, dim), dtype=np.float32); raw_counts = np.zeros(K, dtype=np.float32)
    else:
        if client_data.dtype != global_centers.dtype:
             try: client_data = client_data.astype(global_centers.dtype)
             except ValueError: print("Warning: Data type mismatch, casting to float32."); client_data = client_data.astype(np.float32); global_centers = global_centers.astype(np.float32)
        dist_matrix = pairwise_distances(client_data, global_centers); assignments = np.argmin(dist_matrix, axis=1)
        raw_sums = np.zeros((K, dim), dtype=np.float32); raw_counts = np.zeros(K, dtype=np.float32)
        for k in range(K):
            mask = (assignments == k)
            if np.any(mask): raw_sums[k] = np.sum(client_data[mask], axis=0); raw_counts[k] = np.sum(mask)
    raw_stats_dict = {}
    if config_namespace.send_sums_and_counts: raw_stats_dict['sum_points_per_component'] = raw_sums; raw_stats_dict['num_points_per_component'] = raw_counts
    else: raise NotImplementedError("Only supports 'send_sums_and_counts=True'")
    raw_stats = MappedVectorStatistics(raw_stats_dict)
    mechanism_name = 'fedlloyds' if config_namespace.fedlloyds_privacy else 'no_privacy'
    try:
        if not hasattr(config_namespace, 'fedlloyds_delta'): config_namespace.fedlloyds_delta = config_namespace.overall_target_delta
        mechanism_wrapper = get_mechanism(config_namespace, mechanism_name); underlying_mechanism = mechanism_wrapper.underlying_mechanism
    except Exception as e: print(f"Error getting mechanism '{mechanism_name}': {e}"); raise
    try: sim_seed_clip = seed + 1 if seed is not None else None; clipped_stats, _ = underlying_mechanism.constrain_sensitivity(raw_stats, seed=sim_seed_clip)
    except Exception as e: print(f"Error during constrain_sensitivity: {e}"); clipped_stats = raw_stats
    try: sim_seed_noise = seed + 2 if seed is not None else None; noisy_stats, _ = underlying_mechanism.add_noise(clipped_stats, cohort_size=1, seed=sim_seed_noise)
    except Exception as e: print(f"Error during add_noise: {e}"); noisy_stats = clipped_stats
    final_noisy_sums = np.zeros_like(raw_sums); final_noisy_counts = np.zeros_like(raw_counts)
    if config_namespace.send_sums_and_counts:
        if 'sum_points_per_component' in noisy_stats: final_noisy_sums = noisy_stats['sum_points_per_component']
        else: print("Warning: 'sum_points_per_component' missing.")
        if 'num_points_per_component' in noisy_stats: final_noisy_counts = noisy_stats['num_points_per_component']
        else: print("Warning: 'num_points_per_component' missing.")
    else: raise NotImplementedError("Only supports 'send_sums_and_counts=True'")
    final_noisy_counts = np.maximum(0, final_noisy_counts); return final_noisy_sums, final_noisy_counts

def run_reconstruction_once_single_point(config_file, target_client_id_str, target_sample_idx,
                                         target_datapoint_vector, full_client_data, seed=None):
    """
    Runs one iteration of the single data point reconstruction attack.
    Uses baseline centers trained *without* the target point.
    Returns the squared L2 error between true and reconstructed point.
    """
    try:
        config_namespace = load_config_as_namespace(config_file)
        if not config_namespace.send_sums_and_counts:
            print(f"Error: Config {config_file} has send_sums_and_counts=False. Skipping."); return np.nan
    except Exception as e: print(f"Error loading config {config_file}: {e}"); return np.nan

    # 1. Get Global Centers (Trained WITHOUT the target data point)
    print("Running training WITHOUT target POINT to get baseline global centers...")
    exclude_str = f"{target_client_id_str}:{target_sample_idx}"
    train_seed = seed + 50 if seed is not None else None # Seed for training run
    centers_file = run_training_get_centers(config_file, exclude_datapoint_str=exclude_str, seed=train_seed)

    if centers_file is None: print("Failed to get baseline global centers. Skipping."); return np.nan
    try: global_centers_baseline = np.load(centers_file) # Renamed for clarity
    except Exception as e: print(f"Error loading centers {centers_file}: {e}"); error = np.nan
    finally:
        if os.path.exists(centers_file): os.remove(centers_file)
        if 'error' in locals(): return error

    # 2. Simulate client contribution WITH the target point (using baseline centers)
    print(f"Simulating contribution WITH target point {target_client_id_str}:{target_sample_idx}...")
    sim_seed_in = seed + 101 if seed is not None else None
    try:
        noisy_sums_in, noisy_counts_in = simulate_client_contribution(
            full_client_data, global_centers_baseline, config_namespace, seed=sim_seed_in # Use baseline centers
        )
    except Exception as e: print(f"Error simulating IN contribution: {e}"); return np.nan

    # 3. Simulate client contribution WITHOUT the target point (using baseline centers)
    print(f"Simulating contribution WITHOUT target point {target_client_id_str}:{target_sample_idx}...")
    client_data_out = np.delete(full_client_data, target_sample_idx, axis=0)
    sim_seed_out = seed + 102 if seed is not None else None
    try:
        noisy_sums_out, noisy_counts_out = simulate_client_contribution(
            client_data_out, global_centers_baseline, config_namespace, seed=sim_seed_out # Use baseline centers
        )
    except Exception as e: print(f"Error simulating OUT contribution: {e}"); return np.nan

    # 4. Calculate the difference (noisy contribution of the single point)
    diff_sums = noisy_sums_in - noisy_sums_out
    diff_counts = noisy_counts_in - noisy_counts_out

    # 5. Reconstruct: Find cluster k* with max count difference, use corresponding sum difference
    if diff_counts.size == 0:
        print("Error: diff_counts is empty."); reconstructed_point = np.full(target_datapoint_vector.shape[1], np.nan)
    else:
        tie_breaker = np.random.randn(*diff_counts.shape) * 1e-9
        probable_cluster_idx = np.argmax(diff_counts + tie_breaker)
        reconstructed_point = diff_sums[probable_cluster_idx]

    # 6. Calculate Squared L2 Error
    if np.isnan(reconstructed_point).any():
        error = np.nan
        print(f"Target {target_client_id_str}:{target_sample_idx}: Reconstruction failed (NaN).")
    else:
        error = np.sum((target_datapoint_vector - reconstructed_point)**2)
        print(f"Target {target_client_id_str}:{target_sample_idx}: True Norm={np.linalg.norm(target_datapoint_vector):.4f}, Recon Norm={np.linalg.norm(reconstructed_point):.4f}, Sq L2 Error={error:.6f}")

    return error

def main():
    parser = argparse.ArgumentParser(description="Single Data Point Reconstruction (Gaussian Dataset - DP)")
    parser.add_argument("--num_attacks", type=int, default=50, help="Number of attack iterations (target data points).")
    parser.add_argument("--base_config", type=str, default="../configs/gaussians_data_privacy.yaml", help="Base config for DP settings.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    args, unknown = parser.parse_known_args()

    if args.seed is not None: set_seed(args.seed); print(f"Set global seed to {args.seed}")

    print("--- 1. Creating Single-Point Reconstruction Attack config files ---")
    try: config_non_private_file, config_private_file = create_recon_configs_datapoint(args.base_config)
    except Exception as e: print(f"Fatal Error creating config files: {e}"); return

    print("\n--- 2. Loading client list ---")
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser = add_data_arguments(temp_parser); temp_parser = add_utils_arguments(temp_parser); temp_parser = add_algorithms_arguments(temp_parser)
    original_argv = sys.argv.copy(); sys.argv = [sys.argv[0], '--args_config', config_non_private_file]
    maybe_inject_arguments_from_config()
    try:
        data_args, _ = temp_parser.parse_known_args()
        if args.seed is not None: data_args.data_seed = args.seed
        set_data_args(data_args)
    except Exception as e:
         print(f"Error parsing data arguments: {e}. Using defaults."); data_args = argparse.Namespace(dataset='GaussianMixtureUniform', num_train_clients=100, exclude_client_id_str=None, data_seed=args.seed or 0, K=10, dim=100, samples_per_client=1000, samples_per_mixture_server=20, num_uniform_server=100, variance=0.5, datapoint_privacy=True)
         try:
             with open(config_non_private_file, 'r') as f: cfg = yaml.safe_load(f); data_args.num_train_clients = cfg.get('num_train_clients', 100)
         except: data_args.num_train_clients = 100
    finally: sys.argv = original_argv

    try: all_train_clients, _, _, _ = make_data(data_args)
    except Exception as e: print(f"Fatal Error loading data: {e}"); print("Data Args:", data_args); return

    all_client_ids = [str(i) for i in range(data_args.num_train_clients)]
    print(f"Loaded {len(all_client_ids)} total clients.")
    if not all_client_ids: print("Error: No clients loaded."); return

    num_attacks = args.num_attacks
    print(f"Will run reconstruction attack on {num_attacks} random data points.")

    results = {'non_private': [], 'private': []}
    config_files = {'non_private': config_non_private_file, 'private': config_private_file}
    rng = np.random.default_rng(args.seed)

    attacks_run = 0; client_sample_attempts = 0; max_client_attempts = num_attacks * 5
    while attacks_run < num_attacks and client_sample_attempts < max_client_attempts:
        client_sample_attempts += 1
        target_client_id = rng.choice(all_client_ids)
        print(f"\n--- Trying Client {target_client_id} (Attempt {client_sample_attempts}) ---")
        try: full_client_data = get_target_data(target_client_id, all_train_clients)
        except StopIteration: print(f"Error sampling client {target_client_id}."); continue
        except Exception as e: print(f"Error loading data for {target_client_id}: {e}"); continue
        if full_client_data.shape[0] == 0: print(f"Client {target_client_id} has no data."); continue

        target_sample_idx = rng.integers(0, full_client_data.shape[0])
        target_datapoint_vector = full_client_data[target_sample_idx:target_sample_idx+1]
        print(f"Target point: Client {target_client_id}, Index {target_sample_idx} (of {full_client_data.shape[0]})")

        attack_successful_this_iter = False; current_iter_results = {}
        for mode in ['non_private', 'private']:
            print(f"--- Running {mode.upper()} scenario ---")
            config_file = config_files[mode]
            iter_seed = (args.seed + attacks_run*10 + (0 if mode == 'non_private' else 1)) if args.seed is not None else None
            try:
                #Call the reconstruction function
                error = run_reconstruction_once_single_point(
                    config_file, target_client_id, target_sample_idx,
                    target_datapoint_vector, full_client_data, seed=iter_seed
                )
                if not np.isnan(error):
                    current_iter_results[mode] = error
                    attack_successful_this_iter = True
                else: print(f"Reconstruction failed in {mode} mode (NaN error).")
            except Exception as e: print(f"Attack failed unexpectedly for {mode}: {e}"); import traceback; traceback.print_exc()

        if 'non_private' in current_iter_results and 'private' in current_iter_results:
             results['non_private'].append(current_iter_results['non_private'])
             results['private'].append(current_iter_results['private'])
             attacks_run += 1
        elif attack_successful_this_iter: print("Attack completed for only one mode, results discarded.")

    if attacks_run < num_attacks: print(f"\nWarning: Only completed {attacks_run}/{num_attacks} iterations.")

    print("\n--- FINAL SINGLE-POINT RECONSTRUCTION RESULTS (Squared L2 Error) ---")
    for mode in ['non_private', 'private']:
        errors = results[mode]; count = len(errors)
        if count > 0:
            avg_error=np.mean(errors); std_error=np.std(errors); median_error=np.median(errors); min_error=np.min(errors); max_error=np.max(errors)
            print(f"{mode.upper()} Model Results ({count} points):")
            print(f"  Avg Error: {avg_error:.6f}\n  Std Dev: {std_error:.6f}\n  Median : {median_error:.6f}\n  Min    : {min_error:.6f}\n  Max    : {max_error:.6f}")
        else: print(f"{mode.upper()} Model: No successful attacks.")

    print("\nCleaning up config files...")
    try:
        if os.path.exists(config_non_private_file): os.remove(config_non_private_file)
        if os.path.exists(config_private_file): os.remove(config_private_file)
        if os.path.exists('final_centers.npy'): os.remove('final_centers.npy')
        print("Cleanup complete.")
    except OSError as e: print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()