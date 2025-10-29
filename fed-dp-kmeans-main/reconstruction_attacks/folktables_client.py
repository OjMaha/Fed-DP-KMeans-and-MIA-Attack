import argparse
import subprocess
import os
import numpy as np
import yaml
import random
from sklearn.metrics import pairwise_distances
import sys
import math

from data import make_data, set_data_args, add_data_arguments
from utils import kmeans_cost, add_utils_arguments, set_seed
from utils.argument_parsing import maybe_inject_arguments_from_config
from pfl.data.sampling import get_user_sampler
from pfl.data.federated_dataset import FederatedDataset
from pfl.stats import MappedVectorStatistics
from privacy.utils import get_mechanism
from algorithms import add_algorithms_arguments

# Define the list of state abbreviations
STATE_LIST = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']


# --- create configs ---
def create_recon_configs_folktables_client(base_config_path='configs/folktables.yaml'):
    """Creates non-private and private config files for data-point level attack on Folktables."""
    try:
        with open(base_config_path, 'r') as f: base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Base config file '{base_config_path}' not found. Using defaults.")
        base_config = {'dataset': 'folktables', 'filter_label': 2, 'K': 10, 'samples_per_mixture_server': 10,'num_uniform_server': 1000, 'initialization_algorithm': 'FederatedClusterInitExact','clustering_algorithm': 'FederatedLloyds', 'minimum_server_point_weight': 5, 'fedlloyds_num_iterations': 1,'datapoint_privacy': True, 'outer_product_epsilon': 1, 'weighting_epsilon': 1,'center_init_gaussian_epsilon': 1, 'center_init_epsilon_split': 0.5, 'fedlloyds_epsilon': 1,'fedlloyds_epsilon_split': 0.5, 'outer_product_clipping_bound': 2.65, 'weighting_clipping_bound': 1,'center_init_clipping_bound': 2.65, 'center_init_laplace_clipping_bound': 1, 'fedlloyds_clipping_bound': 2.65,'fedlloyds_laplace_clipping_bound': 1, 'overall_target_delta': 1e-6, 'fedlloyds_delta': 1e-6,'send_sums_and_counts': True}
    except Exception as e:
        print(f"Error loading base config '{base_config_path}': {e}. Using defaults.")
        base_config = {'dataset': 'folktables', 'filter_label': 2, 'K': 10, 'samples_per_mixture_server': 10,'num_uniform_server': 1000, 'initialization_algorithm': 'FederatedClusterInitExact','clustering_algorithm': 'FederatedLloyds', 'minimum_server_point_weight': 5, 'fedlloyds_num_iterations': 1,'datapoint_privacy': True, 'outer_product_epsilon': 1, 'weighting_epsilon': 1,'center_init_gaussian_epsilon': 1, 'center_init_epsilon_split': 0.5, 'fedlloyds_epsilon': 1,'fedlloyds_epsilon_split': 0.5, 'outer_product_clipping_bound': 2.65, 'weighting_clipping_bound': 1,'center_init_clipping_bound': 2.65, 'center_init_laplace_clipping_bound': 1, 'fedlloyds_clipping_bound': 2.65,'fedlloyds_laplace_clipping_bound': 1, 'overall_target_delta': 1e-6, 'fedlloyds_delta': 1e-6,'send_sums_and_counts': True}

    # Ensure necessary keys are present
    base_config.setdefault('fedlloyds_num_iterations', 1); base_config.setdefault('num_train_clients', 51)
    base_config.setdefault('fedlloyds_cohort_size', base_config.get('num_train_clients', 51))
    base_config.setdefault('send_sums_and_counts', True); base_config.setdefault('datapoint_privacy', True)
    base_config.setdefault('fedlloyds_epsilon', 1); base_config.setdefault('fedlloyds_epsilon_split', 0.5)
    default_delta = base_config.get('overall_target_delta', 1e-6)
    base_config.setdefault('overall_target_delta', default_delta); base_config.setdefault('fedlloyds_delta', default_delta)
    base_config.setdefault('fedlloyds_clipping_bound', 2.65); base_config.setdefault('fedlloyds_laplace_clipping_bound', 1)

    os.makedirs("reconstruction_attacks/configs", exist_ok=True)

    # Config 1: Non-Private (DP flags off)
    config_non_private = base_config.copy(); config_non_private.update({'datapoint_privacy': False, 'outer_product_privacy': False, 'point_weighting_privacy': False,'center_init_privacy': False, 'fedlloyds_privacy': False,'fedlloyds_num_iterations': 1})
    config_non_private_fname = 'reconstruction_attacks/configs/folktables_client_non_private.yaml'
    with open(config_non_private_fname, 'w') as f: yaml.dump(config_non_private, f, sort_keys=False)

    # Config 2: Private (Datapoint privacy false, rest True as client level attack)
    config_private = base_config.copy(); config_private.update({'datapoint_privacy': False, 'outer_product_privacy': True, 'point_weighting_privacy': True,'center_init_privacy': True, 'fedlloyds_privacy': True,'fedlloyds_num_iterations': 1})
    config_private['fedlloyds_clipping_bound'] =  100000000000
    config_private['fedlloyds_laplace_clipping_bound'] = 100000000000
    config_private.setdefault('fedlloyds_delta', config_private.get('overall_target_delta', 1e-6))
    config_private_fname = 'reconstruction_attacks/configs/folktables_client_private.yaml'
    with open(config_private_fname, 'w') as f: yaml.dump(config_private, f, sort_keys=False)

    print("Folktables single-point reconstruction attack config files created.")
    return config_non_private_fname, config_private_fname

# get_target_data (Identical to previous Folktables script)
def get_target_data(target_client_id_str, all_train_clients):
    user_sampler = get_user_sampler('minimize_reuse', [target_client_id_str])
    target_dataset = FederatedDataset(all_train_clients.make_dataset_fn, user_sampler)
    (user_dataset, _) = next(target_dataset.get_cohort(1))
    data = user_dataset.raw_data[0]
    if hasattr(data, 'numpy'): data = data.numpy()
    if data.dtype == bool: data = data.astype(np.float32)
    return data

# run_training_get_centers (Identical)
def run_training_get_centers(config_file, exclude_client_id_str=None, exclude_datapoint_str=None, seed=None):
    cmd = ['python', 'run.py', '--args_config', config_file]
    if exclude_client_id_str: cmd.extend(['--exclude_client_id_str', exclude_client_id_str])
    if exclude_datapoint_str: cmd.extend(['--exclude_datapoint', exclude_datapoint_str]) # Keep option
    if seed is not None: cmd.extend(['--seed', str(seed)])
    print(f"\nRunning command: {' '.join(cmd)}")
    try: subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e: print(f"Error running training: {e}"); return None
    except FileNotFoundError: print("Error: 'python' command not found."); return None
    center_file = 'final_centers.npy'
    if not os.path.exists(center_file): print(f"Error: Center file '{center_file}' not found."); return None
    return center_file

# load_config_as_namespace (Adapted for Folktables Client defaults )
def load_config_as_namespace(config_file):
    with open(config_file, 'r') as f: config_dict = yaml.safe_load(f)
    # Add ALL missing defaults
    config_dict.setdefault('num_train_clients', 51); config_dict.setdefault('send_sums_and_counts', True)
    config_dict.setdefault('center_init_send_sums_and_counts', False) 
    config_dict.setdefault('datapoint_privacy', False) # Default False
    default_delta = config_dict.get('overall_target_delta', 1e-6); config_dict.setdefault('overall_target_delta', default_delta)
    config_dict.setdefault('fedlloyds_num_iterations', 1); config_dict.setdefault('fedlloyds_cohort_size', config_dict.get('num_train_clients', 51))
    # Fedlloyds params
    config_dict.setdefault('fedlloyds_epsilon', 1.0); config_dict.setdefault('fedlloyds_epsilon_split', 0.5)
    config_dict.setdefault('fedlloyds_delta', default_delta)
    config_dict.setdefault('fedlloyds_clipping_bound', 2.65) 
    config_dict.setdefault('fedlloyds_laplace_clipping_bound', 1) 
    config_dict.setdefault('fedlloyds_contributed_components_epsilon', 0.2) 
    config_dict.setdefault('fedlloyds_contributed_components_clipping_bound', 10) 
    # Other mechanisms defaults 
    config_dict.setdefault('outer_product_epsilon', 1.0); config_dict.setdefault('outer_product_delta', default_delta)
    config_dict.setdefault('outer_product_clipping_bound', 2.65)
    config_dict.setdefault('weighting_epsilon', 1.0); config_dict.setdefault('weighting_clipping_bound', 1)
    config_dict.setdefault('center_init_gaussian_epsilon', 1.0); config_dict.setdefault('center_init_delta', default_delta)
    config_dict.setdefault('center_init_epsilon_split', 0.5); config_dict.setdefault('center_init_clipping_bound', 2.65)
    config_dict.setdefault('center_init_laplace_clipping_bound', 1)
    config_dict.setdefault('center_init_contributed_components_epsilon', 0.2) 
    config_dict.setdefault('center_init_contributed_components_clipping_bound', 10) 
    config_dict.setdefault('filter_label', 5) 
    return argparse.Namespace(**config_dict)

# simulate_client_contribution 
def simulate_client_contribution(client_data, global_centers, config_namespace, seed=None):
    # (Code identical to Gaussian script, relies on get_mechanism with datapoint_privacy=False)
    K = global_centers.shape[0]; dim = global_centers.shape[1]
    if client_data.dtype != np.float32: client_data = client_data.astype(np.float32)
    if global_centers.dtype != np.float32: global_centers = global_centers.astype(np.float32)

    if client_data.shape[0] == 0: raw_sums = np.zeros((K, dim), dtype=np.float32); raw_counts = np.zeros(K, dtype=np.float32)
    else:
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

# run_reconstruction_once_client_mean (Logic identical)
def run_reconstruction_once_client_mean(config_file, target_client_id_str, target_client_data, seed=None):
    """Runs one iteration of the client mean reconstruction attack."""
    try:
        config_namespace = load_config_as_namespace(config_file)
        if not config_namespace.send_sums_and_counts: print(f"Error: Config {config_file} needs send_sums_and_counts=True."); return np.nan
    except Exception as e: print(f"Error loading config {config_file}: {e}"); return np.nan

    print("Running training without target CLIENT to get global centers...")
    train_seed = seed + 50 if seed is not None else None
    centers_file = run_training_get_centers(config_file, exclude_client_id_str=target_client_id_str, seed=train_seed)
    if centers_file is None: print("Failed to get global centers."); return np.nan
    try: global_centers = np.load(centers_file).astype(np.float32)
    except Exception as e: print(f"Error loading centers {centers_file}: {e}"); error = np.nan
    finally:
        if os.path.exists(centers_file): os.remove(centers_file)
        if 'error' in locals(): return error

    print(f"Simulating contribution for client {target_client_id_str}...")
    sim_seed = seed + 100 if seed is not None else None
    try: noisy_sums, noisy_counts = simulate_client_contribution(target_client_data, global_centers, config_namespace, seed=sim_seed)
    except Exception as e: print(f"Error simulating client contribution: {e}"); return np.nan

    total_noisy_sum = np.sum(noisy_sums, axis=0); total_noisy_count = np.sum(noisy_counts)

    if total_noisy_count <= 1e-9:
        print("Warning: Total noisy count near zero. Recon failed."); error = np.nan
    else:
        reconstructed_mean = total_noisy_sum / total_noisy_count
        true_mean = np.mean(target_client_data, axis=0)
        if not np.isnan(reconstructed_mean).any():
             error = np.sum((true_mean - reconstructed_mean)**2)
             print(f"Target {target_client_id_str}: True Mean Norm={np.linalg.norm(true_mean):.4f}, Recon Mean Norm={np.linalg.norm(reconstructed_mean):.4f}, Sq L2 Error={error:.6f}")
        else: error = np.nan; print(f"Target {target_client_id_str}: Reconstruction resulted in NaN.")
    return error

# --- main (Adapted for Folktables Client Level) ---
def main():
    parser = argparse.ArgumentParser(description="Client Mean Reconstruction Attack (Folktables - Client Privacy from MIA params)")
    parser.add_argument("--num_attacks", type=int, default=10, help="Number of clients to attack (max 51).")
    parser.add_argument("--base_config", type=str, default="configs/folktables.yaml", help="Base config for Folktables DP settings.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    parser.add_argument("--filter_label", type=int, choices=[2, 5, 6], default=5, help="Folktables filter label (from MIA script default).")
    args, unknown = parser.parse_known_args()

    if args.seed is not None: set_seed(args.seed); print(f"Set global seed to {args.seed}")

    print("--- 1. Creating Folktables Single-Point Reconstruction Attack config files ---")
    try:
        config_non_private_file, config_private_file = create_recon_configs_folktables_client(args.base_config)
        # Apply command-line filter_label if provided
        current_filter_label = None
        if args.filter_label is not None:
             current_filter_label = args.filter_label
             for fname in [config_non_private_file, config_private_file]:
                  with open(fname, 'r') as f: config = yaml.safe_load(f)
                  config['filter_label'] = args.filter_label
                  with open(fname, 'w') as f: yaml.dump(config, f, sort_keys=False)
             print(f"Using filter_label={args.filter_label} from command line.")
        else:
             with open(config_non_private_file, 'r') as f: config = yaml.safe_load(f)
             if 'filter_label' not in config: print("Error: --filter_label required if not in base config."); return
             current_filter_label = config['filter_label']
             print(f"Using filter_label={current_filter_label} from config.")
    except Exception as e: print(f"Fatal Error creating config files: {e}"); return

    print("\n--- 2. Loading client list (Folktables States) ---")
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser = add_data_arguments(temp_parser); temp_parser = add_utils_arguments(temp_parser); temp_parser = add_algorithms_arguments(temp_parser)
    original_argv = sys.argv.copy(); sys.argv = [sys.argv[0], '--args_config', config_non_private_file] # Use generated non-private client config
    maybe_inject_arguments_from_config()
    try:
        data_args, _ = temp_parser.parse_known_args()
        if args.seed is not None: data_args.data_seed = args.seed
        # Ensure filter_label from args is used for data loading
        data_args.filter_label = args.filter_label
        set_data_args(data_args)
    except Exception as e:
         print(f"Error parsing data arguments: {e}. Using defaults."); data_args = argparse.Namespace(dataset='folktables', num_train_clients=51, exclude_client_id_str=None, data_seed=args.seed or 0, filter_label=args.filter_label, datapoint_privacy=False); data_args.num_train_clients = 51
    finally: sys.argv = original_argv

    try:
        all_train_clients, _, _, _ = make_data(data_args)
        all_client_ids = STATE_LIST[:data_args.num_train_clients]
        print(f"Loaded {len(all_client_ids)} total clients (states) for filter_label={args.filter_label}: {all_client_ids}")
    except Exception as e: print(f"Fatal Error loading data: {e}"); print("Data Args:", data_args); return
    if not all_client_ids: print("Error: No clients loaded."); return

    num_attacks = min(args.num_attacks, len(all_client_ids))
    print(f"Will run client mean reconstruction attack on {num_attacks} random clients.")

    results = {'non_private': [], 'private': []}
    config_files = {'non_private': config_non_private_file, 'private': config_private_file}
    rng = np.random.default_rng(args.seed)

    if num_attacks >= len(all_client_ids): target_clients_sample = all_client_ids
    else: target_clients_sample = rng.choice(all_client_ids, size=num_attacks, replace=False).tolist()

    attacks_run = 0
    for i, target_client_id in enumerate(target_clients_sample):
        print(f"\n--- ATTACK ITERATION {i + 1} / {num_attacks} ---")
        print(f"Target Client (State): {target_client_id}")
        try: target_data = get_target_data(target_client_id, all_train_clients)
        except StopIteration: print(f"Error sampling client {target_client_id}."); continue
        except Exception as e: print(f"Error loading data for {target_client_id}: {e}"); continue
        if target_data.shape[0] == 0: print(f"Client {target_client_id} has no data."); continue
        print(f"Target data loaded. Shape: {target_data.shape}")

        attack_successful_this_iter = False; current_iter_results = {}
        for mode in ['non_private', 'private']:
            print(f"--- Running {mode.upper()} scenario ---")
            config_file = config_files[mode]
            iter_seed = (args.seed + i*10 + (0 if mode == 'non_private' else 1)) if args.seed is not None else None
            try:
                error = run_reconstruction_once_client_mean(config_file, target_client_id, target_data, seed=iter_seed)
                if not np.isnan(error):
                    current_iter_results[mode] = error; attack_successful_this_iter = True
                else: print(f"Reconstruction failed in {mode} mode (NaN error).")
            except Exception as e: print(f"Attack failed unexpectedly for {mode}: {e}"); import traceback; traceback.print_exc()

        if 'non_private' in current_iter_results and 'private' in current_iter_results:
             results['non_private'].append(current_iter_results['non_private'])
             results['private'].append(current_iter_results['private']); attacks_run += 1
        elif attack_successful_this_iter: print("Attack completed for only one mode, results discarded.")

    if attacks_run < num_attacks: print(f"\nWarning: Only completed {attacks_run}/{num_attacks} iterations.")

    print(f"\n--- FINAL CLIENT MEAN RECONSTRUCTION RESULTS (Sq L2 Error - Folktables Client Privacy, Filter={args.filter_label}) ---")
    for mode in ['non_private', 'private']:
        errors = results[mode]; count = len(errors)
        if count > 0:
            avg_error=np.mean(errors); std_error=np.std(errors); median_error=np.median(errors); min_error=np.min(errors); max_error=np.max(errors)
            print(f"{mode.upper()} Model Results ({count} clients):")
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