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
from pfl.stats import MappedVectorStatistics # Needed for creating stats object
# Import the mechanism factory and potentially mechanisms if type hinting needed
from privacy.utils import get_mechanism
from algorithms import add_algorithms_arguments


def create_recon_configs(base_config_path='../configs/gaussians_client_privacy.yaml'):
    """Creates non-private and private config files based on a template."""
    try:
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Base config file '{base_config_path}' not found. Using default Gaussian settings.")
        # Default settings similar to gaussians_client_privacy.yaml
        base_config = {
            'dataset': 'GaussianMixtureUniform', 'K': 10, 'dim': 100,
            'num_train_clients': 100, 'samples_per_client': 1000,
            'samples_per_mixture_server': 20, 'num_uniform_server': 100,
            'initialization_algorithm': 'FederatedClusterInitExact', 
            'clustering_algorithm': 'FederatedLloyds',
            'minimum_server_point_weight': 5, 'fedlloyds_num_iterations': 1, # Run 1 iteration
            # Client-level privacy parameters
            'datapoint_privacy': False,
            'outer_product_epsilon': 1, 'weighting_epsilon': 1,
            'center_init_gaussian_epsilon': 1, 'center_init_contributed_components_epsilon': 0.2,
            'fedlloyds_epsilon': 1, 'fedlloyds_epsilon_split': 0.5,
            'outer_product_clipping_bound': 1500, 'weighting_clipping_bound': 1,
            'center_init_clipping_bound': 21, 'center_init_contributed_components_clipping_bound': 10,
            'fedlloyds_clipping_bound': 120, 'fedlloyds_laplace_clipping_bound': 50,
            'overall_target_delta': 1e-6,
            'fedlloyds_delta': 1e-6, # Make sure fedlloyds_delta is present
            'send_sums_and_counts': True # Assuming FedLloyds uses sums/counts
        }
    except Exception as e:
        print(f"Error loading base config '{base_config_path}': {e}. Using default Gaussian settings.")
        # Use defaults as above
        base_config = {
            'dataset': 'GaussianMixtureUniform', 'K': 10, 'dim': 100,
            'num_train_clients': 100, 'samples_per_client': 1000,
            'samples_per_mixture_server': 20, 'num_uniform_server': 100,
            'initialization_algorithm': 'FederatedClusterInit',
            'clustering_algorithm': 'FederatedLloyds',
            'minimum_server_point_weight': 5, 'fedlloyds_num_iterations': 1,
            'datapoint_privacy': False, 'outer_product_epsilon': 1, 'weighting_epsilon': 1,
            'center_init_gaussian_epsilon': 1, 'center_init_contributed_components_epsilon': 0.2,
            'fedlloyds_epsilon': 1, 'fedlloyds_epsilon_split': 0.5,
            'outer_product_clipping_bound': 1500, 'weighting_clipping_bound': 1,
            'center_init_clipping_bound': 21, 'center_init_contributed_components_clipping_bound': 10,
            'fedlloyds_clipping_bound': 120, 'fedlloyds_laplace_clipping_bound': 50,
            'overall_target_delta': 1e-6, 'fedlloyds_delta': 1e-6,
            'send_sums_and_counts': True
        }


    # Ensure necessary keys for get_mechanism('fedlloyds') are present
    base_config.setdefault('fedlloyds_num_iterations', 1)
    base_config.setdefault('fedlloyds_cohort_size', base_config.get('num_train_clients', 100)) # Default to all clients
    base_config.setdefault('num_train_clients', 100)
    base_config.setdefault('send_sums_and_counts', True) # Important for mechanism selection in get_mechanism
    base_config.setdefault('datapoint_privacy', False)
    base_config.setdefault('fedlloyds_epsilon', 1)
    base_config.setdefault('fedlloyds_epsilon_split', 0.5)
    # Ensure delta values are present and consistent
    default_delta = base_config.get('overall_target_delta', 1e-6)
    base_config.setdefault('overall_target_delta', default_delta)
    base_config.setdefault('fedlloyds_delta', default_delta) # Ensure fedlloyds_delta exists

    base_config.setdefault('fedlloyds_clipping_bound', 120)
    base_config.setdefault('fedlloyds_laplace_clipping_bound', 50)
    # Add defaults for the mean-sending alternative path in get_mechanism, even if unused
    base_config.setdefault('fedlloyds_contributed_components_epsilon', 0.2)
    base_config.setdefault('fedlloyds_contributed_components_clipping_bound', 10)

    # Config 1: Non-Private (Client Level)
    config_non_private = base_config.copy()
    config_non_private.update({
        'datapoint_privacy': False,
        'outer_product_privacy': False,
        'point_weighting_privacy': False,
        'center_init_privacy': False,
        'fedlloyds_privacy': False, # Explicitly turn off FedLloyds privacy
        'fedlloyds_num_iterations': 1
    })

    with open('configs/gaussian_client_non_private.yaml', 'w') as f:
        yaml.dump(config_non_private, f, sort_keys=False)

    # Config 2: Private (Client Level - using base config parameters)
    config_private = base_config.copy()
    config_private.update({
        'datapoint_privacy': False, # Client Level Attack
        'outer_product_privacy': True, # Keep other mechanisms on if needed by init algo
        'point_weighting_privacy': True,
        'center_init_privacy': True,
        'fedlloyds_privacy': True, # Explicitly turn on FedLloyds privacy (just in case)
        'fedlloyds_num_iterations': 1
    })
    # Ensure fedlloyds_delta exists in the private config too
    config_private.setdefault('fedlloyds_delta', config_private.get('overall_target_delta', 1e-6))

    with open('configs/gaussian_client_private.yaml', 'w') as f:
        yaml.dump(config_private, f, sort_keys=False)

    print("Reconstruction attack config files created.")
    return 'configs/recon_gaussian_non_private.yaml', 'configs/recon_gaussian_private.yaml'


def get_target_data(target_client_id_str, all_train_clients):
    """Fetches the raw data (X matrix) for the target client."""
    user_sampler = get_user_sampler('minimize_reuse', [target_client_id_str])
    target_dataset = FederatedDataset(all_train_clients.make_dataset_fn, user_sampler)
    (user_dataset, _) = next(target_dataset.get_cohort(1))
    # Ensure data is numpy array
    if hasattr(user_dataset.raw_data[0], 'numpy'): # Handle potential TensorFlow tensors
        return user_dataset.raw_data[0].numpy()
    return user_dataset.raw_data[0] # Return X (features)


def run_training_get_centers(config_file, exclude_client_id_str=None, seed=None):
    """
    Runs run.py as a subprocess. Assumes run.py saves centers to 'final_centers.npy'.
    Returns the path to the saved centers file.
    """
    cmd = ['python', '../run.py', '--args_config', config_file]
    if exclude_client_id_str:
        cmd.extend(['--exclude_client_id_str', exclude_client_id_str])
    if seed is not None:
        cmd.extend(['--seed', str(seed)])

    print(f"\nRunning command: {' '.join(cmd)}")
    try:
        # Hide output for cleaner logs unless debugging
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # subprocess.run(cmd, check=True) # Uncomment for debugging run.py output
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")
        # Attempt to capture stderr
        # result = subprocess.run(cmd, capture_output=True, text=True)
        # print("Stderr:", result.stderr)
        return None
    except FileNotFoundError:
        print("Error: 'python' command not found. Make sure Python is in your PATH.")
        return None

    center_file = 'final_centers.npy'
    if not os.path.exists(center_file):
         print(f"Error: Center file '{center_file}' not found after running training.")
         # Maybe run.py failed silently? Check logs if they exist.
         return None
    return center_file

# Helper function to load config into an argparse-like namespace
def load_config_as_namespace(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    # --- Add ALL missing defaults that get_mechanism might expect ---
    # Data args defaults (subset needed by get_mechanism)
    config_dict.setdefault('num_train_clients', 100)

    # Algorithm args defaults (subset needed by get_mechanism)
    config_dict.setdefault('send_sums_and_counts', True) # Crucial for fedlloyds path
    config_dict.setdefault('center_init_send_sums_and_counts', False) # For center_init path

    # Privacy args defaults (crucial for get_mechanism)
    config_dict.setdefault('datapoint_privacy', False)
    default_delta = config_dict.get('overall_target_delta', 1e-6)
    config_dict.setdefault('overall_target_delta', default_delta)

    # --- Defaults for 'fedlloyds' mechanism specifically ---
    config_dict.setdefault('fedlloyds_num_iterations', 1)
    config_dict.setdefault('fedlloyds_cohort_size', config_dict.get('num_train_clients', 100))
    config_dict.setdefault('fedlloyds_epsilon', 1.0) # Default epsilon if missing
    config_dict.setdefault('fedlloyds_epsilon_split', 0.5)
    config_dict.setdefault('fedlloyds_delta', default_delta) # *** ADDED THIS LINE ***
    config_dict.setdefault('fedlloyds_clipping_bound', 120) # L2 bound for sums/means
    config_dict.setdefault('fedlloyds_laplace_clipping_bound', 50) # L1 bound for counts
    # Add mean-sending related defaults even if unused
    config_dict.setdefault('fedlloyds_contributed_components_epsilon', 0.2)
    config_dict.setdefault('fedlloyds_contributed_components_clipping_bound', 10)

    # --- Defaults for other mechanisms (in case base config is minimal) ---
    config_dict.setdefault('outer_product_epsilon', 1.0)
    config_dict.setdefault('outer_product_delta', default_delta)
    config_dict.setdefault('outer_product_clipping_bound', 1500)
    config_dict.setdefault('weighting_epsilon', 1.0)
    config_dict.setdefault('weighting_clipping_bound', 1)
    config_dict.setdefault('center_init_gaussian_epsilon', 1.0)
    config_dict.setdefault('center_init_delta', default_delta)
    config_dict.setdefault('center_init_epsilon_split', 0.5)
    config_dict.setdefault('center_init_clipping_bound', 21)
    config_dict.setdefault('center_init_laplace_clipping_bound', 1)
    config_dict.setdefault('center_init_contributed_components_epsilon', 0.2)
    config_dict.setdefault('center_init_contributed_components_clipping_bound', 10)


    return argparse.Namespace(**config_dict)


def simulate_client_contribution(client_data, global_centers, config_namespace, seed=None):
    """
    Simulates the contribution (sums, counts) a client would send in FedLloyds,
    leveraging the project's existing privacy infrastructure.
    """
    K = global_centers.shape[0]
    dim = global_centers.shape[1]

    # 1. Assign points to centers
    if client_data.shape[0] == 0:
        raw_sums = np.zeros((K, dim), dtype=np.float32)
        raw_counts = np.zeros(K, dtype=np.float32)
    else:
        if client_data.dtype != global_centers.dtype:
             try:
                 client_data = client_data.astype(global_centers.dtype)
             except ValueError:
                 print("Warning: Client data type mismatch, attempting float32 cast.")
                 client_data = client_data.astype(np.float32)
                 global_centers = global_centers.astype(np.float32)

        dist_matrix = pairwise_distances(client_data, global_centers)
        assignments = np.argmin(dist_matrix, axis=1)

        # 2. Calculate true sums and counts
        raw_sums = np.zeros((K, dim), dtype=np.float32)
        raw_counts = np.zeros(K, dtype=np.float32)
        for k in range(K):
            mask = (assignments == k)
            if np.any(mask):
                raw_sums[k] = np.sum(client_data[mask], axis=0)
            raw_counts[k] = np.sum(mask)

    # 3. Create MappedVectorStatistics object
    raw_stats_dict = {}
    if config_namespace.send_sums_and_counts:
        raw_stats_dict['sum_points_per_component'] = raw_sums
        raw_stats_dict['num_points_per_component'] = raw_counts
    else:
        raw_stats_dict['contributed_components'] = (raw_counts > 0).astype(np.float32)
        safe_counts = np.where(raw_counts == 0, 1, raw_counts)
        raw_stats_dict['mean_points_per_component'] = raw_sums / safe_counts[:, np.newaxis]
    raw_stats = MappedVectorStatistics(raw_stats_dict)

    # 4. Instantiate the correct privacy mechanism
    mechanism_name = 'fedlloyds' if config_namespace.fedlloyds_privacy else 'no_privacy'
    try:
        if not hasattr(config_namespace, 'fedlloyds_delta'):
             config_namespace.fedlloyds_delta = config_namespace.overall_target_delta
        mechanism_wrapper = get_mechanism(config_namespace, mechanism_name)
        underlying_mechanism = mechanism_wrapper.underlying_mechanism
    except AttributeError as e:
        print(f"Error: Missing attribute in config_namespace for get_mechanism: {e}")
        raise
    except Exception as e:
        print(f"Error getting mechanism '{mechanism_name}': {e}")
        raise

    # 5. Apply Clipping
    try:
        simulated_seed_clipping = seed if seed is None else seed + 1
        clipped_stats, clip_metrics = underlying_mechanism.constrain_sensitivity(raw_stats, seed=simulated_seed_clipping)
    except Exception as e:
        print(f"Error during constrain_sensitivity: {e}")
        clipped_stats = raw_stats

    # 6. Apply DP Noise
    try:
        simulated_seed_noise = seed if seed is None else seed + 2
        noisy_stats, noise_metrics = underlying_mechanism.add_noise(clipped_stats, cohort_size=1, seed=simulated_seed_noise)
    except Exception as e:
        print(f"Error during add_noise: {e}")
        noisy_stats = clipped_stats

    # 7. Extract noisy results (FIXED PART)
    if config_namespace.send_sums_and_counts:
        # Check if keys exist using 'in' and access using []
        if 'sum_points_per_component' in noisy_stats:
            final_noisy_sums = noisy_stats['sum_points_per_component']
        else:
            print("Warning: 'sum_points_per_component' not found in noisy_stats.")
            final_noisy_sums = np.zeros_like(raw_sums) # Default if key is missing

        if 'num_points_per_component' in noisy_stats:
            final_noisy_counts = noisy_stats['num_points_per_component']
        else:
            print("Warning: 'num_points_per_component' not found in noisy_stats.")
            final_noisy_counts = np.zeros_like(raw_counts) # Default if key is missing
    else:
        # If mean sending is used, the logic here would need to extract noisy means and contribs
        # and then potentially reconstruct sums/counts, which is more complex.
         raise NotImplementedError("Reconstruction attack logic currently only supports 'send_sums_and_counts=True'")


    # Ensure counts are non-negative
    final_noisy_counts = np.maximum(0, final_noisy_counts)

    return final_noisy_sums, final_noisy_counts


def run_reconstruction_once(config_file, target_client_id_str, target_client_data, seed=None):
    """
    Runs one iteration of the reconstruction attack for a specific client.
    Returns the squared L2 error between true and reconstructed mean.
    """
    # 1. Load config into namespace for mechanism simulation
    try:
        config_namespace = load_config_as_namespace(config_file)
        # Ensure we are in sum/count mode for this attack version
        if not config_namespace.send_sums_and_counts:
            print(f"Error: Config file {config_file} has send_sums_and_counts=False. This attack script only supports reconstruction from sums and counts.")
            return np.nan
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        return np.nan # Cannot proceed without config

    # 2. Get Global Centers (Trained without the target client)
    print("Running training without target to get global centers...")
    centers_file = run_training_get_centers(config_file, exclude_client_id_str=target_client_id_str, seed=seed)
    if centers_file is None:
        print("Failed to obtain global centers. Skipping reconstruction for this client.")
        return np.nan # Indicate failure

    try:
        global_centers = np.load(centers_file)
    except Exception as e:
        print(f"Error loading centers file {centers_file}: {e}")
        if os.path.exists(centers_file): os.remove(centers_file)
        return np.nan
    finally:
        if os.path.exists(centers_file): os.remove(centers_file) # Clean up

    # 3. Simulate the target client's contribution (potentially noisy)
    print(f"Simulating contribution for client {target_client_id_str}...")
    try:
        # Use a derived seed for simulation consistency
        sim_seed = seed + 100 if seed is not None else None
        noisy_sums, noisy_counts = simulate_client_contribution(target_client_data, global_centers, config_namespace, seed=sim_seed)
    except Exception as e:
        print(f"Error simulating client contribution: {e}")
        return np.nan # Indicate failure

    # 4. Reconstruct the mean from the noisy contribution
    total_noisy_sum = np.sum(noisy_sums, axis=0)
    total_noisy_count = np.sum(noisy_counts)

    # Use a slightly larger epsilon for check to handle potential floating point inaccuracies
    if total_noisy_count <= 1e-9:
        print("Warning: Total noisy count is near zero. Reconstruction failed.")
        reconstructed_mean = np.full(target_client_data.shape[1], np.nan)
        error = np.nan
    else:
        reconstructed_mean = total_noisy_sum / total_noisy_count

        # 5. Calculate the true mean
        true_mean = np.mean(target_client_data, axis=0)

        # 6. Calculate Squared L2 Error (only if reconstruction succeeded)
        if not np.isnan(reconstructed_mean).any():
             error = np.sum((true_mean - reconstructed_mean)**2)
             print(f"Target {target_client_id_str}: True Mean Norm={np.linalg.norm(true_mean):.4f}, Recon Mean Norm={np.linalg.norm(reconstructed_mean):.4f}, Sq L2 Error={error:.6f}")
        else:
             error = np.nan
             print(f"Target {target_client_id_str}: Reconstruction resulted in NaN values.")

    return error


def main():
    parser = argparse.ArgumentParser(description="Mean Reconstruction Attack Simulation (Gaussian Dataset using PFL Mechanisms)")
    parser.add_argument("--num_attacks", type=int, default=20, help="Number of attack iterations (target clients).")
    parser.add_argument("--base_config", type=str, default="../configs/gaussians_client_privacy.yaml", help="Base config file to derive attack configs and data params.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility.")
    args, unknown = parser.parse_known_args()

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Set global seed to {args.seed}")

    # --- Setup ---
    print("--- 1. Creating Reconstruction Attack config files ---")
    try:
        config_non_private_file, config_private_file = create_recon_configs(args.base_config)
    except Exception as e:
        print(f"Fatal Error: Could not create config files. {e}")
        return

    print("\n--- 2. Loading client list (Gaussian Dataset) ---")
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser = add_data_arguments(temp_parser)
    temp_parser = add_utils_arguments(temp_parser)
    temp_parser = add_algorithms_arguments(temp_parser)

    original_argv = sys.argv.copy()
    # Temporarily modify sys.argv for maybe_inject_arguments_from_config
    # Use the non-private config as it should contain necessary data keys
    sys.argv = [sys.argv[0], '--args_config', config_non_private_file]
    maybe_inject_arguments_from_config()

    try:
        data_args, _ = temp_parser.parse_known_args()
        # Explicitly set data seed if provided
        if args.seed is not None:
            data_args.data_seed = args.seed
        set_data_args(data_args) # Sets num_train_clients etc.
    except Exception as e:
         print(f"Error parsing data arguments from config: {e}")
         print("Attempting to proceed with defaults...")
         data_args = argparse.Namespace(
             dataset='GaussianMixtureUniform', num_train_clients=100,
             exclude_client_id_str=None, data_seed=args.seed if args.seed is not None else 0,
             K=10, dim=100, samples_per_client=1000,
             samples_per_mixture_server=20, num_uniform_server=100, variance=0.5
         )
         try: # Try loading num_train_clients again directly
             with open(config_non_private_file, 'r') as f: cfg = yaml.safe_load(f)
             data_args.num_train_clients = cfg.get('num_train_clients', 100)
         except: data_args.num_train_clients = 100
    finally:
        sys.argv = original_argv # Restore

    # Load all training clients
    data_args.exclude_client_id_str = None
    try:
        all_train_clients, _, _, _ = make_data(data_args)
    except Exception as e:
        print(f"Fatal Error: Could not load data using config '{config_non_private_file}'. Check data paths and parameters. Error: {e}")
        # Print data_args for debugging
        print("Data Args used:", data_args)
        return

    all_client_ids = [str(i) for i in range(data_args.num_train_clients)]
    print(f"Loaded {len(all_client_ids)} total clients.")

    num_attacks = min(args.num_attacks, len(all_client_ids))
    if num_attacks <= 0 and len(all_client_ids) > 0:
        num_attacks = 1 # Ensure at least one attack if possible
    elif len(all_client_ids) == 0:
         print("Error: No clients loaded. Cannot run attacks.")
         return

    print(f"Will run reconstruction attack on {num_attacks} random clients.")

    # --- Run Attacks ---
    results = {
        'non_private': [],
        'private': []
    }
    config_files = {
        'non_private': config_non_private_file,
        'private': config_private_file
    }

    # Sample target clients without replacement
    if num_attacks >= len(all_client_ids):
        target_clients_sample = all_client_ids
    else:
        # Ensure random sampling uses the set seed if provided
        rng = np.random.default_rng(args.seed)
        target_clients_sample = rng.choice(all_client_ids, size=num_attacks, replace=False).tolist()
        # target_clients_sample = random.sample(all_client_ids, num_attacks) # Old way


    for i, target_client_id in enumerate(target_clients_sample):
        print(f"\n--- ATTACK ITERATION {i+1} / {num_attacks} ---")
        print(f"Target Client: {target_client_id}")

        try:
            target_data = get_target_data(target_client_id, all_train_clients)
            if target_data.shape[0] == 0:
                print("Target client has no data. Skipping.")
                continue
            print(f"Target data loaded. Shape: {target_data.shape}")
        except StopIteration:
            print(f"Error: Could not sample target client {target_client_id}. Skipping.")
            continue
        except Exception as e:
            print(f"Error loading data for client {target_client_id}: {e}")
            continue

        for mode in ['non_private', 'private']:
            print(f"--- Running {mode.upper()} scenario ---")
            config_file = config_files[mode]
            # Consistent seed for training step within an iteration/mode
            train_seed = (args.seed + i*10 + (0 if mode == 'non_private' else 1)) if args.seed is not None else None
            # Consistent seed for simulation step within an iteration/mode
            sim_seed = (args.seed + i*10 + (5 if mode == 'non_private' else 6)) if args.seed is not None else None

            try:
                error = run_reconstruction_once(config_file, target_client_id, target_data, seed=train_seed) # Pass train_seed to run_training
                # Note: simulate_client_contribution now uses seed derived inside run_reconstruction_once
                if not np.isnan(error):
                    results[mode].append(error)
                else:
                    print(f"Reconstruction failed for client {target_client_id} in {mode} mode (NaN error).")

            except Exception as e:
                print(f"Attack failed for {mode} on client {target_client_id} due to unexpected error: {e}")
                import traceback
                traceback.print_exc()


    # --- 4. Report Final Results ---
    print("\n--- FINAL RECONSTRUCTION RESULTS (Squared L2 Error) ---")

    for mode in ['non_private', 'private']:
        errors = results[mode]
        count = len(errors)
        if count > 0:
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            median_error = np.median(errors)
            min_error = np.min(errors)
            max_error = np.max(errors)
            print(f"{mode.upper()} Model Results ({count} successful attacks):")
            print(f"  Average Error : {avg_error:.6f}")
            print(f"  StdDev Error  : {std_error:.6f}")
            print(f"  Median Error  : {median_error:.6f}")
            print(f"  Min Error     : {min_error:.6f}")
            print(f"  Max Error     : {max_error:.6f}")
        else:
            print(f"{mode.upper()} Model: No successful attacks to report results.")

    # --- Cleanup ---
    print("\nCleaning up config files...")
    try:
        if os.path.exists(config_non_private_file): os.remove(config_non_private_file)
        if os.path.exists(config_private_file): os.remove(config_private_file)
        # Clean up any lingering centers file
        if os.path.exists('final_centers.npy'):
            os.remove('final_centers.npy')
        print("Cleanup complete.")
    except OSError as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()