import argparse
import subprocess
import os
import numpy as np
import yaml
import random
from sklearn.metrics import pairwise_distances
import sys


from data import make_data, set_data_args, add_data_arguments
from utils import kmeans_cost, add_utils_arguments
from utils.argument_parsing import maybe_inject_arguments_from_config
from pfl.data.sampling import get_user_sampler
from pfl.data.federated_dataset import FederatedDataset
from algorithms import add_algorithms_arguments

# Define the list of state abbreviations used as client IDs in folktables
STATE_LIST = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']


def create_mia_folktables_configs(filter_label=5):
    """Creates two config files for the attack targeting folktables."""

    # Base config values from configs/folktables.yaml, modified for attack
    base_folktables_config = {
        'dataset': 'folktables',
        'filter_label': filter_label,
        'K': 10, # Assuming K=10 for folktables based on other configs
        'samples_per_mixture_server': 10,
        'num_uniform_server': 1000,
        'initialization_algorithm': 'FederatedClusterInitExact', # Use settings from folktables.yaml
        'clustering_algorithm': 'FederatedLloyds',
        'minimum_server_point_weight': 5,
        'fedlloyds_num_iterations': 1, # Keep iterations low for faster attack simulation
         # Use clipping bounds from folktables.yaml
        'outer_product_clipping_bound': 2.65,
        'weighting_clipping_bound': 1,
        'center_init_clipping_bound': 2.65,
        'center_init_laplace_clipping_bound': 1,
        'fedlloyds_clipping_bound': 12039,
        'fedlloyds_laplace_clipping_bound': 4550,
    }

    # Config 1: Non-Private (Client Level)
    config_non_private = base_folktables_config.copy()
    config_non_private.update({
        'datapoint_privacy': False, # Client Level Attack
        'outer_product_privacy': False,
        'point_weighting_privacy': False,
        'center_init_privacy': False,
        'fedlloyds_privacy': False,
    })
    with open('configs/folktables_client_non_private.yaml', 'w') as f:
        yaml.dump(config_non_private, f, sort_keys=False)

    # Config 2: Private (Client Level)
    config_private = base_folktables_config.copy()
    config_private.update({
        'datapoint_privacy': False, # Client Level Attack
        'outer_product_privacy': True,
        'point_weighting_privacy': True,
        'center_init_privacy': True,
        'fedlloyds_privacy': True,
        # Use epsilon values similar to gaussians_client_privacy for a comparable budget
        'outer_product_epsilon': 1,
        'weighting_epsilon': 1,
        'center_init_gaussian_epsilon': 1,
        'center_init_contributed_components_epsilon': 0.2, # Use client-level specific params
        'fedlloyds_epsilon': 1,
        'fedlloyds_epsilon_split': 0.5,
        'center_init_epsilon_split': 0.5, # Add if missing from base
         # Need clipping bounds specific to client-level DP if different
        'center_init_contributed_components_clipping_bound': 10, # Example value, adjust if needed
    })
    # Ensure all required keys for client-level DP are present (referencing gaussians_client_privacy.yaml)
    if 'center_init_contributed_components_clipping_bound' not in config_private:
         config_private['center_init_contributed_components_clipping_bound'] = 10 # Default from gaussians_client_privacy
    if 'fedlloyds_contributed_components_epsilon' not in config_private: # Add if needed for mean sending FedLloyds
         config_private['fedlloyds_contributed_components_epsilon'] = 0.2
    if 'fedlloyds_contributed_components_clipping_bound' not in config_private: # Add if needed
         config_private['fedlloyds_contributed_components_clipping_bound'] = 10


    with open('configs/folktables_client_private.yaml', 'w') as f:
        yaml.dump(config_private, f, sort_keys=False)

    print("MIA config files for folktables created.")


def get_target_data(target_client_id_str, all_train_clients):
    """Fetches the raw data for the target client (state)."""
    # Folktables uses state abbreviations as user_id
    user_sampler = get_user_sampler('minimize_reuse', [target_client_id_str])
    target_dataset = FederatedDataset(all_train_clients.make_dataset_fn, user_sampler)
    (user_dataset, _) = next(target_dataset.get_cohort(1))
    # Folktables data seems to have features at index 0
    return user_dataset.raw_data[0] # Return X (features)


def run_training(config_file, exclude_client_id_str=None):
    """Runs run.py as a subprocess and returns the path to the saved centers."""
    cmd = [
        'python', 'run.py',
        '--args_config', config_file
    ]
    if exclude_client_id_str:
        cmd.extend(['--exclude_client_id_str', exclude_client_id_str])

    print(f"\nRunning command: {' '.join(cmd)}")
    # Hide output for cleaner attack logs
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 'final_centers.npy' # Assumes run.py saves centers here


def calculate_cost(target_data, centers_file):
    """Calculates the K-Means cost for the target data against the centers."""
    if not os.path.exists(centers_file):
        raise FileNotFoundError(f"Centers file not found: {centers_file}")
    try:
        centers = np.load(centers_file)
        # Ensure target_data is numeric (folktables might be boolean)
        if target_data.dtype == bool:
            target_data = target_data.astype(np.float32)
        if centers.dtype != target_data.dtype:
             centers = centers.astype(target_data.dtype)

        # Ensure dimensions match
        if target_data.shape[1] != centers.shape[1]:
            raise ValueError(f"Dimension mismatch: data ({target_data.shape[1]}) vs centers ({centers.shape[1]})")

        return kmeans_cost(target_data, centers)
    except Exception as e:
        print(f"Error calculating cost: {e}")
        print(f"Target data shape: {target_data.shape}, dtype: {target_data.dtype}")
        print(f"Centers file: {centers_file}")
        if 'centers' in locals():
            print(f"Centers shape: {centers.shape}, dtype: {centers.dtype}")
        raise


def run_attack_once(config_file, target_client_id_str, target_client_data):
    """
    Runs one iteration of the MIA.
    Returns True if the attack successfully guessed "IN", False otherwise.
    """
    # 1. Run model WITH target (Model A)
    centers_in_file = run_training(config_file, exclude_client_id_str=None)
    cost_in = calculate_cost(target_client_data, centers_in_file)
    os.remove(centers_in_file) # Clean up intermediate file

    # 2. Run model WITHOUT target (Model B)
    centers_out_file = run_training(config_file, exclude_client_id_str=str(target_client_id_str))
    cost_out = calculate_cost(target_client_data, centers_out_file)
    os.remove(centers_out_file) # Clean up intermediate file

    print(f"Target {target_client_id_str}: Cost(IN) = {cost_in:.4f} | Cost(OUT) = {cost_out:.4f}")

    # 3. Attacker's Guess: Lower cost implies data was likely used in training
    attack_guess_is_in = (cost_in < cost_out)

    # 4. Ground truth is "IN", so success = (guess is "IN")
    return attack_guess_is_in


def main():
    parser = argparse.ArgumentParser(description="Client-Level Membership Inference Attack Simulation for Folktables")
    parser.add_argument("--num_attacks", type=int, default=10, help="Number of attack iterations (Folktables clients are limited). Max 51.")
    parser.add_argument("--filter_label", type=int, default=5, choices=[2, 5, 6], help="Folktables filter label to use for the attack.")
    # We don't need --args_config here as we generate specific ones
    # args = parser.parse_args()
    args, _ = parser.parse_known_args() # Use parse_known_args to ignore extra args injected by maybe_inject

    # --- Setup ---
    print(f"--- 1. Creating MIA config files for Folktables (filter_label={args.filter_label}) ---")
    create_mia_folktables_configs(args.filter_label)

    print("\n--- 2. Loading client list (Folktables States) ---")
    # Use a temporary parser to load data args based on generated config
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser = add_data_arguments(temp_parser)
    temp_parser = add_utils_arguments(temp_parser)
    temp_parser = add_algorithms_arguments(temp_parser)

    # Inject args from one of the generated config files to set dataset='folktables' etc.
    sys.argv.extend(['--args_config', 'configs/folktables_client_non_private.yaml'])
    maybe_inject_arguments_from_config()
    data_args, _ = temp_parser.parse_known_args()
    # data_args = temp_parser.parse_args() # This might fail if maybe_inject adds unexpected args
    set_data_args(data_args) # Sets num_train_clients = 51 etc.

    # Load all training clients (states)
    data_args.exclude_client_id_str = None # Ensure we load all clients initially
    all_train_clients, _, _, _ = make_data(data_args)

    # Folktables client IDs are the state abbreviations
    all_client_ids = STATE_LIST[:data_args.num_train_clients] # Use the actual number determined by set_data_args
    print(f"Loaded {len(all_client_ids)} total clients (states): {all_client_ids}")

    num_attacks = min(args.num_attacks, len(all_client_ids))
    print(f"Running {num_attacks} attack iterations.")

    # --- Run Attacks ---
    attack_results = {
        'non_private': {'success': 0, 'total': 0},
        'private': {'success': 0, 'total': 0}
    }

    config_files = {
        'non_private': 'configs/folktables_client_non_private.yaml',
        'private': 'configs/folktables_client_private.yaml'
    }

    # Sample target clients without replacement since there are few
    target_clients_sample = random.sample(all_client_ids, num_attacks)

    for i, target_client_id in enumerate(target_clients_sample):
        print(f"\n--- ATTACK ITERATION {i+1} / {num_attacks} ---")
        print(f"Target Client: {target_client_id}")

        # Get the target's data
        try:
            target_data = get_target_data(target_client_id, all_train_clients)
            print(f"Target data loaded. Shape: {target_data.shape}")
        except Exception as e:
            print(f"Error loading data for client {target_client_id}: {e}")
            continue # Skip this iteration

        for mode in ['non_private', 'private']:
            print(f"--- Running {mode.upper()} scenario ---")
            config_file = config_files[mode]

            try:
                is_success = run_attack_once(config_file, target_client_id, target_data)

                attack_results[mode]['total'] += 1
                if is_success:
                    attack_results[mode]['success'] += 1
                print(f"Guess: {'IN' if is_success else 'OUT'}. Attack Successful: {is_success}")

            except FileNotFoundError as e:
                 print(f"Attack failed for {mode}: {e}. Skipping.")
            except ValueError as e:
                 print(f"Attack failed for {mode} due to ValueError: {e}. Skipping.")
            except subprocess.CalledProcessError as e:
                print(f"Training run failed for {mode}: {e}. Skipping attack.")
            except Exception as e:
                print(f"An unexpected error occurred during attack for {mode}: {e}. Skipping.")


    # --- 4. Report Final Accuracy ---
    print("\n--- FINAL ATTACK RESULTS (Folktables Client Level) ---")

    total_non_private = attack_results['non_private']['total']
    success_non_private = attack_results['non_private']['success']
    acc_non_private = (success_non_private / total_non_private * 100) if total_non_private > 0 else 0
    print(f"NON-PRIVATE Model Attack Accuracy: {acc_non_private:.2f}% ({success_non_private} / {total_non_private})")

    total_private = attack_results['private']['total']
    success_private = attack_results['private']['success']
    acc_private = (success_private / total_private * 100) if total_private > 0 else 0
    print(f"PRIVATE Model Attack Accuracy:     {acc_private:.2f}% ({success_private} / {total_private})")

    # Clean up generated files
    print("\nCleaning up...")
    try:
        os.remove('configs/folktables_client_non_private.yaml')
        os.remove('configs/folktables_client_private.yaml')
        # Remove final_centers.npy if it exists from a failed run
        if os.path.exists('final_centers.npy'):
            os.remove('final_centers.npy')
        print("Cleanup complete.")
    except OSError as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Need to handle command line args potentially injected by maybe_...
    original_argv = sys.argv.copy()
    try:
        main()
    finally:
        # Restore original sys.argv to prevent issues if run in interactive env
        sys.argv = original_argv