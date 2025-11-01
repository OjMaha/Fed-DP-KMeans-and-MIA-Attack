# -------------------------
# DATA-LEVEL MIA ATTACK SCRIPT
# -------------------------
import argparse
import subprocess
import os
import numpy as np
import yaml
import random
from sklearn.metrics import pairwise_distances

# Import make_data to get client data, and kmeans_cost
from data import make_data, set_data_args, add_data_arguments
from utils import kmeans_cost, add_utils_arguments
from utils.argument_parsing import maybe_inject_arguments_from_config
from pfl.data.sampling import get_user_sampler
from pfl.data.federated_dataset import FederatedDataset
from algorithms import add_algorithms_arguments


def create_mia_configs():
    """Creates two config files for the attack (same as your original)."""

    os.makedirs("MIA_attacks/configs", exist_ok=True)

    # Config 1: Non-Private
    config_non_private = {
        'dataset': 'GaussianMixtureUniform',
        # 'K': 10,
        # 'dim': 100,
        'num_train_clients': 100,
        'samples_per_client': 1000,
        'samples_per_mixture_server': 20,
        'num_uniform_server': 100,
        'datapoint_privacy': False,  # --- NON-PRIVATE ---
        'outer_product_privacy': False,
        'point_weighting_privacy': False,
        'center_init_privacy': False,
        'fedlloyds_privacy': False,  # --- NON-PRIVATE ---
        'fedlloyds_num_iterations': 1
    }
    with open('MIA_attacks/configs/gaussian_datapoint_non_private.yaml', 'w') as f:
        yaml.dump(config_non_private, f)

    # Config 2: Private
    config_private = {
        'dataset': 'GaussianMixtureUniform',
        # 'K': 10,
        # 'dim': 100,
        'num_train_clients': 100,
        'samples_per_client': 1000,
        'samples_per_mixture_server': 20,
        'num_uniform_server': 100,
        'datapoint_privacy': True,  # --- PRIVATE ---
        'outer_product_epsilon': 0.1,
        'weighting_epsilon': 0.1,
        'center_init_gaussian_epsilon': 0.1,
        'center_init_epsilon_split': 0.5,
        'fedlloyds_epsilon': 0.1,   # --- Total Epsilon approx 1.0 ---

        'outer_product_clipping_bound': 11,
        'weighting_clipping_bound': 1,
        'center_init_clipping_bound': 11,
        'fedlloyds_clipping_bound': 11,
        'fedlloyds_laplace_clipping_bound': 1,
        
        'initialization_algorithm': 'FederatedClusterInitExact',
        'clustering_algorithm': 'FederatedLloyds',
        'minimum_server_point_weight': 5,
        'fedlloyds_num_iterations': 1
    }
    with open('MIA_attacks/configs/gaussian_datapoint_private.yaml', 'w') as f:
        yaml.dump(config_private, f)
    
    print("MIA config files created.")


def get_target_client_data(target_client_id_str, all_train_clients):
    """Return the full local data matrix (X) for a single client (as you had)."""
    user_sampler = get_user_sampler('minimize_reuse', [target_client_id_str])
    target_dataset = FederatedDataset(all_train_clients.make_dataset_fn, user_sampler)
    (user_dataset, _) = next(target_dataset.get_cohort(1))
    return user_dataset.raw_data[0]  # X matrix (n_local_samples, dim)


def get_target_datapoint(target_client_id_str, sample_index, all_train_clients):
    """
    Return a single datapoint (1 x dim) from a client's local dataset.
    Raises IndexError if sample_index out-of-range.
    """
    X = get_target_client_data(target_client_id_str, all_train_clients)
    if sample_index < 0 or sample_index >= X.shape[0]:
        raise IndexError(f"sample_index {sample_index} out of range (0..{X.shape[0]-1}) for client {target_client_id_str}")
    return X[sample_index:sample_index+1]  # keep as 2D array for kmeans_cost


def run_training(config_file, exclude_client_id_str=None, exclude_datapoint=None, seed=None, output_centers=None):
    """
    Run run.py and optionally pass an exclude_client_id_str OR exclude_datapoint tuple (client_id, sample_idx).
    exclude_datapoint: (client_id_str, sample_index)
    seed: optional int to pass --seed (deterministic runs)
    output_centers: optional output path to pass to run.py if you've added that CLI (not required here)
    """
    cmd = ['python', 'run.py', '--args_config', config_file]
    if exclude_client_id_str:
        cmd.extend(['--exclude_client_id_str', exclude_client_id_str])
    if exclude_datapoint:
        client_id_str, sample_idx = exclude_datapoint
        cmd.extend(['--exclude_datapoint', f'{client_id_str}:{sample_idx}'])
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    if output_centers:
        cmd.extend(['--output_centers', output_centers])
    print(f"\nRunning command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # run.py writes final_centers.npy by default
    return output_centers if output_centers else 'final_centers.npy'


def calculate_cost(target_data, centers_file):
    """Calculates the K-Means cost for the target data against the centers."""
    centers = np.load(centers_file)
    return kmeans_cost(target_data, centers)


def run_attack_once_datalevel(config_file, target_client_id_str, sample_index, target_datapoint, seed=None):
    """
    Run the data-level attack for one datapoint (client_id, sample_index).
    Returns True if attacker guessed IN (i.e., cost_in < cost_out).
    """
    # Derive two different seeds for the IN and OUT runs
    seed_in = seed      # e.g., seed 0
    seed_out = seed + 1   # e.g., seed 1 (DIFFERENT SEED)

    # 1) WITH datapoint present
    centers_in_file = run_training(config_file, exclude_client_id_str=None, exclude_datapoint=None, seed=seed_in)
    cost_in = calculate_cost(target_datapoint, centers_in_file)

    # 2) WITHOUT datapoint (exclude that datapoint)
    centers_out_file = run_training(config_file, exclude_client_id_str=None,
                                exclude_datapoint=(target_client_id_str, sample_index), seed=seed_out)
    cost_out = calculate_cost(target_datapoint, centers_out_file)

    print(f"Target {target_client_id_str}:{sample_index} -> Cost(IN)={cost_in:.6f}, Cost(OUT)={cost_out:.6f}")

    attack_guess_is_in = (cost_in < cost_out)
    return attack_guess_is_in


def main():
    parser = argparse.ArgumentParser(description="Data-level Membership Inference Attack (white-box)")
    parser.add_argument("--num_attacks", type=int, default=50, help="Number of attack iterations.")
    parser.add_argument("--args_config", type=str, default="configs/gaussians_data_privacy.yaml", help="Base config to get data params.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed to pass to runs for determinism.")
    args, _ = parser.parse_known_args()

    # Setup
    print("--- Creating MIA config files ---")
    create_mia_configs()

    print("\n--- Loading client list ---")
    temp_parser = argparse.ArgumentParser()
    temp_parser = add_data_arguments(temp_parser)
    temp_parser = add_utils_arguments(temp_parser)
    temp_parser = add_algorithms_arguments(temp_parser)

    maybe_inject_arguments_from_config()
    data_args, _ = temp_parser.parse_known_args()
    set_data_args(data_args)

    # Load all training clients (exclude none)
    data_args.exclude_client_id_str = None
    all_train_clients, _, _, _ = make_data(data_args)

    all_client_ids = [str(i) for i in range(data_args.num_train_clients)]
    print(f"Loaded {len(all_client_ids)} total clients.")

    # Attack bookkeeping
    attack_results = {
        'non_private': {'success': 0, 'total': 0},
        'private': {'success': 0, 'total': 0}
    }
    config_files = {
        'non_private': 'MIA_attacks/configs/gaussian_datapoint_non_private.yaml',
        'private': 'MIA_attacks/configs/gaussian_datapoint_private.yaml'
    }

    random.seed(26)

    for i in range(args.num_attacks):
        print(f"\n--- ATTACK ITER {i+1}/{args.num_attacks} ---")
        # pick a random client and a random datapoint index from that client
        target_client_id = random.choice(all_client_ids)
        # get number of samples for that client
        X_client = get_target_client_data(target_client_id, all_train_clients)
        n_local = X_client.shape[0]
        sample_index = random.randrange(n_local)
        print(f"Target datapoint -> client {target_client_id}, index {sample_index} (client has {n_local} samples)")

        target_datapoint = get_target_datapoint(target_client_id, sample_index, all_train_clients)

        for mode in ['non_private', 'private']:
            print(f"--- Mode: {mode.upper()} ---")
            config_file = config_files[mode]
            try:
                # Create a unique seed for this iteration, even if args.seed is None
                iter_seed = i*10 + (args.seed if args.seed is not None else 0)
                is_success = run_attack_once_datalevel(config_file, target_client_id, sample_index, target_datapoint, seed=iter_seed)
                attack_results[mode]['total'] += 1
                if is_success:
                    attack_results[mode]['success'] += 1
                print(f"Guess: {'IN' if is_success else 'OUT'} (successful={is_success})")
            except Exception as e:
                print(f"Attack failed for {mode} due to error: {e}")

    # Report
    print("\n--- FINAL ATTACK RESULTS ---")
    for mode in ['non_private', 'private']:
        total = attack_results[mode]['total']
        succ = attack_results[mode]['success']
        acc = (succ / total * 100) if total > 0 else 0.0
        print(f"{mode.upper()} Attack Accuracy: {acc:.2f}% ({succ}/{total})")

    # Cleanup
    try:
        os.remove('MIA_attacks/configs/gaussian_datapoint_non_private.yaml')
        os.remove('MIA_attacks/configs/gaussian_datapoint_private.yaml')
    except Exception:
        pass
    try:
        os.remove('final_centers.npy')
    except Exception:
        pass


if __name__ == "__main__":
    main()
