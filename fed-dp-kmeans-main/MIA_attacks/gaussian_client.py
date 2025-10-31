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
    """Creates two config files for the attack."""

    os.makedirs("MIA_attacks/configs", exist_ok=True)
    
    # Config 1: Non-Private
    config_non_private = {
        'dataset': 'GaussianMixtureUniform',
        'K': 10,
        'dim': 100,
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
    with open('MIA_attacks/configs/gaussian_client_non_private.yaml', 'w') as f:
        yaml.dump(config_non_private, f)

    # Config 2: Private
    config_private = {
        'dataset': 'GaussianMixtureUniform',
        'K': 10,
        'dim': 100,
        'num_train_clients': 100,
        'samples_per_client': 1000,
        'samples_per_mixture_server': 20,
        'num_uniform_server': 100,
        'datapoint_privacy': False,  # --- PRIVATE ---
        'outer_product_epsilon': 0.25,
        'weighting_epsilon': 0.25,
        'center_init_gaussian_epsilon': 0.25,
        'center_init_epsilon_split': 0.5,
        'fedlloyds_epsilon': 0.25,   # --- Total Epsilon approx 1.0 ---

        'outer_product_clipping_bound': 1500,
        'weighting_clipping_bound': 1,
        'center_init_clipping_bound': 21,
        'center_init_contributed_components_clipping_bound': 10,
        'fedlloyds_clipping_bound': 120,
        'fedlloyds_laplace_clipping_bound': 50,
        
        'initialization_algorithm': 'FederatedClusterInitExact',
        'clustering_algorithm': 'FederatedLloyds',
        'minimum_server_point_weight': 5,
        'fedlloyds_num_iterations': 1
    }
    with open('MIA_attacks/configs/gaussian_client_private.yaml', 'w') as f:
        yaml.dump(config_private, f)
    
    print("MIA config files created.")


def get_target_data(target_client_id_str, all_train_clients):
    """Fetches the raw data for the target client."""
    # This is a bit of a hack to get the dataset for one user.
    # We create a sampler that *only* knows about our target user.
    user_sampler = get_user_sampler('minimize_reuse', [target_client_id_str])
    
    # Create a new FederatedDataset pointing to the same data, but with the new sampler
    target_dataset = FederatedDataset(all_train_clients.make_dataset_fn, user_sampler)
    
    # Sample the user
    (user_dataset, _) = next(target_dataset.get_cohort(1))
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
    
    # We hide the output to keep the attack log clean
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # run.py saves its output here
    return 'final_centers.npy'


def calculate_cost(target_data, centers_file):
    """Calculates the K-Means cost for the target data against the centers."""
    centers = np.load(centers_file)
    return kmeans_cost(target_data, centers)


def run_attack_once(config_file, target_client_id_str, target_client_data):
    """
    Runs one iteration of the MIA.
    Returns True if the attack successfully guessed "IN", False otherwise.
    """
    
    # 1. Run model WITH target (Model A)
    centers_in_file = run_training(config_file, exclude_client_id_str=None)
    cost_in = calculate_cost(target_client_data, centers_in_file)
    
    # 2. Run model WITHOUT target (Model B)
    centers_out_file = run_training(config_file, exclude_client_id_str=str(target_client_id_str))
    cost_out = calculate_cost(target_client_data, centers_out_file)
    
    print(f"Target {target_client_id_str}: Cost(IN) = {cost_in:.4f} | Cost(OUT) = {cost_out:.4f}")
    
    # 3. Attacker's Guess
    attack_guess_is_in = (cost_in < cost_out)
    
    # 4. We know the ground truth is "IN", so the attack is successful if it guessed "IN"
    return attack_guess_is_in


def main():
    parser = argparse.ArgumentParser(description="Membership Inference Attack Simulation")
    parser.add_argument("--num_attacks", type=int, default=50, help="Number of attack iterations.")
    parser.add_argument("--args_config", type=str, default="configs/gaussians_data_privacy.yaml", help="Base config to get data params.")
    args, _ = parser.parse_known_args()

    # --- Setup ---
    print("--- 1. Creating MIA config files ---")
    create_mia_configs()
    
    print("\n--- 2. Loading client list ---")
    
    # We need to load the full dataset once to know all client IDs.
    temp_parser = argparse.ArgumentParser()
    temp_parser = add_data_arguments(temp_parser)
    temp_parser = add_utils_arguments(temp_parser)
    temp_parser = add_algorithms_arguments(temp_parser)
    
    # Inject args from the base config file
    maybe_inject_arguments_from_config() 
    # data_args = temp_parser.parse_args()
    data_args, _ = temp_parser.parse_known_args()
    set_data_args(data_args)
    
    # Load all training clients (we set exclude to None)
    data_args.exclude_client_id_str = None
    all_train_clients, _, _, _ = make_data(data_args)
    
    # Get the *full* list of client IDs
    all_client_ids = [str(i) for i in range(data_args.num_train_clients)]
    print(f"Loaded {len(all_client_ids)} total clients.")
    
    
    # --- Run Attacks ---
    attack_results = {
        'non_private': {'success': 0, 'total': 0},
        'private': {'success': 0, 'total': 0}
    }
    
    config_files = {
        'non_private': 'MIA_attacks/configs/gaussian_client_non_private.yaml',
        'private': 'MIA_attacks/configs/gaussian_client_private.yaml'
    }

    for i in range(args.num_attacks):
        print(f"\n--- ATTACK ITERATION {i+1} / {args.num_attacks} ---")
        
        # Pick a random target client
        target_client_id = random.choice(all_client_ids)
        print(f"Target Client: {target_client_id}")
        
        # Get the target's data
        target_data = get_target_data(target_client_id, all_train_clients)
        
        for mode in ['non_private', 'private']:
            print(f"--- Running {mode.upper()} scenario ---")
            config_file = config_files[mode]
            
            try:
                is_success = run_attack_once(config_file, target_client_id, target_data)
                
                attack_results[mode]['total'] += 1
                if is_success:
                    attack_results[mode]['success'] += 1
                print(f"Guess: {'IN' if is_success else 'OUT'}. Attack Successful: {is_success}")
                
            except Exception as e:
                print(f"Attack failed for {mode} due to error: {e}")
                print("This can sometimes happen if clustering fails with noisy data.")

    # --- 4. Report Final Accuracy ---
    print("\n--- FINAL ATTACK RESULTS ---")
    
    acc_non_private = (attack_results['non_private']['success'] / attack_results['non_private']['total']) * 100
    print(f"NON-PRIVATE Model Attack Accuracy: {acc_non_private:.2f}% ({attack_results['non_private']['success']} / {attack_results['non_private']['total']})")
    
    acc_private = (attack_results['private']['success'] / attack_results['private']['total']) * 100
    print(f"PRIVATE Model Attack Accuracy:     {acc_private:.2f}% ({attack_results['private']['success']} / {attack_results['private']['total']})")

    # Clean up
    os.remove('MIA_attacks/configs/gaussian_client_non_private.yaml')
    os.remove('MIA_attacks/configs/gaussian_client_private.yaml')
    os.remove('final_centers.npy')

if __name__ == "__main__":
    main()