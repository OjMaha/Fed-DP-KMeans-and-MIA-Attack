# calculate_folktables_bounds.py
import argparse
import numpy as np
import yaml
import sys
import math

# Import necessary functions from your project
from data import make_data, set_data_args, add_data_arguments
from utils import add_utils_arguments # Need for data loading setup
from algorithms import add_algorithms_arguments # Need for data loading setup
from utils.argument_parsing import maybe_inject_arguments_from_config
# Import sampler and dataset for type checking if needed, but not strictly required
from pfl.data.sampling import get_user_sampler
from pfl.data.federated_dataset import FederatedDataset


# Define the list of state abbreviations (used to construct client IDs)
STATE_LIST = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

def main():
    parser = argparse.ArgumentParser(description="Calculate Client-Level Clipping Bounds for Folktables")
    parser.add_argument("--base_config", type=str, default="configs/folktables.yaml", help="Path to a base folktables config file (used for dataset parameters).")
    # --filter_label is added by add_data_arguments
    parser = add_data_arguments(parser)
    parser = add_utils_arguments(parser)
    parser = add_algorithms_arguments(parser)

    # --- Data Loading Setup ---
    original_argv = sys.argv.copy()
    temp_args, other_args = parser.parse_known_args()

    if temp_args.filter_label is None:
        print("Error: --filter_label is a required argument for this script.")
        parser.print_help()
        sys.exit(1)

    sys.argv = [sys.argv[0], '--args_config', temp_args.base_config] + other_args
    maybe_inject_arguments_from_config()
    try:
        args = parser.parse_args()
        args.filter_label = temp_args.filter_label # Ensure command-line priority
        args.data_seed = getattr(args, 'seed', 0)
        args.dataset = 'folktables'
        args.exclude_client_id_str = None
        args.datapoint_privacy = False
        set_data_args(args) # This sets args.num_train_clients = 51
    except Exception as e:
         print(f"Error parsing arguments: {e}")
         args = argparse.Namespace(
             dataset='folktables', num_train_clients=51, exclude_client_id_str=None,
             data_seed=0, filter_label=temp_args.filter_label, datapoint_privacy=False
         )
         print("Warning: Using minimal default arguments for data loading.")
    finally:
        sys.argv = original_argv

    print(f"\n--- Analyzing Folktables for filter_label={args.filter_label} ---")

    try:
        all_train_clients, _, _, _ = make_data(args)

        # *** MODIFICATION START: Get client IDs directly ***
        # Since set_data_args ensures args.num_train_clients is correct for folktables,
        # we can construct the list from STATE_LIST.
        if args.dataset == 'folktables':
            client_ids = STATE_LIST[:args.num_train_clients]
        else:
            # Fallback for other datasets (though this script is specific to folktables)
            print("Warning: Script intended for Folktables, but dataset is not 'folktables'. Attempting generic ID generation.")
            client_ids = [str(i) for i in range(args.num_train_clients)]
        # *** MODIFICATION END ***

        num_clients = len(client_ids)
        print(f"Using {num_clients} client IDs: {client_ids[:5]}...{client_ids[-5:]}") # Print first/last few

    except Exception as e:
        print(f"Fatal Error loading data or getting client IDs: {e}")
        print("Please ensure the Folktables data has been preprocessed for this filter label.")
        print("Data Args used:", args)
        return

    if num_clients == 0:
        print("Error: No clients identified. Cannot calculate bounds.")
        return

    max_n = 0
    max_l2_norm_sq_per_point = 0
    num_features = -1

    print("Iterating through clients to find max size and max point norm...")
    processed_clients = 0
    try:
        # We still need to iterate using the make_dataset_fn provided by all_train_clients
        for client_id in client_ids:
            user_sampler_single = get_user_sampler('minimize_reuse', [client_id])
            client_dataset_provider = FederatedDataset(all_train_clients.make_dataset_fn, user_sampler_single)
            (user_dataset, _) = next(client_dataset_provider.get_cohort(1))

            client_data = user_dataset.raw_data[0]
            if hasattr(client_data, 'numpy'): client_data = client_data.numpy()

            current_n = client_data.shape[0]
            if current_n > max_n:
                max_n = current_n
                print(f"  New max client size found: {max_n} (Client: {client_id})")

            if num_features == -1 and current_n > 0:
                num_features = client_data.shape[1]
                print(f"  Detected number of features: {num_features}")

            if current_n > 0:
                l2_norms_sq = np.sum(client_data**2, axis=1)
                client_max_l2_sq = np.max(l2_norms_sq) if l2_norms_sq.size > 0 else 0
                if client_max_l2_sq > max_l2_norm_sq_per_point:
                    max_l2_norm_sq_per_point = client_max_l2_sq

            processed_clients += 1
            if processed_clients % 10 == 0:
                print(f"  Processed {processed_clients}/{num_clients} clients...")

    except StopIteration:
         print("Finished iterating through clients.")
    except Exception as e:
        print(f"An error occurred during client iteration: {e}")
        import traceback
        traceback.print_exc()
        print("Results might be incomplete.")

    # ... (rest of the script remains the same: calculating and printing bounds) ...
    if num_features == -1:
         print("\nError: Could not determine the number of features (all clients might be empty?).")
         return

    max_l2_norm_per_point = math.sqrt(max_l2_norm_sq_per_point)
    estimated_l2_sum_bound = max_n * max_l2_norm_per_point

    print("\n--- Calculated Bounds ---")
    print(f"Maximum data points per client (max_n): {max_n}")
    print(f"Number of features: {num_features}")
    print(f"Maximum L2 norm squared per point: {max_l2_norm_sq_per_point}")
    print(f"Maximum L2 norm per point (sqrt): {max_l2_norm_per_point:.4f}")
    print(f"--- Recommended Client-Level Clipping Bounds ---")
    print(f"fedlloyds_laplace_clipping_bound (L1 Sens. for Counts): {max_n}")
    print(f"fedlloyds_clipping_bound (L2 Sens. for Sums, Est.): {estimated_l2_sum_bound:.4f}")
    print(f"(Calculated as max_n * max_l2_norm_per_point)")
    print("\nNote: These are upper bounds. Tighter bounds might be possible with more analysis.")
    print("Use these values in your client-level private configuration.")

if __name__ == "__main__":
    main()