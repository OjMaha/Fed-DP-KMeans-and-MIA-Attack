#!/usr/bin/env python3
"""
Data-level Membership Inference Attack for Folktables (fixed client ID handling)

Usage (from repo root):
    python MIA_attacks/folktables_datapoint.py --num_attacks 5 --args_config configs/folktables.yaml
"""

import os
import sys

# Make sure repo root is importable (so `from data import ...` works)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import subprocess
import numpy as np
import yaml
import random
import time
from sklearn.metrics import pairwise_distances

from data import make_data, set_data_args, add_data_arguments
from utils import kmeans_cost, add_utils_arguments
from utils.argument_parsing import maybe_inject_arguments_from_config
from pfl.data.sampling import get_user_sampler
from pfl.data.federated_dataset import FederatedDataset
from algorithms import add_algorithms_arguments

# Try importing STATE_LIST used by the repo's folktables loader
try:
    # path may be data/folktables.py or similar; try likely locations
    from data.folktables import STATE_LIST as FOLK_STATE_LIST
except Exception:
    # fallback -- the repo's list, kept in same order
    FOLK_STATE_LIST = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                       'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                       'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                       'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                       'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']


# ---------------------------
# Config creation
# ---------------------------
def create_mia_configs():
    """Create simple private / non-private configs for Folktables MIA."""
    base = {
        'dataset': 'folktables',
        'filter_label': 5,
        'samples_per_mixture_server': 10,
        'num_uniform_server': 1000,
        'initialization_algorithm': 'FederatedClusterInitExact',
        'clustering_algorithm': 'FederatedLloyds',
        'minimum_server_point_weight': 5,
        'fedlloyds_num_iterations': 1
    }

    non_private = {
        **base,
        'datapoint_privacy': False,
        'outer_product_privacy': False,
        'point_weighting_privacy': False,
        'center_init_privacy': False,
        'fedlloyds_privacy': False
    }

    private = {
        **base,
        'datapoint_privacy': True,
        'outer_product_epsilon': 1,
        'weighting_epsilon': 1,
        'center_init_gaussian_epsilon': 1,
        'center_init_epsilon_split': 0.5,
        'fedlloyds_epsilon': 1,
        'fedlloyds_epsilon_split': 0.5,
        'outer_product_clipping_bound': 2.65,
        'weighting_clipping_bound': 1,
        'center_init_clipping_bound': 2.65,
        'center_init_laplace_clipping_bound': 1,
        'fedlloyds_clipping_bound': 2.65,
        'fedlloyds_laplace_clipping_bound': 1
    }

    os.makedirs("configs", exist_ok=True)
    with open("configs/folktables_datapoint_non_private.yaml", "w") as f:
        yaml.dump(non_private, f)
    with open("configs/folktables_datapoint_private.yaml", "w") as f:
        yaml.dump(private, f)

    print("✅ Created configs for Folktables data-level MIA.")


# ---------------------------
# Data helpers
# ---------------------------
def ensure_numeric_array(X):
    """Convert DataFrame/torch/tensor -> numeric 2D numpy array float32."""
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None and isinstance(X, pd.DataFrame):
        X = X.select_dtypes(include=[np.number]).values
    # if it's a torch tensor or has .numpy()
    if hasattr(X, "numpy"):
        X = X.numpy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float32)


def get_target_client_data(client_id, all_train_clients):
    """
    Return processed/encoded features for a client identified by `client_id`.
    This tries `processed_data` -> `data` -> `raw_data` and converts to numeric array.
    """
    user_sampler = get_user_sampler('minimize_reuse', [client_id])
    target_dataset = FederatedDataset(all_train_clients.make_dataset_fn, user_sampler)
    (user_dataset, _) = next(target_dataset.get_cohort(1))

    # prefer processed_data if available (this matches training pipeline encoding)
    if hasattr(user_dataset, "processed_data"):
        X_raw = user_dataset.processed_data[0]
    elif hasattr(user_dataset, "data"):
        X_raw = user_dataset.data[0]
    else:
        X_raw = user_dataset.raw_data[0]

    X = ensure_numeric_array(X_raw)
    return X


def get_target_datapoint(client_id, sample_index, all_train_clients):
    """Return a single datapoint as (1, d) array for given client and sample index."""
    X = get_target_client_data(client_id, all_train_clients)
    if X.shape[0] == 0:
        raise IndexError(f"Client {client_id} has zero samples")
    if not (0 <= sample_index < X.shape[0]):
        raise IndexError(f"sample_index {sample_index} out of range for client {client_id} (0..{X.shape[0]-1})")
    return X[sample_index:sample_index + 1]


# ---------------------------
# run / cost functions
# ---------------------------
def run_training(config_file, exclude_datapoint=None, seed=None):
    """
    Run run.py with the given args_config.
    exclude_datapoint: (client_id, idx) or None
    Returns path to saved centers ('final_centers.npy').
    """
    cmd = [sys.executable, "run.py", "--args_config", config_file]
    if exclude_datapoint is not None:
        client_id, idx = exclude_datapoint
        cmd += ["--exclude_datapoint", f"{client_id}:{idx}"]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    print(f"Running: {' '.join(cmd)}")
    # run without suppressing output so we see errors if they occur
    subprocess.run(cmd, check=True)
    return "final_centers.npy"


def calculate_cost(target_data, centers_file):
    centers = np.load(centers_file)
    # debug print to ensure dims match
    print(f"DEBUG: target shape {target_data.shape}, centers shape {centers.shape}, target dtype {target_data.dtype}, centers dtype {centers.dtype}")
    if centers.shape[1] != target_data.shape[1]:
        raise ValueError(f"Dimension mismatch: centers {centers.shape} vs target {target_data.shape}")
    return kmeans_cost(target_data, centers)


def run_attack_once(config_file, client_id, sample_idx, target_datapoint, seed=None):
    # Train with datapoint present
    centers_in = run_training(config_file, exclude_datapoint=None, seed=seed)
    try:
        cost_in = calculate_cost(target_datapoint, centers_in)
    finally:
        if os.path.exists(centers_in):
            os.remove(centers_in)

    # Train without that datapoint
    centers_out = run_training(config_file, exclude_datapoint=(client_id, sample_idx), seed=seed)
    try:
        cost_out = calculate_cost(target_datapoint, centers_out)
    finally:
        if os.path.exists(centers_out):
            os.remove(centers_out)

    print(f"{client_id}:{sample_idx} -> Cost(IN)={cost_in:.6f}, Cost(OUT)={cost_out:.6f}")
    return cost_in < cost_out


# ---------------------------
# main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Folktables Data-Level MIA")
    parser.add_argument("--num_attacks", type=int, default=10)
    parser.add_argument("--args_config", type=str, default="configs/folktables.yaml",
                        help="Base folktables config used by make_data()")
    parser.add_argument("--seed", type=int, default=None)
    args, _ = parser.parse_known_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # create small private/non-private configs used by each run
    create_mia_configs()

    # build data args parser & inject config defaults
    temp_parser = argparse.ArgumentParser(add_help=False)
    add_data_arguments(temp_parser)
    add_utils_arguments(temp_parser)
    add_algorithms_arguments(temp_parser)

    # ensure maybe_inject will read --args_config from command line
    # If user passed --args_config on command line, it is already in sys.argv
    maybe_inject_arguments_from_config()
    data_args, _ = temp_parser.parse_known_args()
    set_data_args(data_args)

    data_args.exclude_client_id_str = None
    all_train_clients, _, _, _ = make_data(data_args)

    # prefer the repo's STATE_LIST ordering for Folktables clients
    try:
        # realistic Folktables clients are states (e.g., 'CA','NY',...)
        num_clients = data_args.num_train_clients
        client_ids = FOLK_STATE_LIST[:num_clients]
    except Exception:
        # fallback to whatever the dataset exposes
        try:
            client_ids = list(all_train_clients.client_ids)
        except Exception:
            client_ids = [str(i) for i in range(data_args.num_train_clients)]

    print(f"Final Server Dataset Size: (if printed by make_data above)")
    print(f"Loaded {len(client_ids)} clients: {client_ids[:10]}{'...' if len(client_ids)>10 else ''}")

    configs = {
        "non_private": "configs/folktables_datapoint_non_private.yaml",
        "private": "configs/folktables_datapoint_private.yaml"
    }

    results = {k: {"succ": 0, "tot": 0} for k in configs}

    mode_dataset_cache = {}
    for mode, cfg in configs.items():
        # Build data_args for this specific config so make_data() matches run.py's config
        saved_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0], '--args_config', cfg]
            maybe_inject_arguments_from_config()
            temp_parser_mode = argparse.ArgumentParser(add_help=False)
            temp_parser_mode = add_data_arguments(temp_parser_mode)
            temp_parser_mode = add_utils_arguments(temp_parser_mode)
            temp_parser_mode = add_algorithms_arguments(temp_parser_mode)
            data_args_mode, _ = temp_parser_mode.parse_known_args()
            set_data_args(data_args_mode)
        finally:
            sys.argv = saved_argv

        # Ensure we load all clients for this config (no exclude)
        data_args_mode.exclude_client_id_str = None
        try:
            all_train_clients_mode, _, _, _ = make_data(data_args_mode)
            mode_dataset_cache[mode] = (data_args_mode, all_train_clients_mode)
        except Exception as e:
            print(f"[ERROR] make_data() failed for config {cfg} (mode {mode}): {e}")
            mode_dataset_cache[mode] = None

    # Decide which clients to attack
    attacks_to_run = min(args.num_attacks, len(client_ids))
    sampled_clients = random.sample(client_ids, attacks_to_run)

    for i, client_id in enumerate(sampled_clients, start=1):
        print(f"\n--- ATTACK {i}/{attacks_to_run} on client {client_id} ---")

        # Try to load client data from the "base" dataset (this is optional/for logging)
        try:
            Xc_base = get_target_client_data(client_id, all_train_clients)
            print(f"(base dataset) Client {client_id} samples: {Xc_base.shape[0]}, dim: {Xc_base.shape[1]}")
        except Exception as e:
            # Not fatal — continue, but warn
            print(f"(base dataset) Warning: failed loading client {client_id}: {e}")

        # For each mode, pick a sample index using the dataset built from that mode's config
        for mode, cfg in configs.items():
            print(f"-> Mode: {mode}")

            cache_entry = mode_dataset_cache.get(mode)
            if cache_entry is None:
                print(f"Skipping {mode} because dataset build failed for its config.")
                continue

            data_args_mode, all_train_clients_mode = cache_entry

            # Get the client's processed data as it will appear to run.py under this config
            try:
                Xc_mode = get_target_client_data(client_id, all_train_clients_mode)
            except Exception as e:
                print(f"Failed loading client {client_id} for mode {mode}: {e}. Skipping this mode.")
                continue

            if Xc_mode.shape[0] == 0:
                print(f"Client {client_id} has zero samples under mode {mode}; skipping.")
                continue

            # Choose index from the mode-specific dataset (guaranteed valid for subprocess)
            sample_idx_mode = random.randrange(Xc_mode.shape[0])
            target_dp_mode = Xc_mode[sample_idx_mode:sample_idx_mode + 1]
            print(f"Selected sample {sample_idx_mode} for client {client_id} under mode {mode} (size {Xc_mode.shape[0]})")

            # Run the per-mode attack (this will call run.py using the same config)
            try:
                succ = run_attack_once(cfg, client_id, sample_idx_mode, target_dp_mode, seed=args.seed)
                results[mode]["tot"] += 1
                results[mode]["succ"] += int(succ)
                print(f"Result: {'IN' if succ else 'OUT'}")
            except subprocess.CalledProcessError as e:
                print(f"Training failed (subprocess) for {mode}: {e}")
            except ValueError as e:
                print(f"Value error (likely dim mismatch) for {mode}: {e}")
            except Exception as e:
                print(f"Unexpected error for {mode}: {e}")

    print("\n--- FINAL RESULTS ---")
    for mode, r in results.items():
        tot = r["tot"]
        succ = r["succ"]
        acc = 100.0 * succ / tot if tot > 0 else 0.0
        print(f"{mode.upper()} Accuracy: {acc:.2f}% ({succ}/{tot})")

    # cleanup
    for f in ("configs/folktables_datapoint_non_private.yaml",
              "configs/folktables_datapoint_private.yaml",
              "final_centers.npy"):
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


if __name__ == "__main__":
    main()
