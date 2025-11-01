import argparse
import os
import pathlib
import h5py
import numpy as np

from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import get_user_sampler

from utils import set_seed

from .folktables import make_folktables_datasets


def make_data(args: argparse.Namespace):
    set_seed(args.data_seed)
    dataset = args.dataset

    target_client_to_exclude = args.exclude_client_id_str

    if dataset == 'GaussianMixtureUniform':
        K, d = args.K, args.dim
        means = make_uniform_gaussian_means(K, d)
        covs = np.stack([args.variance * np.eye(d) for _ in range(args.K)])
        mixture_weights = np.ones(K) / K

        def num_samples_fn():
            return args.samples_per_client

        all_clients = []
        central_data = []
        for num_clients in [args.num_train_clients, args.num_val_clients]:
            client_datasets = []
            all_x, all_y = [], []

            for i in range(num_clients):
                
                n = num_samples_fn()
                x, y = generate_gaussian_mixture_data(n, mixture_weights, means, covs)

                if str(i) == target_client_to_exclude:
                    print(f"--- MIA ATTACK: Excluding client {i} ---")
                    continue # <-- NOW ONLY SKIPS APPENDING
                
                client_datasets.append(Dataset((x, y), user_id=str(i)))
                all_x.append(x)
                all_y.append(y)

            all_x = np.vstack(all_x)
            all_y = np.hstack(all_y)
            central_data.append(Dataset((all_x, all_y)))

            user_id_list = [d.user_id for d in client_datasets] # Get filtered list of user IDs
            user_id_to_dataset = {d.user_id: d for d in client_datasets}

            def make_dataset_fn(user_id, datasets=user_id_to_dataset):
                # Use dictionary lookup
                return datasets[user_id]

            user_sampler = get_user_sampler('minimize_reuse', user_id_list)

            all_clients.append(FederatedDataset(make_dataset_fn, user_sampler))

        samples_per_mixture = args.samples_per_mixture_server * np.ones(K, dtype=int)
        server_x, server_y = [], []
        for k in range(K):
            mixture_weights = np.zeros(K)
            mixture_weights[k] = 1
            x, y = generate_gaussian_mixture_data(samples_per_mixture[k], mixture_weights, means, covs)
            assert y.sum() == len(y) * k
            server_x.append(x)
            server_y.append(y)

        num_uniform_server = args.num_uniform_server
        server_x.append(np.random.uniform(size=(num_uniform_server, d)))
        server_y.append(K * np.ones(num_uniform_server))
        server_dataset = Dataset((np.vstack(server_x), np.hstack(server_y)))

    elif dataset == 'stackoverflow':
        path_to_data_file = pathlib.Path(__file__).parent.absolute()
        path_to_data = os.path.join(path_to_data_file, 'stackoverflow', 'topic_extracted_data', args.topics_list)
        if not os.path.exists(path_to_data):
            raise NotADirectoryError('Please download stackoverflow dataset following the provided instuctions.')

        topic_abreviations = {'machine-learning': 'ml',
                              'math': 'math',
                              'facebook': 'fb',
                              'hibernate': 'hb',
                              'github': 'gith',
                              'pdf': 'pdf',
                              'plot': 'plt',
                              'cookies': 'cook'
                              }
        abreviations_lookup = {abr: topic for topic, abr in topic_abreviations.items()}
        topics = [abreviations_lookup[abr] for abr in args.topics_list.split('-')]

        all_clients = []
        central_data = []
        for split in ['train', 'val']:
            num_clients = args.num_train_clients if split == 'train' else args.num_val_clients
            client_datasets = []
            filename = f'{split}_users.hdf5'
            user_id_to_dataset = {} # Use dict here too
            with h5py.File(os.path.join(path_to_data, filename), 'r') as f:
                d = f[split]
                user_ids = list(d.keys()) # Get actual user IDs
                print(f'Number of {split} users: {len(user_ids)}')
                for user_id in user_ids:

                    # --- MIA ATTACK exclusion check ---
                    if split == 'train' and str(user_id) == target_client_to_exclude:
                        print(f"--- MIA ATTACK: Excluding client {user_id} ---")
                        continue # Skip this client
                    # ------------------------------------

                    user_data = d[user_id]
                    x = user_data['embeddings'][()]
                    y = user_data['labels'][()]
                    dataset_obj = Dataset((x, y), user_id=str(user_id)) # Ensure user_id is str
                    client_datasets.append(dataset_obj)
                    user_id_to_dataset[str(user_id)] = dataset_obj # Store in dict

                all_x = np.vstack([d.raw_data[0] for d in client_datasets])
                all_y = np.hstack([d.raw_data[1] for d in client_datasets])
                centralized_dataset = Dataset((all_x, all_y))
                central_data.append(centralized_dataset)

                # Define make_dataset_fn using the dictionary
                def make_dataset_fn(user_id, datasets=user_id_to_dataset):
                     # Use dictionary lookup
                    return datasets[user_id]

                # Use the actual user IDs (which might have exclusions)
                user_id_list = list(user_id_to_dataset.keys())
                user_sampler = get_user_sampler('minimize_reuse', user_id_list)
                all_clients.append(FederatedDataset(make_dataset_fn, user_sampler))


        path_to_server_data = os.path.join(path_to_data, 'server_data.hdf5')
        if not os.path.exists(path_to_server_data):

            train_x, train_y = central_data[0].raw_data
            sampled_idxs = np.random.choice(np.arange(len(train_y)), size=args.samples_per_mixture_server * len(topics))
            server_x = train_x[sampled_idxs]
            server_y = train_y[sampled_idxs]

            uniform_server_x = []
            uniform_server_y = []
            path_to_uniform_file = os.path.join('data', 'stackoverflow', 'embedded_data', 'embedded_stackoverflow_train_0.hdf5')
            with h5py.File(path_to_uniform_file, 'r') as f:
                done = False
                user_ids = list(f['train'].keys())
                for user_id in user_ids:
                    user_data = f['train'][user_id]
                    tags = user_data['tags'][()].astype(str)
                    embeddings = user_data['embeddings'][()]
                    for embedding, tag in zip(embeddings, tags):
                        if not set(tag.split('|')).intersection(set(topics)):
                            uniform_server_x.append(embedding)
                            uniform_server_y.append(len(topics))

                        if len(uniform_server_y) == args.num_uniform_server:
                            done = True
                            break

                    if done:
                        break

            uniform_server_x = np.vstack(uniform_server_x)
            uniform_server_y = np.hstack(uniform_server_y)

            server_x = np.vstack([server_x, uniform_server_x])
            server_y = np.hstack([server_y, uniform_server_y])

            with h5py.File(path_to_server_data, 'w') as server_f:
                server_f.create_dataset('server_x', data=server_x)
                server_f.create_dataset('server_y', data=server_y)
        else:
            with h5py.File(path_to_server_data, 'r') as server_f:
                server_x = server_f['server_x'][()]
                server_y = server_f['server_y'][()]
        server_dataset = Dataset((server_x, server_y))

    elif dataset == 'folktables':
        path_to_data_file = pathlib.Path(__file__).parent.absolute()
        path_to_data = os.path.join(path_to_data_file, 'folktables', 'cow_extracted_data', f'fl-{args.filter_label}')
        
        if not os.path.exists(path_to_data):
            make_folktables_datasets(args)

        all_clients, central_data = [], []
        for split in ['train', 'val']:
            client_datasets = []
            # Use a dictionary to map user_id (state) to dataset
            user_id_to_dataset = {}
            with h5py.File(os.path.join(path_to_data, f'{split}_users.hdf5'), 'r') as f:
                user_ids = list(f.keys()) # These are the state strings 'AL', 'AK', etc.
                for user_id in user_ids:

                    # --- MIA ATTACK exclusion check ---
                    if split == 'train' and str(user_id) == target_client_to_exclude:
                        print(f"--- MIA ATTACK: Excluding client {user_id} ---")
                        continue # Skip this client
                    # ------------------------------------

                    features = f[user_id]['features'][()].astype(int)
                    labels = f[user_id]['labels'][()]
                    # Create Dataset object and store in the dictionary
                    dataset_obj = Dataset((features, labels), user_id=str(user_id)) # Ensure user_id is string
                    client_datasets.append(dataset_obj) # Keep the list for central_data calculation
                    user_id_to_dataset[str(user_id)] = dataset_obj # Store with string key

            # Define make_dataset_fn using the dictionary
            def make_dataset_fn(user_id, datasets=user_id_to_dataset):
                # Use dictionary lookup with the string user_id
                return datasets[user_id]

            # Get the list of user IDs actually included (handles exclusion)
            user_id_list = list(user_id_to_dataset.keys())
            user_sampler = get_user_sampler('minimize_reuse', user_id_list)
            all_clients.append(FederatedDataset(make_dataset_fn, user_sampler))

            # Central data calculation remains the same (uses the list)
            if client_datasets: # Check if list is not empty after exclusion
                 all_x = np.vstack([d.raw_data[0] for d in client_datasets])
                 all_y = np.hstack([d.raw_data[1] for d in client_datasets])
                 central_data.append(Dataset((all_x, all_y)))
            else: # Handle case where the only client was excluded
                 # Need to decide what central_data should be - maybe empty or raise error?
                 # For now, append an empty dataset placeholder
                 central_data.append(Dataset((np.array([]), np.array([]))))


        # --- Server data loading/creation
        with h5py.File(os.path.join(path_to_data, 'server_data.hdf5'), 'r') as f:
            # Check if central_data[0] exists and is not empty
            if central_data and len(central_data[0].raw_data[0]) > 0:
                 train_x, train_y = central_data[0].raw_data
                 # Ensure enough data points for sampling
                 num_samples_available = len(train_x)
                 num_to_sample = min(num_samples_available, 2 * args.samples_per_mixture_server)
                 if num_to_sample > 0:
                     in_dist_idxs = np.random.choice(range(num_samples_available), size=num_to_sample, replace=False)
                     in_dist_x = train_x[in_dist_idxs]
                     in_dist_y = train_y[in_dist_idxs]
                 else:
                      in_dist_x, in_dist_y = np.array([]).reshape(0, train_x.shape[1]), np.array([]) # Empty arrays with correct feature dim
            else:
                 # Handle case where central_data is empty or excluded client was the only one
                 in_dist_x, in_dist_y = np.array([]), np.array([]) # Define empty arrays


            total_num_datapoints = 0
            for state in f.keys():
                total_num_datapoints += len(f[state]['labels'])

            uniform_features = []
            uniform_labels = []
            feature_dim = None # To store the feature dimension

            # Ensure there's data to sample from
            if total_num_datapoints > 0:
                 for state in f.keys():
                     features, labels = f[state]['features'][()].astype(int), f[state]['labels'][()].reshape(-1)
                     if feature_dim is None and len(features) > 0:
                         feature_dim = features.shape[1] # Get feature dim from first non-empty client

                     if len(features) > 0:
                         num_state_samples = int(args.num_uniform_server * (len(features) / total_num_datapoints)) + 1
                         num_to_sample = min(len(features), num_state_samples) # Don't sample more than available
                         idxs = np.random.choice(range(len(features)),
                                                 size=num_to_sample,
                                                 replace=False
                                                 )
                         uniform_features.append(features[idxs])
                         uniform_labels.append(labels[idxs])

            if uniform_features: # Check if any uniform features were collected
                 uniform_features = np.vstack(uniform_features)
                 uniform_labels = np.hstack(uniform_labels)
                 # Sample exactly num_uniform_server points if available
                 num_uniform_available = len(uniform_features)
                 num_uniform_to_sample = min(num_uniform_available, args.num_uniform_server)
                 if num_uniform_to_sample > 0:
                     uniform_idxs = np.random.choice(range(num_uniform_available), size=num_uniform_to_sample, replace=False)
                     final_uniform_features = uniform_features[uniform_idxs]
                     final_uniform_labels = uniform_labels[uniform_idxs]
                 else:
                      # If feature_dim was found, create empty array with correct shape
                      feat_shape = (0, feature_dim) if feature_dim is not None else (0,0)
                      final_uniform_features, final_uniform_labels = np.array([]).reshape(feat_shape), np.array([])
            else:
                 # If no uniform features found and feature_dim known, create empty arrays
                 feat_shape = (0, feature_dim) if feature_dim is not None else (0,0) # Attempt to get dim if possible
                 final_uniform_features, final_uniform_labels = np.array([]).reshape(feat_shape), np.array([])


            # Combine in-distribution and uniform server data
            # Ensure arrays are not empty before vstack/hstack
            server_x_parts = []
            server_y_parts = []
            if len(in_dist_x) > 0:
                server_x_parts.append(in_dist_x)
                server_y_parts.append(in_dist_y)
            if len(final_uniform_features) > 0:
                 server_x_parts.append(final_uniform_features)
                 server_y_parts.append(final_uniform_labels)

            if server_x_parts: # If there's any server data
                 server_x = np.vstack(server_x_parts)
                 server_y = np.hstack(server_y_parts)
            else: # If both parts were empty
                 # Create empty server dataset, attempting to get feature dim if known
                 feat_shape = (0, feature_dim) if feature_dim is not None else (0,0)
                 server_x, server_y = np.array([]).reshape(feat_shape), np.array([])


            server_dataset = Dataset((server_x, server_y))


    else:
        raise ValueError("Dataset not recognized.")
    
    if server_dataset and server_dataset.raw_data and len(server_dataset.raw_data) > 0:
         print(f"Final Server Dataset Size: {len(server_dataset.raw_data[0])} samples")
         if len(server_dataset.raw_data[0]) < args.K:
              print(f"WARNING: Server dataset has fewer samples ({len(server_dataset.raw_data[0])}) than K ({args.K}). This will likely cause KMeans errors.")
    else:
         print("WARNING: Server dataset appears to be empty.")

    return all_clients[0], all_clients[1], server_dataset, central_data


def make_uniform_gaussian_means(k, d, x_min=0, x_max=1):
    """
    Generates k uniformly random vectors of dimension d in [x_min, x_max] hypercube.

    :param k: number of vectors.
    :param d: dimension.
    :param x_min: hypercube lower bound.
    :param x_max: hypercube upper bound.
    :return: np.array of shape (k, d)
    """
    return np.random.uniform(low=x_min, high=x_max, size=(k, d)).astype(np.float32)


def generate_gaussian_mixture_data(num_samples: int, mixture_weights: np.array, means: np.array, covs: np.array):
    """
    Generates samples from a mixture of Gaussians.

    :param num_samples: number of samples to generate
    :param mixture_weights: array of length k, where k is the number of gaussians
    :param means: array of shape (k, d), where d is the dimension
    :param covs:: array of shape (k, d, d), each (d, d) array is the covariance matrix of a gaussian
    :return: (x, y) arrays of shape (num_samples, d) and (num_samples). Samples and labels of each sample.
    """
    d = len(means[0])
    cumulative_mixture_sum = np.cumsum(mixture_weights)
    z = np.random.uniform(size=(num_samples, 1))
    components_of_all_samples = (z <= cumulative_mixture_sum).argmax(axis=1)
    unique_components, per_component_counts = np.unique(components_of_all_samples, return_counts=True)

    x = np.zeros((num_samples, d), dtype=np.float32)
    y = np.zeros(num_samples, dtype=int)
    for k, count in zip(unique_components, per_component_counts):
        # Handle case where covariance matrix might become non-positive semi-definite due to numerical issues
        try:
             points = np.random.multivariate_normal(mean=means[k], cov=covs[k], size=count).astype(np.float32)
        except np.linalg.LinAlgError:
             print(f"Warning: Covariance matrix for component {k} not positive semi-definite. Adding small identity matrix.")
             jitter = np.eye(d) * 1e-6 # Add small jitter
             points = np.random.multivariate_normal(mean=means[k], cov=covs[k] + jitter, size=count).astype(np.float32)

        x[components_of_all_samples == k] = points
        y[components_of_all_samples == k] = k * np.ones(count)


    return x, y