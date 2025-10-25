import argparse

def add_data_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds all data-related arguments to the main parser in `run.py`.
    This centralizes the configuration for which dataset to use and how to generate it.
    """
    # Add arguments for specific real-world datasets.
    parser = add_stackoverflow_arguments(parser)
    parser = add_folktables_arguments(parser)

    # --- General Data Arguments ---
    parser.add_argument("--data_seed", type=int, default=0, help="Seed for data generation and splitting.")
    parser.add_argument("--dataset", type=str,
                        choices=["GaussianMixtureUniform", "stackoverflow", "folktables"],
                        default="GaussianMixtureUniform", help="Which dataset to use for the experiment.")
    parser.add_argument("--num_train_clients", type=int, default=100, help="Total number of clients in the training set.")
    parser.add_argument("--num_val_clients", type=int, default=100, help="Total number of clients in the validation set.")

    # --- Synthetic Data (GaussianMixture) Arguments ---
    # Paper Connection: These parameters control the generation of the synthetic data
    # described in Section 5.1 and 5.2, and Appendix G.1.
    parser.add_argument("--samples_per_client", type=int, default=1000, help="Number of data points on each client.")
    parser.add_argument("--dim", type=int, default=100, help="Dimension of the synthetic data.")
    parser.add_argument("--variance", type=float, default=0.5, help="Variance of the Gaussian components.")
    
    # --- Server Data Arguments ---
    # Paper Connection: These arguments control the composition of the server's dataset `Q`.
    # The paper experiments with a mix of in-distribution and out-of-distribution (OOD) data.
    parser.add_argument("--samples_per_mixture_server", type=int, default=20,
                        help="Number of in-distribution samples per cluster for the server dataset.")
    parser.add_argument("--num_uniform_server", type=int, default=100,
                        help="Number of out-of-distribution (uniform) samples for the server dataset.")

    return parser


def add_stackoverflow_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds arguments for the Stack Overflow dataset.
    Paper Connection: Corresponds to the Stack Overflow experiments in Section 5.2 and Appendix G.1.
    """
    parser.add_argument("--topics_list", type=str,
                        choices=['fb-hb',  'gith-pdf',  'ml-math',  'plt-cook'],
                        default='gith-pdf',
                        help="Which pair of topic tags to use for creating the clustering task.")
    return parser


def add_folktables_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds arguments for the Folktables (US Census) dataset.
    Paper Connection: Corresponds to the US Census experiments in Section 5.1 and Appendix G.1.
    """
    parser.add_argument("--filter_label", type=int,
                        choices=[2, 5, 6],
                        default=5,
                        help="Which employment category to use as the client data for the clustering task.")
    return parser


def set_data_args(args):
    """
    This function dynamically sets some arguments (like the number of clients) based on the
    chosen dataset, as the real-world datasets have a fixed number of users. This is a
    convenience function to avoid specifying incorrect numbers of clients for real datasets.
    """
    if args.dataset == 'stackoverflow':
        # Lookup table for the number of users per topic pair.
        num_clients_lookup = {
            'plt-cook': (2720, 328),
            'gith-pdf': (9237, 1157),
            'fb-hb': (23266, 2736),
            'ml-math': (10394, 1265)
        }
        args.num_train_clients, args.num_val_clients = num_clients_lookup[args.topics_list]
    elif args.dataset == 'folktables':
        # The Folktables dataset is partitioned by US state, so there are 51 clients.
        args.num_train_clients = 51
        args.num_val_clients = 51