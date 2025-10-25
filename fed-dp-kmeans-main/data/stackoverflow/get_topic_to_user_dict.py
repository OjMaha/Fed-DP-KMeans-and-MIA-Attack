import argparse
import pickle
import h5py
import os

# =================================================================================================
# Paper Connection: This script is the third step in preparing the Stack Overflow dataset.
# After the posts are embedded, this script iterates through the HDF5 files to build an
# index (a dictionary) that maps each topic tag (e.g., 'python', 'github') to the set of
# user IDs who have posted a question with that tag. This index is essential for creating
# the specific clustering tasks described in Appendix G.1.
# =================================================================================================


def main():
    parser = argparse.ArgumentParser()
    # This script is also designed to be parallelized.
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--num_jobs_per_split', type=int, default=1)
    args = parser.parse_args()

    split = ['train', 'val', 'test'][args.job_id // args.num_jobs_per_split]
    filename = f'embedded_stackoverflow_{split}_{args.job_id % args.num_jobs_per_split}.hdf5'
    data_dir = 'embedded_data'
    
    # This dictionary will store the mapping: { 'topic_tag': {user_id_1, user_id_2, ...} }
    mapping = dict()

    # Open the HDF5 file containing the embedded data for this job's chunk.
    with h5py.File(os.path.join(data_dir, filename), 'r') as f:
        d = f[split]
        user_ids = list(d.keys())
        # Iterate through every user in this chunk.
        for user_id in user_ids:
            # Get all the tags for all posts by this user.
            tag_array = d[user_id]['tags'][()].astype(str)
            # Iterate through each post's tags.
            for tag_str in tag_array:
                # A single post can have multiple tags, separated by '|'.
                for tag in tag_str.split('|'):
                    # Add the current user ID to the set for this tag.
                    try:
                        mapping[tag].add(user_id)
                    except KeyError:
                        # If this is the first time we've seen this tag, create a new set.
                        mapping[tag] = {user_id}

    # Save the resulting mapping dictionary to a pickle file for the next step.
    output_filename = f'{data_dir}/tag_to_user_{split}_{args.job_id % args.num_jobs_per_split}.pkl'
    with open(output_filename, 'wb') as pickle_f:
        pickle.dump(mapping, pickle_f)


if __name__ == '__main__':
    main()