# Some of the code in this file is adapted from:
#
# pfl-research:
# Copyright 2024, Apple Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import argparse
import lzma
import os
from tqdm import tqdm
import urllib.parse
import urllib.request

# =================================================================================================
# Paper Connection: This script is the first step in preparing the Stack Overflow dataset
# as described in Appendix G.1. It handles the download of the large, compressed SQLite
# database containing the raw Stack Overflow posts.
# =================================================================================================


def fetch_lzma_file(origin: str, filename: str):
    """
    Fetches a LZMA compressed file from a URL and decompresses it on the fly while downloading.
    This is memory-efficient for very large files.
    """
    chunk_size = 2**20  # Process in 1MB chunks.
    decompressor = lzma.LZMADecompressor()
    with urllib.request.urlopen(origin) as in_stream, open(filename, 'wb') as out_stream:
        length = in_stream.headers.get('content-length')
        total_size = int(length) if length is not None else None
        download_chunk = in_stream.read(chunk_size)
        with tqdm(total=total_size, desc=f'Downloading {urllib.parse.urlparse(origin).path.rsplit("/", maxsplit=1)[-1]}') as progbar:
            while download_chunk:
                progbar.update(len(download_chunk))
                out_stream.write(decompressor.decompress(download_chunk))
                download_chunk = in_stream.read(chunk_size)


def main():
    """
    Main function to download the Stack Overflow database if it doesn't already exist.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='embedded_data',
                        help='Directory to save the downloaded SQLite file.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    database_filepath = os.path.join(args.output_dir, "stackoverflow.sqlite")
    if not os.path.exists(database_filepath):
        print(f'Downloading StackOverflow data to {args.output_dir}')
        # This is the public URL for the TFF Stack Overflow dataset.
        database_origin = ("https://storage.googleapis.com/tff-datasets-public/"
                           "stackoverflow.sqlite.lzma")
        fetch_lzma_file(origin=database_origin, filename=database_filepath)


if __name__ == '__main__':
    main()