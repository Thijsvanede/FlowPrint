import argparse
import numpy as np
import time

from collections import Counter
from pprint import pprint

import os
from os import path
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../flowprint/')))
from cluster import Cluster
from fingerprint import Fingerprint
from fingerprints import Fingerprints
from loader import Loader
from preprocessor import Preprocessor
from split import Splitter

def chunk(flows, total, seed=42):
    """Get permutation of flows to sum up to total."""
    # Create random state
    rs = np.random.RandomState(seed)
    return rs.choice(flows, total)


if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint IP-TCP analyser.')
    parser.add_argument('-l', '--load', nargs='+',
                        help='load preprocessed data from given file.')
    parser.add_argument('-p', '--min-flows', type=int, default=1,
                        help='Minimum number of flows required per application '
                         '(default=1).')

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                          Preprocessing step                          #
    ########################################################################

    # Initialise preprocessor
    preprocessor = Preprocessor(verbose=True)

    # Load data if necessary
    if args.load:
        if any('andrubis' in x.lower() for x in args.load):
            loader = Loader('../../data/Andrubis/appdict.p', version=True)
        else:
            loader = Loader()

        X, y = loader.load_files(args.load, args.min_flows)

    else:
        print("Please provide input")
        exit()

    ########################################################################
    #                            Run Evaluation                            #
    ########################################################################

    # Make sure the timestamps are in the same range
    for x in X:
        x.timestamps = [t-x.timestamps[0] for t in x.timestamps]

    # Create cluster
    cluster   = Cluster()
    # Create fingerprint generator
    generator = Fingerprints(cluster, batch=300, window=30, correlation=0.1, similarity=0.9)

    # Loop over sizes 100kb, 1Mb, 10Mb, 100Mb, 1Gb, 5Gb, 10Gb
    for size in [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        for random_seed in range(10):
            # Get slices as input
            f = chunk(X, int(size), random_seed)
            # Start time
            time_start = time.time()

            # Perform experiment
            result = generator._fit_single_batch_(f)

            # End time
            time_end   = time.time()
            # Print total time
            print("FP: size={:12}({:12}), rs={:2}, time={} seconds".format(size, f.shape[0], random_seed, time_end-time_start))
