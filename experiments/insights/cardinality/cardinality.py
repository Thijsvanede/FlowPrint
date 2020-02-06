from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import classification_report
from sklearn.metrics import adjusted_rand_score
from sklearn.exceptions import UndefinedMetricWarning
import argparse
import matplotlib.pyplot as plt
import numpy as np
import warnings

from os import path
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../../flowprint')))
from cluster import Cluster
from fingerprints import Fingerprints
from loader import Loader
from preprocessor import Preprocessor
from split import Splitter


def degree(fingerprints, y):
    """Compute the degree of each application within the flow.

        Parameters
        ----------
        fingerprints : np.array of shape=(n_samples,)
            Array of fingerprints corresponding to given labels.

        y : np.array of shape=(n_samples,)
            Array of labels corresponding to given fingerprints.

        Returns
        -------
        result : dict
            Dictionary of label -> # fingerprints.
        """
    # Initialise dictionaries
    result  = dict()
    mapping = dict()

    # Loop over all fingerprint, label combinations
    for fp, label in zip(fingerprints, y):
        # Add fingerprint to label
        mapping[label] = mapping.get(label, set()) | set([fp])

    # Count # fingerprints per label
    for label, fps in mapping.items():
        # Get number of fingerprints per label
        result[label] = len(fps)

    # Return result
    return result

def degree_hist(fingerprints, y):
    """Compute histogram of degrees in dataset.

        Parameters
        ----------
        fingerprints : np.array of shape=(n_samples,)
            Array of fingerprints corresponding to given labels.

        y : np.array of shape=(n_samples,)
            Array of labels corresponding to given fingerprints.

        Returns
        -------
        result : dict
            Dictionary of #fingerprints -> count.
        """
    # Initialise result
    result = dict()

    # Get degree
    degree = degree(fingerprints, y)

    # Create histogram of degrees
    for label, count in degree.items():
        # Add result to histogram
        result[count] = result.get(count, 0) + 1

    # Return result
    return result


if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint evaluator.')
    parser.add_argument('-l', '--load', nargs='+',
                        help='load preprocessed data from given file.')
    parser.add_argument('-b', '--batch', type=float, default=300,
                        help='Batch size in seconds (default=300).')
    parser.add_argument('-w', '--window', type=float, default=30,
                        help='Window size in seconds (default=30).')
    parser.add_argument('-c', '--correlation', type=float, default=0.1,
                        help='Cross-correlation threshold (default=0.1).')
    parser.add_argument('-s', '--similarity', type=float, default=0.9,
                        help='Similarity threshold (default=0.9).')
    parser.add_argument('-k', '--kfolds', type=int, default=1,
                        help='Number of k-folds to perform (default=1).')
    parser.add_argument('-m', '--max-apps', type=int,
                        help='Maximum number of applications per k-fold (optional).')
    parser.add_argument('-p', '--min-flows', type=int, default=1,
                        help='Minimum number of flows required per application '
                             '(default=1).')

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                          Preprocessing step                          #
    ########################################################################

    # Load data if necessary
    if args.load:
        if any('andrubis' in x.lower() for x in args.load):
            loader = Loader('../../data/Andrubis/appdict.p')
        else:
            loader = Loader()

        # Initialise data
        X = list()
        y = list()

        # Load each file individually
        for load in args.load:
            # Load data
            X_, y_ = loader.load_file(load)
            # Create specific labels for cross_market
            y_ = np.asarray([load.split('/')[-1][:-2]]*X_.shape[0])

            # Add input file to data
            if X_.shape[0] >= args.min_flows:
                X.append(X_)
                y.append(y_)

        # Concatenate results
        X = np.concatenate(X)
        y = np.concatenate(y)

    # Print number of applications in dataset
    print("Number of applications: {}".format(np.unique(y).shape[0]))
    print("Number of flows       : {}".format(X.shape[0]))

    ########################################################################
    #                            Run Evaluation                            #
    ########################################################################

    # # Create fingerprint
    # # Create clustering object
    # cluster = Cluster()
    # # Create Fingerprinter
    # flowprint = Fingerprints(cluster, args.batch, args.window, args.correlation, args.similarity)
    # # Get fingerprints
    # fps = flowprint.fit_predict_fingerprints(X)
    #
    # # Compute cardinality
    # cardinality = degree(fps, y)
    #
    # from pprint import pprint
    # pprint(Counter(cardinality.values()))

    cluster = Cluster()
    # Create Fingerprinter
    flowprint = Fingerprints(cluster, args.batch, args.window, args.correlation, args.similarity)
    # Predict labels
    fps = flowprint.fit_predict_labels(X, y)

    print(degree(y, fps))
