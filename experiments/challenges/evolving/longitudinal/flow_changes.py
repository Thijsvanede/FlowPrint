from collections import Counter
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import classification_report
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import argparse
import copy
import ipaddress
import json
import numpy as np
import pickle
import warnings

from os import path
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../../../flowprint')))
from cluster import Cluster
from fingerprint import Fingerprint
from fingerprints import Fingerprints
from loader import Loader
from split import Splitter

sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../../recognition')))
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../../anomaly')))
from recognition import Recogniser
from anomaly import AnomalyDetector


################################################################################
#                               FlowTransformer                                #
################################################################################

class FlowTransformer(object):

    def transform_certificates(self, X, fraction=0.1, random_state=42):
        """Change a given fraction of TLS certificates in dataset.

            Parameters
            ----------
            X : array-like of shape=(n_flows,)
                Flows for which to change certificates.

            fraction : float, default=0.1
                Fraction of certificates to change in dataset.

            random_state : int, default=42
                Randomness for reproducability
            """
        # Get X as numpy array
        X = np.asarray([copy.deepcopy(x) for x in X])

        # Extract certificates from X
        certificates = set().union(*[x.certificates for x in X])

        # Create mapping
        mapping = dict()
        # Loop over all certificates
        for certificate in certificates:
            # Compute randomness based on certificate + random state
            rs = (certificate + random_state) & 0xFFFFFFFF

            # Get mapping based on value of certificate
            if np.random.RandomState(rs).rand(1) >= fraction:
                # Set original as mapping
                mapping[certificate] = certificate
            else:
                mapping[certificate] = np.random.RandomState(rs).randint(10e9)

        # Modify certificates in X
        for x in X:
            x.certificates = set([mapping.get(c, c) for c in x.certificates])

        # Return modified certificates
        return X


    def transform_ips(self, X, fraction=0.1, random_state=42):
        """Change a given fraction of TLS certificates in dataset.

            Parameters
            ----------
            X : array-like of shape=(n_flows,)
                Flows for which to change certificates.

            fraction : float, default=0.1
                Fraction of certificates to change in dataset.

            random_state : int, default=42
                Randomness for reproducability
            """
        # Initialise random state
        rs = np.random.RandomState(random_state)

        # Get X as numpy array
        X = np.asarray([copy.deepcopy(x) for x in X])

        # Extract certificates from X
        ips = set().union(*[x.ips for x in X])

        # Create mapping
        mapping = dict()
        # Loop over all certificates
        for ip in ips:
            # Compute randomness based on certificate + random state
            rs = (int(ipaddress.ip_address(ip[0])) + random_state) & 0xFFFFFFFF
            # Get mapping based on value of ip
            if np.random.RandomState(rs).rand(1) >= fraction:
                # Set original as mapping
                mapping[ip] = ip
            else:
                ip2 = ipaddress.ip_address(np.random.RandomState(rs).bytes(4))
                mapping[ip] = (str(ip2), ip[1])

        # Modify certificates in X
        for x in X:
            x.ips = set([mapping.get(ip, ip) for ip in x.ips])

        # Return modified certificates
        return X


################################################################################
#                            Recognition experiment                            #
################################################################################

class RecogniserTransformed(Recogniser):

    def __init__(self, fraction_certificates=0, fraction_ips=0):
        """RecogniserTransformed for measuring recognition performance on
            transformed flows.

            Parameters
            ----------
            fraction_certificates : float, default=0
                Fraction of certificates to change.

            fraction_ips : float, default = 0
                Fraction of ips to change.
            """
        # Call super constructor
        super().__init__()

        # Create transformer object
        self.transformer = FlowTransformer()
        # Add transformation fractions
        self.fraction_certificates = fraction_certificates
        self.fraction_ips          = fraction_ips


    def run(self, X, y, batch, window, correlation, similarity, k_folds=1,
            max_labels=None, y_malicious=None, verbose=False):
        """See experiments/cross_market/recognition.py -> Recogniser.run()."""
        # Initialise result
        result   = dict()
        result_m = dict()

        # Iterate over all k-folds
        for i, (X_, y_) in enumerate(self.splitter.k_fold_label(X, y, k_folds,
                                                max_labels=max_labels)):
            # Split in train-test data
            X_train, X_test, y_train, y_test = train_test_split(X_, y_,
                                                                test_size=0.33,
                                                                random_state=i)

            # NOTE : This is the modified part of the code !!!
            # Transform certificates
            X_test = self.transformer.transform_certificates(
                        X_test, self.fraction_certificates, i)

            # Transform IPs
            X_test = self.transformer.transform_ips(
                        X_test, self.fraction_ips, i)
            # NOTE : This is the modified part of the code !!!

            # Perform evaluation
            try:
                partial_result, m = self.evaluate(X_train, y_train, X_test, y_test,
                                               batch=batch, window=window,
                                               correlation=correlation,
                                               similarity=similarity,
                                               y_malicious=y_malicious,
                                               verbose=verbose)
            except:
                print("Execution failed")
                continue

            # Show result if verbose
            if verbose:
                print("----- K-fold: {} -----".format(i))
                self.show(partial_result)
                if m:
                    print("----- Malice: {} -----".format(i))
                    self.show(m)

            # Add partial result
            result   = {k: v+result.get(k, 0) for k, v in partial_result.items()}
            if m:
                result_m = {k: v+result_m.get(k, 0) for k, v in m.items()}

        # Average result
        result   = {k: v/k_folds for k, v in result.items()}
        result_m = {k: v/k_folds for k, v in result_m.items()}

        # Return average result
        return result, result_m


################################################################################
#                            Unseen app experiment                             #
################################################################################
class AnomalyDetectorTransformed(AnomalyDetector):

    # TODO modify class such that we introduce X% changes upon evaluation

    def __init__(self, fraction_certificates=0, fraction_ips=0):
        """RecogniserTransformed for measuring unseen app detection performance
            on transformed flows.

            Parameters
            ----------
            fraction_certificates : float, default=0
                Fraction of certificates to change.

            fraction_ips : float, default = 0
                Fraction of ips to change.
            """
        # Call super constructor
        super().__init__()

        # Create transformer object
        self.transformer = FlowTransformer()
        # Add transformation fractions
        self.fraction_certificates = fraction_certificates
        self.fraction_ips          = fraction_ips

    def run(self, X, y, batch, window, correlation, similarity, k_folds=1,
            max_labels=None, verbose=False):
        """See experiments/cross_market/recognition.py -> Recogniser.run()."""
        # Initialise result
        result = dict()

        # Iterate over all k-folds
        for i, (X_, y_) in enumerate(self.splitter.k_fold_label(X, y, k_folds,
                                                max_labels=max_labels)):

            # Split in train, known, unknown data
            X_train, y_train,\
            X_known, y_known,\
            X_unknown, y_unknown = self.splitter.split_normal(X_, y_,
                                                              size_known=0.8,
                                                              size_train=0.8)

            # NOTE : This is the modified part of the code !!!
            # Transform certificates
            X_known = self.transformer.transform_certificates(
                X_known, self.fraction_certificates, i)
            X_unknown = self.transformer.transform_certificates(
                X_unknown, self.fraction_certificates, i)

            # Transform IPs
            X_known = self.transformer.transform_ips(
                X_known, self.fraction_ips, i)
            X_unknown = self.transformer.transform_ips(
                X_unknown, self.fraction_ips, i)
            # NOTE : This is the modified part of the code !!!

            # Perform evaluation
            partial_result = self.evaluate(X_train, y_train,
                                           X_known, y_known,
                                           X_unknown, y_unknown,
                                           batch=batch, window=window,
                                           correlation=correlation,
                                           similarity=similarity,
                                           verbose=verbose)
            # Show result if verbose
            if verbose:
                print("----- K-fold: {} -----".format(i))
                self.show(partial_result)

            # Add partial result
            result = {k: v+result.get(k, 0) for k, v in partial_result.items()}

        # Return average result
        return result


if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint recogniser.')
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
                        help='Maximum number of applications '
                             'per k-fold (optional).')
    parser.add_argument('-p', '--min-flows', type=int, default=1,
                        help='Minimum number of flows required per application '
                             '(default=1).')
    parser.add_argument('-f', '--fraction-certificates', type=float, default=0,
                        help="Fraction of certificates to change in test set. "
                             "(Default=0)")
    parser.add_argument('-i', '--fraction-ips', type=float, default=0,
                        help="Fraction of IPs to change in test set. "
                             "(Default=0)")
    parser.add_argument('-r', '--recognition', action='store_true',
                        help='If given perform recognition evaluation.')
    parser.add_argument('-a', '--anomaly', action='store_true',
                        help='If given perform anomaly detection evaluation.')
    parser.add_argument('--filter', action='store_true',
                        help='If given, filter out non-encrypted traffic.')


    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                          Preprocessing step                          #
    ########################################################################

    # Load data if necessary
    if args.load:
        loader = Loader()
        y_malicious = None

        # Cross-market dataset
        if any('cross_market' in l for l in args.load):
            print("Processing cross market dataset")

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

        elif any('andrubis' in x.lower() for x in args.load):
            print("Processing Andrubis dataset")

            # Load virus total file
            with open('../../data/Andrubis/appdict_scores.json', 'rb') as infile:
                relabel = json.load(infile)

            # Load data
            X, y = loader.load_files(args.load, args.min_flows)
            y = np.array([relabel.get(y_, {}).get('name', y_) for y_ in y])

        elif any('recon' in x.lower() for x in args.load):
            print("Processing ReCon dataset")

            # Initialise data
            X = list()
            y = list()

            # Load each file individually
            for load in args.load:
                # Load data
                X_, y_ = loader.load_file(load)
                # Create specific labels for cross_market
                y_ = np.asarray([load.split('/')[-2]]*X_.shape[0])

                # Add input file to data
                if X_.shape[0] >= args.min_flows:
                    X.append(X_)
                    y.append(y_)

            # Concatenate results
            X = np.concatenate(X)
            y = np.concatenate(y)

        else:
            print("Processing other dataset.")
            X, y = loader.load_files(args.load, args.min_flows)

    # If set, only use encrypted traffic
    if args.filter:
        print("{} items before filtering...".format(X.shape[0]))
        # Create mask
        mask = np.asarray([bool(x.certificates) for x in X])
        # Perform filtering
        X = X[mask]
        y = y[mask]
        print("{} items after filtering...".format(X.shape[0]))

    # Print number of applications in dataset
    print("Number of applications: {}".format(np.unique(y).shape[0]))
    print("Number of flows       : {}".format(X.shape[0]))

    ########################################################################
    #                            Run Evaluation                            #
    ########################################################################

    # If specified perform app recognition evaluation
    if args.recognition:
        # Create recogniser
        recogniser = RecogniserTransformed(
                        fraction_certificates = args.fraction_certificates,
                        fraction_ips          = args.fraction_ips)

        # Run recogniser
        result, result_m = recogniser.run(X, y,
                                 args.batch,
                                 args.window,
                                 args.correlation,
                                 args.similarity,
                                 args.kfolds,
                                 args.max_apps,
                                 y_malicious,
                                 verbose=True)
        # Show results
        recogniser.show(result)

        if result_m:
            print("Malicious apps only")
            recogniser.show(result_m)

    # If specified perform unseen app detection evaluation
    if args.anomaly:
        # Create AnomalyDetector
        detector = AnomalyDetectorTransformed(
                        fraction_certificates = args.fraction_certificates,
                        fraction_ips          = args.fraction_ips)

        # Run detector
        result = detector.run(X, y,
                                 args.batch,
                                 args.window,
                                 args.correlation,
                                 args.similarity,
                                 args.kfolds,
                                 args.max_apps,
                                 verbose=True)
        # Show results
        detector.show(result)
