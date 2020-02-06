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
import numpy as np
import warnings

from os import path
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../flowprint')))
from cluster import Cluster
from fingerprint import Fingerprint
from fingerprints import Fingerprints
from loader import Loader
from preprocessor import Preprocessor
from split import Splitter
from knn import KNN

class AnomalyDetector(object):

    def __init__(self):
        """AnomalyDetector object for evaluating FlowPrint anomaly detection capabilities."""
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        # Set splitter oject
        self.splitter = Splitter()

    ########################################################################
    #                     Evaluation execution methods                     #
    ########################################################################

    def run(self, X, y, batch, window, correlation, similarity, k_folds=1,
            max_labels=None, verbose=False):
        """Run an evaluation with given data and FlowPrint parameters.

            Parameters
            ----------
            X : np.array of shape=(n_flows,)
                Array of flows to evaluate.

            y : np.array of shape=(n_flows,)
                Array of labels corresponding to flows.

            batch : float
                Length of a single input batch in seconds.
                Inputs of X are first divided into batches of the given length
                and each batch is subsequently run through the system.

            window : float
                Length of an activity window in seconds.
                Each batch is subdivided into windows and activity correlation
                is computed when destination clusters are active in the same
                time window.

            correlation : float
                Minimum required correlation between destination clusters in
                order to be regarded as correlating. If they are correlating,
                clusters receive an edge between them in the cross-correlation
                graph.

            similarity : float
                Minimum required Jaccard similarity for fingerprints to be
                considered equivalent.

            k_folds : int, default=1
                Number of K-folds to run.

            max_labels : int, optional
                If given, return a maximum number of labels per k-fold.

            verbose : boolean, default=False
                If given, output performance for each k.

            Returns
            -------
            result : dict of string -> float
                Dictionary of metric -> value.
            """
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

            # print("Train  :\n  -{}".format("\n  -".join(np.unique(y_train))))
            # print("Known  :\n  -{}".format("\n  -".join(np.unique(y_known))))
            # print("Unknown:\n  -{}".format("\n  -".join(np.unique(y_unknown))))

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

    def setup(self, batch, window, correlation, similarity):
        """Setup clustering system with given parameters.

            Parameters
            ----------
            batch : float
                Length of a single input batch in seconds.
                Inputs of X are first divided into batches of the given length
                and each batch is subsequently run through the system.

            window : float
                Length of an activity window in seconds.
                Each batch is subdivided into windows and activity correlation
                is computed when destination clusters are active in the same
                time window.

            correlation : float
                Minimum required correlation between destination clusters in
                order to be regarded as correlating. If they are correlating,
                clusters receive an edge between them in the cross-correlation
                graph.

            similarity : float
                Minimum required Jaccard similarity for fingerprints to be
                considered equivalent.
            """
        # Create clustering object
        self.cluster = Cluster()
        # Create Fingerprinter
        self.fingerprints = Fingerprints(self.cluster,
                                         batch,
                                         window,
                                         correlation,
                                         similarity)

    def evaluate(self, X_train, y_train, X_known, y_known, X_unknown, y_unknown,
                 batch=300, window=30, correlation=0.1, similarity=0.9,
                 verbose=False):
        """Run fingerprinting system and return evaluation.

            Parameters
            ----------
            X_train : np.array of shape=(n_flows,)
                Array of flows to train FlowPrint.

            y_train : np.array of shape=(n_flows,)
                Array of labels to train FlowPrint.

            X_known : np.array of shape=(n_flows,)
                Array of known flows to test FlowPrint.

            y_known : np.array of shape=(n_flows,)
                Array of known labels to test FlowPrint.

            X_unknown : np.array of shape=(n_flows,)
                Array of known flows to test FlowPrint.

            y_unknown : np.array of shape=(n_flows,)
                Array of known labels to test FlowPrint.

            verbose : boolean, default=False
                Prints progress if True.

            Returns
            -------
            result : dict of string -> float
                Dictionary of metric -> value.
            """
        # Create test data
        X_test = np.concatenate((X_known, X_unknown))
        y_test = np.concatenate((np.zeros(X_known  .shape[0]) + 1,
                                 np.zeros(X_unknown.shape[0]) - 1))

        # Get fingerprints for training
        self.setup(batch, window, correlation, similarity)
        fp_train = self.fingerprints.fit_predict_fingerprints(X_train, y_train)
        # Get fingerprints for testing
        self.setup(batch, window, correlation, similarity)
        fp_test  = self.fingerprints.fit_predict_fingerprints(X_test)

        # Match test fingerprints with train
        fp_unique_train = set(fp_train)
        fp_unique_test  = set(fp_test)

        # Map fingerprints
        mapping = self.fingerprints.isin(fp_test, fp_train, 0.1, verbose)
        # Get mapping from fingerprints to True/False (known/unknown)
        y_pred = np.array([mapping.get(fp, False) for fp in fp_test])
        # Get prediction as +1 (benign) -1 (anomaly)
        y_pred = (y_pred.astype(int) * 2) - 1

        # Compute True/False Positives/Negatives
        # True positive, anomaly detected as anomaly
        tp = np.logical_and(y_pred == -1, y_test == -1).sum()
        # True negative, benign detected as benign
        tn = np.logical_and(y_pred ==  1, y_test ==  1).sum()
        # False positive, benign detected as anomaly
        fp = np.logical_and(y_pred == -1, y_test ==  1).sum()
        # False negative, anomaly detected as benign
        fn = np.logical_and(y_pred ==  1, y_test == -1).sum()

        # Check first time an app is detected as anomalous
        labels = np.concatenate((y_known, y_unknown))
        first_detected        = np.logical_and(y_pred == -1, y_test == -1)
        first_detected_flows  = X_test[first_detected]
        first_detected_labels = labels[first_detected]
        unknowns = np.unique(y_unknown)
        detection = np.zeros((unknowns.shape[0], 2))

        for i, label in enumerate(unknowns):
            if label in unknowns:
                flows = X_test[labels == label]
                start = min(flow.time_start() for flow in flows)
                end   = max(flow.time_end()   for flow in flows)
                detected_flows = X_test[np.logical_and(labels == label, first_detected)]
                if detected_flows.shape[0] != 0:
                    first_detected_flow = min(flow.time_start() for flow in detected_flows)
                else:
                    first_detected_flow = float('inf')
                detection[i, 0] = first_detected_flow - start
                detection[i, 1] = end-start

        print(detection)


        # Loop over unknown labels
        detected_apps = 0
        total_apps    = np.unique(y_unknown).shape[0]
        # Get prediction
        pred_unknown = y_pred[-y_unknown.shape[0]:]
        for label in np.unique(y_unknown):
            # Get relevant prediction for application
            pred_label = pred_unknown[y_unknown == label]
            # Check if at least 1 flow was predicted as anomalous
            if np.any(pred_label == -1):
                # Increment number of detected applications
                detected_apps += 1
            else:
                print("Undetected application with {:4} flows {}".format(pred_label.shape[0], label))

        print("Detected {}/{} = {:.4f} unknown applications".format(detected_apps, total_apps, detected_apps/total_apps))

        # Compute metrics and return
        return {
            'true positives' : tp,
            'true negatives' : tn,
            'false positives': fp,
            'false negatives': fn
        }

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def show(self, metrics):
        """Print given metrics in a user readable format.

            Parameters
            ----------
            metrics : dict
                Dictionary of metric -> value.
            """
        tp = int(metrics.get("true positives"))
        tn = int(metrics.get("true negatives"))
        fp = int(metrics.get("false positives"))
        fn = int(metrics.get("false negatives"))

        print("""
Flowprint evaluation
--------------------------
  True  Positives: {}
  True  Negatives: {}
  False Positives: {}
  False Negatives: {}
  Precision      : {:.4f}
  Recall         : {:.4f}
  F1-score       : {:.4f}
  Accuracy       : {:.4f}
  """.format(tp, tn, fp, fn,
             tp/(tp+fp),
             tp/(tp+fn),
             2*tp/(2*tp+fp+fn),
             (tp+tn)/(tp+tn+fp+fn)))


if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint Anomaly Detector.')
    parser.add_argument('-f', '--files', nargs='+',
                        help='pcap files to run through FlowPrint. We use the '
                             'directory of each file as label.')
    parser.add_argument('-a', '--save',
                        help='Save preprocessed data to given file.')
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

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                          Preprocessing step                          #
    ########################################################################

    # Initialise preprocessor
    preprocessor = Preprocessor(verbose=True)

    # Parse files if required
    if args.files:
        # Extract files and labels
        files  = args.files
        labels = extract_labels(args.files)

        # Load data
        X, y = preprocessor.process(files, labels)

        # Save preprocessed data if necessary
        if args.save:
            preprocessor.save(args.save, X, y)

    # Load data if necessary
    if args.load:
        if any('andrubis' in x.lower() for x in args.load):
            loader = Loader('../../data/Andrubis/appdict.p', version=True)
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
    # exit()

    ########################################################################
    #                            Run Evaluation                            #
    ########################################################################

    # Create AnomalyDetector
    detector = AnomalyDetector()
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
