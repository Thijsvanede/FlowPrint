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
import pickle
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
from jaccard import closest

class Recogniser(object):

    def __init__(self):
        """Recogniser object for evaluating FlowPrint recognition capabilities."""
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        # Set splitter oject
        self.splitter = Splitter()

    ########################################################################
    #                     Evaluation execution methods                     #
    ########################################################################

    def run(self, X, y, batch, window, correlation, similarity, k_folds=1,
            max_labels=None, y_malicious=None, verbose=False):
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

            y_malicious : dict, optional
                Dictionary of label (string) -> malicious (boolean).

            verbose : boolean, default=False
                If given, output performance for each k.

            Returns
            -------
            result : dict of string -> float
                Dictionary of metric -> value.
            """
        # Initialise result
        result   = dict()
        result_m = dict()

        # TODO remove
        return self.evaluate(X, y, X, y, batch=batch, window=window,
                            correlation=correlation, similarity=similarity,
                            y_malicious=y_malicious, verbose=verbose)

        # Iterate over all k-folds
        for i, (X_, y_) in enumerate(self.splitter.k_fold_label(X, y, k_folds,
                                                max_labels=max_labels)):
            # Split in train-test data
            X_train, X_test, y_train, y_test = train_test_split(X_, y_,
                                                                test_size=0.33,
                                                                random_state=i)
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

    def evaluate(self, X_train, y_train, X_test, y_test,
                 batch=300, window=30, correlation=0.1, similarity=0.9,
                 y_malicious=None, verbose=False):
        """Run fingerprinting system and return evaluation.

            Parameters
            ----------
            X_train : np.array of shape=(n_flows,)
                Array of flows to train FlowPrint.

            y_train : np.array of shape=(n_flows,)
                Array of labels to train FlowPrint.

            X_test : np.array of shape=(n_flows,)
                Array of flows to test FlowPrint.

            y_test : np.array of shape=(n_flows,)
                Array of labels to test FlowPrint.

            y_malicious : dict, optional
                Dictionary of label (string) -> malicious (boolean).

            verbose : boolean, default=False
                Prints progress if True.

            Returns
            -------
            result : dict of string -> float
                Dictionary of metric -> value.
            """
        # Get fingerprints for training
        self.setup(batch, window, correlation, similarity)
        fp_train = self.fingerprints.fit_predict_fingerprints(X_train, y_train)
        # Get fingerprints for testing
        self.setup(batch, window, correlation, similarity)
        fp_test  = self.fingerprints.fit_predict_fingerprints(X_test)
        # Print number of flows if verbose
        if verbose:
            print("Flows:", fp_train.shape[0] + fp_test.shape[0])

        # return {
        #     'accuracy'  : 0,
        #     'precision' : 0,
        #     'recall'    : 0,
        #     'f1-score'  : 0,
        # }

        # # Map fingerprints
        # mapping = self.fingerprints.map(fp_test, fp_train, verbose)
        #
        # # Get mapping from fingerprints to labels of train fingerprints
        # y_pred = np.array([mapping.get(fp, fp) for fp in fp_test])

        # Get dictionaries for super quick mapping
        fp_set_train = {fp:fp.as_set() for fp in fp_train}
        fp_set_test  = {fp:fp.as_set() for fp in fp_test }
        set_fp_train = {fp.as_set():fp for fp in fp_train}
        set_fp_test  = {fp.as_set():fp for fp in fp_test }

        # Transform sets to numpy arrays
        nparray_train = np.array(list(fp_set_train.values()))
        nparray_test  = np.array(list(fp_set_test .values()))

        # Get mapping
        mapping = closest(nparray_train, nparray_test)

        # Map prediction to closest match
        y_pred = [mapping.get(fp_set_test[fp], fp) or fp for fp in fp_test]
        # Get corresponding fingerprint
        y_pred = np.array([set_fp_train.get(fp, fp) for fp in y_pred])

        # Extract labels from fingerprint
        y_pred = np.array([pred.label for pred in y_pred])

        # Perform evaluation between predicted labels and test labels
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Print classification report if verbose
        if verbose:
            print(classification_report(y_test, y_pred, digits=4))

        # Compute metrics and return
        result = {
            'accuracy'  : accuracy_score(y_test, y_pred),
            'precision' : class_report.get('weighted avg').get('precision'),
            'recall'    : class_report.get('weighted avg').get('recall'),
            'f1-score'  : class_report.get('weighted avg').get('f1-score'),
        }

        r_malicious = None
        # If malicious is given, perform separate analysis for malicious apps
        if y_malicious is not None:
            X_train = X_train[[y_malicious.get(label, False) for label in y_train]]
            y_train = y_train[[y_malicious.get(label, False) for label in y_train]]
            X_test  = X_test [[y_malicious.get(label, False) for label in y_test ]]
            y_test  = y_test [[y_malicious.get(label, False) for label in y_test ]]
            # Recompute statistics for malicious app
            r_malicious = self.evaluate(X_train, y_train, X_test, y_test,
                                        batch, window, correlation, similarity,
                                        None, verbose)[0]

        # Return result
        return result, r_malicious


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
        print("""
Flowprint evaluation
--------------------------
  Precision: {:.4f}
  Recall   : {:.4f}
  F1-score : {:.4f}
  Accuracy : {:.4f}
  """.format(metrics.get("precision"),
             metrics.get("recall"),
             metrics.get("f1-score"),
             metrics.get("accuracy")))


if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint recogniser.')
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
            with open('../../data/Andrubis/vt.p', 'rb') as infile:
                malicious = pickle.load(infile)

            # Load relabel file
            with open('../../data/Andrubis/appdict.p', 'rb') as infile:
                relabel = pickle.load(infile)
            relabel = {k: (v, malicious.get(k, [0, 0, 0])) for k, v in relabel.items()}

            # Load data
            X, y = loader.load_files(args.load, args.min_flows)
            # Relabel data
            y_malicious = {relabel.get(label, ([label],))[0][0]: relabel.get(label, ([label], [0, 0, 0]))[1][2] > 0.5 for label in np.unique(y)}
            y           = np.array([relabel.get(label, ([label], [0, 0, 0]))[0][0]     for label in y])

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

    # Print number of applications in dataset
    print("Number of applications: {}".format(np.unique(y).shape[0]))
    print("Number of flows       : {}".format(X.shape[0]))

    ########################################################################
    #                            Run Evaluation                            #
    ########################################################################

    # Create recogniser
    recogniser = Recogniser()
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
