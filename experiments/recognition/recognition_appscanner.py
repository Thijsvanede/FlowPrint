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
import multiprocessing
import numpy as np
import warnings

from os import path
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../flowprint')))
from loader import Loader
from preprocessor import Preprocessor
from split import Splitter
from appscanner.features import Features
from appscanner.appscanner import AppScanner

class Recogniser(object):

    def __init__(self):
        """Recogniser object for evaluating FlowPrint recognition capabilities."""
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        # Set splitter oject
        self.splitter = Splitter()

    ########################################################################
    #                     Evaluation execution methods                     #
    ########################################################################

    def run(self, X, y, threshold=0.9, k_folds=1, max_labels=None,
            verbose=False):
        """Run an evaluation with given data and FlowPrint parameters.

            Parameters
            ----------
            X : np.array of shape=(n_flows,)
                Array of flows to evaluate.

            y : np.array of shape=(n_flows,)
                Array of labels corresponding to flows.

            threshold : float, default=0.9
                Threshold of AppScanner system

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

        # Perform single transformation
        table, X_trans, y_trans = self.transform(X, y, verbose=verbose)

        # Iterate over all k-folds
        for i, (X_, y_) in enumerate(self.splitter.k_fold_label(X, y, k_folds,
                                                   max_labels=max_labels)):

            # Split in train-test data
            X_train, X_test, y_train, y_test = train_test_split(X_, y_,
                                                                test_size=0.33,
                                                                random_state=i)

            # Get appscanner data from each train-test instance
            i_train = [table.get(d, list()) for d in X_train]
            i_test  = [table.get(d, list()) for d in X_test ]

            # Flatten lists
            i_train = np.array([item for sublst in i_train for item in sublst])
            i_test  = np.array([item for sublst in i_test  for item in sublst])

            # Retrieve data
            X_train = X_trans[i_train]
            y_train = y_trans[i_train]
            X_test  = X_trans[i_test]
            y_test  = y_trans[i_test]

            # Perform evaluation
            partial_result = self.evaluate(X_train, y_train, X_test, y_test,
                                           threshold, verbose)

            # Show result if verbose
            if verbose:
                print("----- K-fold: {} -----".format(i))
                self.show(partial_result)

            # Add partial result
            result = {k: v+result.get(k, 0) for k, v in partial_result.items()}

        # Average result
        result = {k: v/k_folds for k, v in result.items()}

        # Return average result
        return result

    def evaluate(self, X_train, y_train, X_test, y_test,
                       threshold=0.9, verbose=False):
        """Run fingerprinting system and return evaluation.

            Parameters
            ----------
            X_train : np.array of shape=(n_flows,)
                Array of flows to train FlowPrint, must be in AppScanner format.

            y_train : np.array of shape=(n_flows,)
                Array of labels to train FlowPrint, must be in AppScanner format.

            X_test : np.array of shape=(n_flows,)
                Array of flows to test FlowPrint.

            y_test : np.array of shape=(n_flows,)
                Array of labels to test FlowPrint.

            threshold : float, default=0.9
                Threshold of AppScanner system

            verbose : boolean, default=False
                If given, output progress.

            Returns
            -------
            result : dict of string -> float
                Dictionary of metric -> value.
            """
        # Initialise AppScanner instance
        scanner = AppScanner(threshold=threshold)
        if verbose: print("Fitting {} samples...".format(X_train.shape))
        # Fit scanner with training data
        scanner.fit(X_train, y_train)
        if verbose: print("Fitting complete.")
        if verbose: print("Predicting {} samples...".format(X_test.shape))
        # Predict testing data
        y_pred = scanner.predict(X_test)
        if verbose: print("Predicting complete.")

        # Perform evaluation between predicted labels and test labels
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # # Compute metrics and return
        # return {
        #     'accuracy'  : accuracy_score(y_test, y_pred),
        #     'precision' : precision_score(y_test, y_pred),
        #     'recall'    : recall_score(y_test, y_pred),
        #     'f1-score'  : f1_score(y_test, y_pred),
        # }

        return {
            'accuracy'  : accuracy_score(y_test, y_pred),
            'precision' : class_report.get('weighted avg').get('precision'),
            'recall'    : class_report.get('weighted avg').get('recall'),
            'f1-score'  : class_report.get('weighted avg').get('f1-score'),
        }

    def transform(self, X, y, processes=4, verbose=False):
        """Transform flows into AppScanner readable data."""
        ########################################################################
        #            Transform flows into AppScanner readable flows            #
        ########################################################################

        # Initialise translation table
        table = dict()

        # Get flow lengths in bursts
        flows  = list()
        labels = list()

        # Loop over all flows and labels
        for flow, label in zip(X, y):
            # Compute difference between packets
            diff = np.diff(flow.timestamps)
            # Select indices where difference is greater than 1 second
            indices_split = np.argwhere(diff > 1)

            # Add 0 as start and length as end index
            indices_split = [0] + list(indices_split.flatten()) + [len(flow.lengths)]

            for start, end in zip(indices_split, indices_split[1:]):
                # Add to translation table
                table[flow] = table.get(flow, list()) + [len(labels)]
                # Append flows and labels
                flows .append(flow.lengths[start+1:end+1])
                labels.append(label)

        # Transform to numpy arrays
        X = np.asarray(flows)
        y = np.asarray(labels)

        # Create jobs for multiprocessing
        jobs = list()
        # Create queue
        queue = multiprocessing.Queue()
        # Split input for each job
        split = np.array_split(np.arange(X.shape[0]), processes)

        for job in range(processes):
            # Create process
            process = multiprocessing.Process(target=self.transform_process,
                                args=(X[split[job]], y[split[job]], queue, job,
                                      verbose))
            # Append to jobs
            jobs.append(process)

        # Start jobs
        for job in jobs:
            job.start()

        # Get retrieved items
        retrieved_X = [None for job in jobs]
        retrieved_y = [None for job in jobs]

        # Read queue
        while any([x is None for x in retrieved_X]):
            flows, labels, job = queue.get()
            retrieved_X[job] = flows
            retrieved_y[job] = labels

        # End jobs
        for job in jobs:
            job.join()

        X = np.concatenate(retrieved_X)
        y = np.concatenate(retrieved_y)

        # Return result as numpy array
        return table, np.asarray(X), np.asarray(y)

    def transform_process(self, X, y, queue, job, verbose=True):
        """Transform flows in parallel.

            Parameters
            ----------
            X : iterable of flows
                Flows to process.

            y : iterable of labels
                Labels corresponding to flows.

            queue : multiprocessing.Queue
                Queue to write output to

            job : int
                Job index

            verbose : boolean, default=False
                If True, print verbosity level
            """
        # Get feature extractor
        features = Features()

        # Initialise result
        flows  = np.zeros((len(X), 54), dtype=float)
        labels = np.zeros(len(y), dtype=str)

        # Loop over all flows
        for pos, (flow, label) in enumerate(zip(X, y)):
            flows [pos] = features.extract_single(np.asarray(flow))
            labels[pos] = label
            # Print if verbose
            if verbose and not pos % 5000:
                print("{}: {}/{}".format(job, pos, X.shape[0]))

        # Append flow to queue
        queue.put((flows, labels, job))


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
    parser.add_argument('-k', '--kfolds', type=int, default=1,
                        help='Number of k-folds to perform (default=1).')
    parser.add_argument('-m', '--max-apps', type=int,
                        help='Maximum number of applications '
                             'per k-fold (optional).')
    parser.add_argument('-p', '--min-flows', type=int, default=1,
                        help='Minimum number of flows required per application '
                             '(default=1).')
    parser.add_argument('-t', '--threshold', type=float, default=0.7,
                        help='AppScanner threshold (default=0.7).')

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

    # Create recogniser
    recogniser = Recogniser()
    # Run recogniser
    result = recogniser.run(X, y, args.threshold, args.kfolds, args.max_apps, verbose=True)
    # Show results
    recogniser.show(result)
