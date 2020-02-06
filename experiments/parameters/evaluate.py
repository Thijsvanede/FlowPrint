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
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../flowprint')))
from cluster import Cluster
from fingerprints import Fingerprints
from loader import Loader
from preprocessor import Preprocessor
from split import Splitter

class Evaluator(object):

    def __init__(self):
        """Evaluator for evaluating FlowPrint under different parameters."""
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
            # Setup fingerprinting system
            self.setup(batch, window, correlation, similarity)
            # Perform evaluation
            partial_result = self.evaluate(X_, y_)

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

    def evaluate(self, X, y):
        """Run fingerprinting system and return evaluation.

            Parameters
            ----------
            X : np.array of shape=(n_flows,)
                Array of flows to evaluate.

            y : np.array of shape=(n_flows,)
                Array of labels corresponding to flows.

            Returns
            -------
            result : dict of string -> float
                Dictionary of metric -> value.
            """
        # Get fingerprints
        y_pred = self.fingerprints.fit_predict_fingerprints(X, y)
        # Get labels and fingerprints
        y_pred_labels = np.array([fp.label for fp in y_pred]).astype(y.dtype)
        y_pred_fingerprints = np.array([hash(y_) for y_ in y_pred])

        # print("{:4} applications.".format(np.unique(y).shape[0]))
        # print("{:4} fingerprints.".format(np.unique(y_pred_fingerprints).shape[0]))

        # Compute classification report
        class_report = classification_report(y, y_pred_labels, output_dict=True)

        # Compute metrics and return
        return {
            'accuracy'    : accuracy_score (y, y_pred_labels),
            'precision'   : class_report.get('weighted avg').get('precision'),
            'recall'      : class_report.get('weighted avg').get('recall'),
            'f1-score'    : class_report.get('weighted avg').get('f1-score'),
            'homogeneity' : homogeneity_score  (y, y_pred_fingerprints),
            'completeness': completeness_score (y, y_pred_fingerprints),
            'v-measure'   : v_measure_score    (y, y_pred_fingerprints),
            'ari'         : adjusted_rand_score(y, y_pred_fingerprints)
        }

    def degree(self, fingerprints, y):
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

    def degree_hist(self, fingerprints, y):
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
        degree = self.degree(fingerprints, y)

        # Create histogram of degrees
        for label, count in degree.items():
            # Add result to histogram
            result[count] = result.get(count, 0) + 1

        # Return result
        return result

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def plot_degree(self, histogram, cap=-1):
        """Plot degree histogram.

            Parameters
            ----------
            histogram : dict
                Dictionary of #fingerprints -> count.

            cap : int, optional
                If positive, cap at
            """
        # Set cap if not given
        if cap <= 0: cap = max(histogram.keys())

        # Initialise counts
        labels = list(range(cap))
        counts = list()

        # Loop over all labels
        for label in labels:
            # Get counts from histogram
            counts.append(histogram.get(label, 0))

        # Plot bar chart
        plt.bar(labels, counts, align='center')
        plt.show()

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

  Class based evaluation
  ----------------------
  Precision: {:.4f}
  Recall   : {:.4f}
  F1-score : {:.4f}
  Accuracy : {:.4f}

  Cluster based evaluation
  ------------------------
  Homogeneity : {:.4f}
  Completeness: {:.4f}
  V-measure   : {:.4f}
  """.format(metrics.get("precision"),
             metrics.get("recall"),
             metrics.get("f1-score"),
             metrics.get("accuracy"),
             metrics.get("homogeneity"),
             metrics.get("completeness"),
             metrics.get("v-measure")))


if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint evaluator.')
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
                        help='Maximum number of applications per k-fold (optional).')

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

        X, y = loader.load_files(args.load)

    ########################################################################
    #                            Run Evaluation                            #
    ########################################################################

    # Create evaluator
    evaluator = Evaluator()
    # Run evaluator
    result = evaluator.run(X, y, args.batch,
                                 args.window,
                                 args.correlation,
                                 args.similarity,
                                 args.kfolds,
                                 args.max_apps)
    # Show results
    evaluator.show(result)
