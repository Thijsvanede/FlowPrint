from collections import Counter
from preprocessor import Preprocessor
import numpy as np
import pickle

class Loader():

    def __init__(self, relabel=None, version=False):
        """Loader object for loading preprocessed files.

            Parameters
            ----------
            relabel : string, optional
                Path to file containing relabeling dictionary.

            version : boolean, default=False
                If True, consider different version applications as different
                applications.
            """
        self.preprocessor = Preprocessor()
        self.relabels = relabel if relabel is None else self.relabel(relabel)
        self.version = version

    def relabel(self, infile):
        """Get relabeling dictionary from file.

            Parameters
            ----------
            infile : string
                Path to relabeling file.

            Returns
            -------
            result : dict
                Dictionary used for relabeling labels.
            """
        # Load relabel file
        with open(infile, 'rb') as infile:
            result = pickle.load(infile)

        # Transofrm relabel file and return
        return {k: v for k, v in result.items()}

    def load_files(self, infiles, min_flows=1):
        """Load multiple input files.

            Parameters
            ----------
            infile : string
                Path to input file.

            min_flows : int, default=1
                Minimum number of flows to load, otherwise discard input.

            Returns
            -------
            X : np.array of shape=(n_flows,)
                Array of loaded flows.

            y : np.array of shape=(n_flows,)
                Array of loaded labels.
            """
        # Compute set of labels to keep
        labels = self.get_labels(infiles, min_flows) if min_flows > 1 else None

        # Initialise results
        X = list()
        y = list()

        # Loop over all input files
        for infile in infiles:
            # Load input file
            X_, y_ = self.load_file(infile, labels)

            # Add input file to data
            X.append(X_)
            y.append(y_)

        # Concatenate results
        X = np.concatenate(X)
        y = np.concatenate(y)

        # Return result
        return X, y

    def load_file(self, infile, labels=None):
        """Load single input file.

            Parameters
            ----------
            infile : string
                Path to input file.

            labels : array-like of shape=(n_labels,), optional
                If given, only keep given labels.

            Returns
            -------
            X : np.array of shape=(n_flows,)
                Array of loaded flows.

            y : np.array of shape=(n_flows,)
                Array of loaded labels.
            """
        # Load files
        X, y = self.preprocessor.load(infile)

        # If relabel, relabel all y
        if self.relabels is not None:
            # Transform y
            y = np.array([self.relabels.get(x, (x, '0')) for x in y])
            # If version is False, only get application name
            if not self.version:
                y = np.array([label[0] for label in y])
            else:
                y = np.array(["{}::{}".format(label[0], label[1]) for label in y])

        # If labels, only keep given labels
        if labels is not None:
            # Get indices to keep
            indices = np.isin(y, np.asarray(labels))
            # Only keep given indices
            X = X[indices]
            y = y[indices]

        # Return result
        return X, y

    def get_labels(self, infiles, min_flows):
        """Get all flow labels with a minimum of min_flows entries in infiles.

            Parameters
            ----------
            infiles : list of string
                Files to check for flows.

            min_flows : int
                Minimum number of flows required.

            Returns
            -------
            result : np.array of shape=(n_labels,)
                Set of labels to keep.
            """
        # Initialise result
        result = Counter()

        # Loop over all files
        for infile in infiles:
            # Load contents of file
            X, y = self.load_file(infile)
            # Update result with y
            result.update(y)

        # Compute result and return
        return np.array(list({k for k, v in result.items() if v >= min_flows}))
