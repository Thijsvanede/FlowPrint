from cluster import Cluster
from collections import Counter
from fingerprint import Fingerprint
from fingerprints import FingerprintGenerator
import copy
import json
import numpy as np

class FlowPrint(object):

    def __init__(self, batch=300, window=30, correlation=0.1, similarity=0.9,
                 threshold=0.1):
        """FlowPrint object for creating fingerprints from mobile network traffic.

            Parameters
            ----------
            batch : float, default=300
                Threshold for the batch size in seconds.

            window : float, default=30
                Threshold for the window size in seconds.

            correlation : float, default=0.1
                Threshold for the minimum required correlation

            similarity : float, default=0.9
                Threshold for the minimum required similarity.

            threshold : float, default=0.1
                Threshold for anomaly detection.
            """
        # Set parameters
        self.batch       = batch
        self.window      = window
        self.correlation = correlation
        self.similarity  = similarity

        # Set default anomaly threshold
        self.threshold = threshold

        # Create fingerprint generator
        self.fingerprinter = FingerprintGenerator(
            batch       = self.batch,
            window      = self.window,
            correlation = self.correlation,
            similarity  = self.similarity
        )

        # Store fingerprints
        self.fingerprints = dict()

    ########################################################################
    #                         Fit/predict methods                          #
    ########################################################################

    def fit(self, X, y=None):
        """Fit FlowPrint object with fingerprints from given flows.

            Parameters
            ----------
            X : np.array of shape=(n_samples,)
                Flows for fitting FlowPrint.

            y : np.array of shape=(n_samples,), optional
                If given, attach labels to fingerprints from X.

            Returns
            -------
            self : self
                Returns FlowPrint object
            """
        # Reset fingerprints
        self.fingerprints = dict()

        # Update fingerprints and return self
        return self.update(X, y)


    def update(self, X, y=None):
        """Update list of fingerprints with given flows.

            Parameters
            ----------
            X : np.array of shape=(n_samples,)
                Flows for fitting FlowPrint.

            y : np.array of shape=(n_samples,), optional
                If given, attach labels to fingerprints from X.

            Returns
            -------
            self : self
                Returns FlowPrint object
            """
        # Transform X and y to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y) if y is not None else None

        # Create fingerprints from X
        fingerprints = self.fingerprinter.fit_predict(X)

        # Set all fingerprints to 1 in case of no label
        if y is None:
            # Add all fingerprints
            for fp in set(fingerprints):
                # Set fingerprint label to 1
                self.fingerprints[fp] = 1

        # Set all fingerprints to label in case of label
        else:
            # Loop over all fingerprints
            for fingerprint, label in zip(fingerprints, y):
                # Get counter
                counter = self.fingerprints.get(fingerprint, Counter())
                # Check if result is of type counter
                if isinstance(counter, Counter):
                    # Add label to counter
                    counter.update([label])
                    # Set counter
                    self.fingerprints[fingerprint] = counter
                else:
                    # We cannot deal with non-counter entries
                    raise ValueError("Not implemented yet.")

            # Get most common labels
            self.fingerprints = {k: v.most_common(1)[0][0]
                             for k, v in self.fingerprints.items()}

        # Return self
        return self

    def predict(self, X, y=None, threshold=None):
        """Wrapper for predict anomaly"""
        return self.predict_anomaly(X, y, threshold)


    def predict_anomaly(self, X, y=None, threshold=None):
        """Predict whether samples of X are anomalous or not.

            Parameters
            ----------
            X : np.array of shape=(n_samples,)
                Flows for fitting FlowPrint.

            y : Ignored

            threshold : float, default=None
                Minimum required threshold to consider point benign.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Prediction of samples in X: +1 if benign, -1 if anomalous.
            """
        # Transform X and y to numpy arrays
        X = np.asarray(X)

        # Create fingerprints from X
        fingerprints = self.fingerprinter.fit_predict(X)
        fingerprints = np.asarray([fp.as_set() for fp in fingerprints])

        # Compute anomalies from fingerprints and return
        return self.predict_anomaly_fingerprints(fingerprints, X.shape[0],
            threshold=threshold or self.threshold)


    def predict_anomaly_fingerprints(self, fingerprints, size, threshold=None):
        """"""
        # Compute distance of each fingerprint
        distance, X_unique, Y_unique = self.jaccard(fingerprints,
                                          list(self.fingerprints.keys()))
        # Lookup indices of X
        X_mapping = {x: i for i, x in enumerate(X_unique)}
        # Check which fingerprints are benign
        benign = np.any(distance >= (threshold or self.threshold), axis=1)

        # Initialise result
        result = np.zeros(size, dtype=int) - 1

        # Loop over all fingerprints
        for i, fingerprint in enumerate(fingerprints):
            # Add fingerprint prediction
            result[i] = 1 if benign[X_mapping.get(fingerprint)] else -1

        # Return result
        return result


    def fit_predict_anomaly(self, X, y=None):
        """Fit FlowPrint with given flows and predict the same flows.

            TODO"""
        # Call fit and predict respectively
        return self.fit(X, y).predict_anomalous(X, y)

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def store(self, file, fingerprints=None):
        """Store fingerprints to file.

            Parameters
            ----------
            file : string
                File in which to store flowprint fingerprints.

            fingerprints : iterable of Fingerprint (optional)
                If None export fingerprints from fitted FlowPrint object,
                otherwise, export given fingerprints.
            """
        # Prepare output as dictionary
        output = {
            'batch'       : self.batch,
            'window'      : self.window,
            'correlation' : self.correlation,
            'similarity'  : self.similarity,
            'threshold'   : self.threshold,
            'fingerprints': [[fp.to_dict(), self.fingerprints.get(fp, file)]
                             for fp in fingerprints or self.fingerprints]
        }

        # Open output file
        with open(file, 'w') as outfile:
            # Dump fingerprints to outfile
            json.dump(output, outfile)

    def load(self, *files, store=False):
        """Load fingerprints from files.

            Parameters
            ----------
            file : string
                Files from which to load fingerprints.

            store : boolean, default=False
                If True, also update FlowPrint parameters from file

            Returns
            -------
            result : dict of Fingerprint -> label
                Fingerprints imported from file.
            """
        # Loop over all files
        for file in files:
            # Open input file
            with open(file, 'r') as infile:
                # Load fingerprints
                data = json.load(infile)

                # Store values if necessary
                if store:
                    self.batch       = data.get('batch'      , self.batch)
                    self.window      = data.get('window'     , self.window)
                    self.correlation = data.get('correlation', self.correlation)
                    self.similarity  = data.get('similarity' , self.similarity)
                    self.threshold   = data.get('threshold'  , self.threshold)

                # Add fingerprints
                for fp, label in data.get('fingerprints'):
                    # Transform json to Fingerprint
                    fp = Fingerprint().from_dict(fp)
                    # Get label
                    label = self.fingerprints.get(fp, set()) | set([label])
                    # Set fingerprint
                    self.fingerprints[fp] = label

        # Return self
        return self


    ########################################################################
    #                          Comparison methods                          #
    ########################################################################

    def jaccard(self, X, Y=None):
        """Compare fingerprints using the Jaccard distance.

            Parameters
            ----------
            X : np.array of shape=(n_samples,)
                Fingerprints used to compute the jaccard knn with Y (if given)
                or self, if Y not given.

            Y : np.array of shape=(n_samples,), optional
                Fingerprints used to compute the jaccard knn with X if given

            Returns
            -------
            scores : np.array of shape=(n_unique_x, n_unique_y)
                Similarity scores of fingerprints

            x_unique : np.array of shape=(n_unique_x,)
                Unique samples in X indexed same as scores

            y_unique : np.array of shape=(n_unique_y,)
                Unique samples in Y indexed same as scores
            """
        # Get both inputs as numpy array
        X = np.asarray(X)
        Y = np.asarray(Y) if Y is not None else X

        # Get unique inputs
        X_unique = np.asarray(list(sorted(set(X))))
        Y_unique = np.asarray(list(sorted(set(Y))))

        # Get mapping of destination -> fingerprints
        mapping = dict()
        # Loop over all training fingerprints
        for j, fingerprint in enumerate(Y_unique):
            # Loop over all destinations in fingerprint
            for destination in fingerprint:
                # Get fingerprint set
                fps = mapping.get(destination, set()) | set([(j, fingerprint)])
                # Add fingerprint to destination
                mapping[destination] = fps

        # Initialise result
        result = np.zeros((X_unique.shape[0], Y_unique.shape[0]), dtype=float)

        # Loop over all testing fingerprints
        for i, fingerprint in enumerate(X_unique):
            # Initialise partial matches
            matches = set()
            # Find partial matches
            for destination in fingerprint:
                matches |= mapping.get(destination, set())

            # Compute score per partial match
            for j, match in matches:
                # Compute fingerprint matching score
                score = len(fingerprint & match) / max(len(fingerprint | match), 1)
                # Assign score to result
                result[i, j] = score

        # Return result
        return result, X_unique, Y_unique

    ########################################################################
    #                             Copy method                              #
    ########################################################################

    def copy(self):
        """Create copy of self."""
        # Create new flowprint instance
        result = FlowPrint(self.batch, self.window, self.correlation,
                         self.similarity)

        # Set fingerprints
        for fingerprint, label in self.fingerprints.items():
            result.fingerprints[copy.deepcopy(fingerprint)] = copy.deepcopy(label)

        # Return result
        return result
