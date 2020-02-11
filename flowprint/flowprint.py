from collections import Counter
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

    def predict(self, X, y=None, default='common'):
        """Find closest fingerprint to trained fingerprints

            Parameters
            ----------
            X : Array-like of Fingerprint of shape=(n_fingerprints,)
                Fingerprints to compare against training set.

            y : ignored

            default : 'common'|'largest'|other, default='common'
                Default to this strategy if no match is found
                 - 'common' : return the fingerprint with most flows
                 - 'largest': return the largest fingerprint
                 - other: return <other> as match, e.g. Fingerprint()/None

            Returns
            -------
            result : np.array of shape=(n_fingerprints,)
                Closest matching fingerprints to original.
                If no match is found, fall back on default
            """
        # Initialise result
        result = np.zeros(len(X), dtype=object)
        # Set default strategy
        if default == 'common':
            default = max(self.fingerprints, key=lambda x: x.n_flows)
        elif default == 'largest':
            default = max(self.fingerprints, key=lambda x: len(x))

        # Transform Fingerprints into quick lookup dictionary
        lookup = dict()
        # Loop over trained fingerprints
        for fingerprint in self.fingerprints:
            # For all destination in fingerprint
            for dst in fingerprint:
                # Add corresponding fingerprints
                lookup[dst] = lookup.get(dst, set()) | set([fingerprint])

        # Loop over all fingerprints in X
        for i, fingerprint in enumerate(X):
            # Get all matches corresponding to fingerprint
            matches = list(set().union(*[lookup.get(x, set()) for x in fingerprint]))
            # Default strategy if no match
            if not matches:
                result[i] = default
            else:
                # Find highest match between possible matches
                scores = np.asarray([fingerprint.compare(m) for m in matches])
                # Set maximum score
                result[i] = matches[scores.argmax()]

        # Return result
        return result

    def recognize(self, X, y=None):
        """Return labels corresponding to closest matching fingerprints

            Parameters
            ----------
            X : Array-like of Fingerprint of shape=(n_fingerprints,)
                Fingerprints to compare against training set.

            y : ignored

            Returns
            -------
            result : np.array of shape=(n_fingerprints,)
                Label of closest matching fingerprints to original
            """
        # Perform predict and return corresponding fingerprints
        return np.asarray([self.fingerprints.get(x) for x in self.predict(X)])


    def detect(self, X, y=None, threshold=None):
        """Predict whether samples of X are anomalous or not.

            Parameters
            ----------
            X : np.array of shape=(n_samples,)
                Flows for fitting FlowPrint.

            y : Ignored

            threshold : float, default=None
                Minimum required threshold to consider point benign.
                If None is given, use FlowPrint default

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Prediction of samples in X: +1 if benign, -1 if anomalous.
            """
        # Get best match for each fingerprint
        prediction = self.predict(X, default=Fingerprint())
        # Compute match score between each best match
        prediction = np.asarray([x.compare(fp) for x, fp in zip(X, prediction)])
        # Return whether matching score is high enough
        return (prediction >= (threshold or self.threshold)) * 2 - 1

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

    def load(self, *files, store=True, parameters=False):
        """Load fingerprints from files.

            Parameters
            ----------
            file : string
                Files from which to load fingerprints.

            store : boolean, default=True
                If True, store fingerprints in FlowPrint object

            parameters : boolean, default=False
                If True, also update FlowPrint parameters from file

            Returns
            -------
            result : dict of Fingerprint -> label
                Fingerprints imported from file.
            """
        # Initialise fingerprints
        fingerprints = dict()

        # Loop over all files
        for file in files:
            # Open input file
            with open(file, 'r') as infile:
                # Load fingerprints
                data = json.load(infile)

                # Store parameters if necessary
                if parameters:
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
                    label = fingerprints.get(fp, set()) | set([label])
                    # Set fingerprint
                    fingerprints[fp] = label

        # Store fingerprints if necessary
        if store:
            for k, v in fingerprints.items():
                self.fingerprints[k] = self.fingerprints.get(k, set()) | v

        # Return fingerprints
        return fingerprints
