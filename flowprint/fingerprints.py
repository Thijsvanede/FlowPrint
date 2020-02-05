from cross_correlation_graph import CrossCorrelationGraph
from cluster import Cluster
from fingerprint import Fingerprint
from itertools import combinations
import networkx as nx
import numpy as np

class Fingerprints(object):

    def __init__(self, batch=300, window=30, correlation=0.1, similarity=0.9):
        # Initialise fingerprints
        self.fingerprints = list()

        # Set FlowPrint parameters
        self.batch       = batch
        self.window      = window
        self.correlation = correlation
        self.similarity  = similarity

    ########################################################################
    #                          Fit-Predict method                          #
    ########################################################################

    def fit_predict(self, X, y=None):
        """Create fingerprints from given samples in X.

            Parameters
            ----------
            X : array-like of shape=(n_samples,)
                Samples (Flow objects) from which to generate fingerprints.

            y : array-like of shape=(n_samples,), optional
                Labels corresponding to X. If given, they will be encorporated
                into each fingerprint.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Resulting fingerprints.
            """
        ####################################################################
        #                        Setup fingerprints                        #
        ####################################################################

        # Empty fingerprints
        self.fingerprints = list()

        # Transform to arrays
        X = np.asarray(X)
        y = np.asarray(y) if y is not None else np.zeros(X.shape[0], dtype=int)

        # Initialise result
        result = (np.zeros(X.shape[0]) - 1).astype(object)

        ####################################################################
        #                       Divide into batches                        #
        ####################################################################

        # Sort X and y by time
        sort_time = np.argsort(X)
        sort_orig = np.argsort(sort_time)
        X = X[sort_time]
        y = y[sort_time]

        # Divide X and y in batches of size batch
        sort_batch = np.array([x.time_start() for x in X])
        # Compute number of required batches
        if sort_batch.shape[0]:
            batches = int(np.ceil((max(sort_batch) - min(sort_batch)) / self.batch))
        else:
            batches = 0
        # Get edges of batches
        _, edges = np.histogram(sort_batch, bins=max(1, batches))
        # Sort indices into batches
        batches = np.digitize(sort_batch, edges[:-1])

        ####################################################################
        #                Create fingerprints for each batch                #
        ####################################################################

        # Loop over each batch
        for batch in np.unique(batches):

            # Get data for given batch
            X_batch = X[batches == batch]
            y_batch = y[batches == batch]

            # Create fingerprints for single batch
            prediction = self._fit_single_batch_(X_batch, y_batch)
            result[batches == batch] = prediction

        ####################################################################
        #                    Merge similar fingerprints                    #
        ####################################################################

        # Merge similar fingerprints on all timeslots
        result = self.merge_fingerprints(result, self.similarity)
        # For unknown results assign nearest neighbour
        result = self.assign_nearest(X, result)

        # Return result
        return result[sort_orig]


    def _fit_single_batch_(self, X, y=None):
        """Create fingerprints for a given batch of flows.

            Parameters
            ----------
            X : array-like of shape=(n_samples_batch,)
                Samples (Flow objects) from which to generate fingerprints.

            y : array-like of shape=(n_samples_batch,), optional
                Labels corresponding to X. If given, they will be encorporated
                into each fingerprint.

            Returns
            -------
            np.array of shape=(n_samples,)
                Resulting fingerprints corresponding to each flow.
            """
        ####################################################################
        #                       Create fingerprints                        #
        ####################################################################

        # Create clustering instance
        cluster = Cluster()
        # Cluster flows into network destinations
        cluster.fit(X, y)

        # Find cliques in clusters
        cliques = CrossCorrelationGraph(
                    window      = self.window,     # Set window size
                    correlation = self.correlation # Set correlation threshold
                  ).fit_predict(cluster)           # Get cliques

        # Transform cliques to fingerprints
        fingerprints = list(
            Fingerprint(c)                 # Cast to fingerprint
            for c in cliques if len(c) > 1 # Only select cliques > 1
        )

        ####################################################################
        #                   Assign fingerprints per flow                   #
        ####################################################################

        # TODO

        # Predict cluster
        pred = cluster.predict(X)
        translation = cluster.cluster_dict()
        pred = [translation.get(p) for p in pred]

        # Assign each prediction to a fingerprint
        table = dict()
        for p in np.unique(pred):
            entry = [fp for fp in fingerprints if p in fp]
            if not entry:
                table[p] = Fingerprint()
            else:
                table[p] = max(entry, key=lambda x: len(x))

        # Get results
        prediction = np.array([table.get(p) for p in pred])
        # For unknown results assign nearest neighbour
        prediction = self.assign_nearest(X, prediction)
        # Merge similar fingerprints
        prediction = self.merge_fingerprints(prediction, self.similarity)

        # Append fingerprints to histsory of fingerprints
        self.fingerprints.append(fingerprints)

        # Return prediction
        return prediction

    ########################################################################
    #                          Auxiliary methods                           #
    ########################################################################

    def assign_nearest(self, X, y):
        """Set unassigned labels to that of nearest neighbours.

            Parameters
            ----------
            X : np.array of shape=(n_flows,)
                Array of original flows.

            y : np.array of shape=(n_flows,) and dtype=int
                Array of original labels. Unlabeled items should have label -1.

            Returns
            -------
            result : np.array of shape=(n_flows,) and dtype=int
                Array of new labels. Without any -1 labels.
            """
        # Get sortings
        sort_time = np.argsort(X)
        sort_orig = np.argsort(sort_time)

        # Sort by time
        X = X[sort_time]
        y = y[sort_time]

        # Get blocks of unassigned labels
        blocks = list()
        block  = list()
        for i, fingerprint in enumerate(y):
            if fingerprint and block:
                blocks.append(block)
                block = list()
            elif not fingerprint:
                block.append(i)
        if block:
            blocks.append(block)

        # For each block compute new labels
        for block in blocks:
            # Get left and right index of block
            left  = min(block) - 1
            right = max(block) + 1

            # Get left and right times + labels
            left_t  = X[left ].time_start() if left >= 0 else float('inf')
            left_y  = y[left ]              if left >= 0 else Fingerprint()
            right_t = X[right].time_start() if right < X.shape[0] else float('inf')
            right_y = y[right]              if right < X.shape[0] else Fingerprint()

            # Assign new labels per block
            for i in block:
                if abs(X[i].time_start()-left_t) < abs(X[i].time_start()-right_t):
                    y[i] = left_y
                else:
                    y[i] = right_y

        # Sort back to original
        y = y[sort_orig]

        # Return result
        return y

    def merge_fingerprints(self, fingerprints, threshold=1):
        """Merge fingerprints based on similarity.

            Parameters
            ----------
            fingerprints : list
                List of fingerprints to merge.

            Returns
            -------
            result : list
                Merged fingerprints
            """
        # Initialise result
        result = np.zeros(fingerprints.shape[0], dtype=fingerprints.dtype)

        # Retrieve unique fingerprints
        unique = set(fingerprints)

        # If threshold is 0 everything is equal
        if threshold == 0:
            # Create empty fingerprint
            fingerprint = Fingerprint()
            # Merge all fingerprints
            for fp in unique:
                fingerprint = fingerprint.merge(fp)
            # Set result to big fingerprint
            result = np.array([fingerprint for fp in fingerprints])

        # Check if partial matches should be merged, i.e. 0 < threshold < 1
        elif threshold < 1:
            # Initialise fingerprinting pairs to merge
            pairs = set()
            # Loop over fingerprinting pairs
            for fp1, fp2 in self.score_combinations(unique, threshold):
                # Check if comparison score is above threshold
                if fp1.compare(fp2) >= threshold:
                    # Add pairs
                    pairs.add((fp1, fp2))

            # Create mapping
            mapping = dict()
            # Loop over all fingerprints to be merged
            for fp1, fp2 in pairs:
                # Create merged fingerprint
                fp_merged = fp1.merge(fp2)
                # Check for double mapping fp1
                if fp1 in mapping:
                    fp_merged = fp_merged.merge(mapping.get(fp1))
                # Check for double mapping fp2
                if fp2 in mapping:
                    fp_merged = fp_merged.merge(mapping.get(fp2))

                mapping[fp1] = fp_merged
                mapping[fp2] = fp_merged

            # Apply mapping
            result = np.array([mapping.get(fp, fp) for fp in fingerprints])

        # Return new fingerprints
        return result


    def score_combinations(self, fingerprints, threshold):
        """Generator for combinations of fingerprints where can be > threshold.

            Parameters
            ----------
            fingerprints : iterable
                Iterable of fingerprints

            threshold : float
                Threshold that specifies whether fingerprints should match.

            Yields
            ------
            fingerprint1, fingerprint2 : Fingerprint
                Fingerprints which may have a similarity score >= threshold
            """
        # Get unique fingerprints
        fingerprints = set(fingerprints)

        # Only compare fingerprints of similar lengths
        lengths = dict()
        # Loop over all fingerprints
        for fp in fingerprints:
            # Get fingerprint length
            length = len(fp.destinations) + len(fp.certificates)
            # Set fingerprint lengths
            lengths[length] = lengths.get(length, set()) | set([fp])

        # Get possible combinations of lengths
        for length, fingerprints in sorted(lengths.items()):
            # Skip length 0, they cannot be equal
            if length == 0: continue
            # Initialise set of length combinations to explore
            length_comb = set()
            # Get length of possible next combination
            length2 = length
            # Check if score is possible
            while (length2-length) / max(length2, 1) <= (1-threshold):
                # Check if length is possible
                if length2 in lengths:
                    # If both are possible, add length as combination
                    length_comb.add(length2)
                # Increment next combination to check
                length2 += 1
            # Loop over all combinations of lengths
            for length2 in length_comb:
                # If lengths are equal, return combinations in self
                if length == length2:
                    yield from combinations(lengths.get(length), 2)
                # If lengths are not equal, return combinations between sets
                else:
                    a = lengths.get(length )
                    b = lengths.get(length2)
                    yield from ((x,y) for x in a for y in b)





    def map(self, fingerprints_test, fingerprints_train, verbose=False):
        """Map training fingerprints to testing fingerprints.

            Parameters
            ----------
            fingerprints_test : list
                List of fingerprints from which to map (keys).

            fingerprints_train : list
                List of fingerprints to which to map (values).

            verbose : boolean, default=False
                If set, prints progress

            Returns
            -------
            result : dict
                Dictionary of fingerprint_test -> closest_match
            """
        # Get unique fingerprints
        fingerprints_train = np.unique(fingerprints_train)
        fingerprints_test  = np.unique(fingerprints_test)

        # Create mappings
        mapping_train = dict()

        # Loop over fingerprints
        for fp in fingerprints_train:
            # Extract keys
            for key in fp.destinations | fp.certificates:
                # Add fingerprint to each key
                mapping_train[key] = mapping_train.get(key, set()) | set([fp])

        # Refine mapping to fp_test -> set([fps_train labels])
        mapping = dict()
        # Loop over all testing fingerprints
        for i, fp in enumerate(fingerprints_test):

            # Print progress if verbose
            if verbose:
                print("{}/{}".format(i+1, fingerprints_test.shape[0]), end='\r')

            # Initialise set
            matches = set()

            # Loop over all keys of fingerprint
            for key in fp.destinations | fp.certificates:
                # Get possible fingerprints
                matches |= mapping_train.get(key, set())

            # Initialise highest score
            highest_score = 0

            # Loop over all matches
            for match in matches:
                # Get score
                score = fp.compare(match)
                # If larger than highest score, replace match
                if score > highest_score:
                    mapping[fp] = match
                    highest_score = score

        # Return result
        return mapping

    def isin(self, fingerprints_test, fingerprints_train, similarity, verbose=False):
        """Check if testing fingerprints are in training fingerprints.

            Parameters
            ----------
            fingerprints_test : list
                List of fingerprints to check.

            fingerprints_train : list
                List of fingerprints to check against.

            similarity : float
                Minimum required similarity for mapping.

            verbose : boolean, default=False
                If set, prints progress

            Returns
            -------
            result : dict
                Dictionary of fingerprint_test -> True/False
            """
        # Get unique fingerprints
        fingerprints_train = np.unique(fingerprints_train)
        fingerprints_test  = np.unique(fingerprints_test)

        # Create mappings
        mapping_train = dict()

        # Loop over fingerprints
        for fp in fingerprints_train:
            # Extract keys
            for key in fp.destinations | fp.certificates:
                # Add fingerprint to each key
                mapping_train[key] = mapping_train.get(key, set()) | set([fp])

        # Refine mapping to fp_test -> True/False
        mapping = {fp: False for fp in fingerprints_test}
        # Loop over all testing fingerprints
        for i, fp in enumerate(fingerprints_test):

            # Print progress if verbose
            if verbose:
                print("{}/{}".format(i+1, fingerprints_test.shape[0]), end='\r')

            # Initialise set
            matches = set()

            # Loop over all keys of fingerprint
            for key in fp.destinations | fp.certificates:
                # Get possible fingerprints
                matches |= mapping_train.get(key, set())

            # Loop over all matches
            for match in matches:
                # Get score
                score = fp.compare(match)
                # If >= than similarity, set match to True
                if score >= similarity:
                    mapping[fp] = True
                    break

        # Return result
        return mapping
