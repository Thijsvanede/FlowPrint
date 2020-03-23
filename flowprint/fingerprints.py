from itertools               import combinations
import numpy as np

try:
    from .cluster import Cluster
    from .cross_correlation_graph import CrossCorrelationGraph
    from .fingerprint import Fingerprint
except:
    try:
        from cluster import Cluster
        from cross_correlation_graph import CrossCorrelationGraph
        from fingerprint import Fingerprint
    except Exception as e:
        raise ValueError(e)

class FingerprintGenerator(object):
    """Generator of FlowPrint Fingerprint objects from flows

        Attributes
        ----------
        batch : float
            Threshold for the batch size in seconds

        window : float
            Threshold for the window size in seconds

        correlation : float
            Threshold for the minimum required correlation

        similarity : float
            Threshold for the minimum required similarity
    """

    def __init__(self, batch=300, window=30, correlation=0.1, similarity=0.9):
        """Generate FlowPrint Fingerprint objects from flows

            Parameters
            ----------
            batch : float, default=300
                Threshold for the batch size in seconds

            window : float, default=30
                Threshold for the window size in seconds

            correlation : float, default=0.1
                Threshold for the minimum required correlation

            similarity : float, default=0.9
                Threshold for the minimum required similarity
            """
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
        sort_batch = np.array([x.time_start for x in X])
        # Compute number of required batches
        if sort_batch.shape[0]:
            batches = int(
                np.ceil((max(sort_batch) - min(sort_batch)) / self.batch)
            )
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

        # Get network destination per flow
        destinations = cluster.predict(X)     # Get destination id per flow
        translation  = cluster.cluster_dict() # Get destinations for each id
        destinations = [translation.get(d) for d in destinations]

        # Get fingerprint per network destination
        mapping_fingerprints = dict()
        # Map destination to largest fingerprint by (#destinations, #flows)
        for fingerprint in sorted(fingerprints):
            for destination in fingerprint:
                mapping_fingerprints[destination] = fingerprint

        # Apply mapping
        prediction = np.array([
            mapping_fingerprints.get(x.destination,
            mapping_fingerprints.get(x.certificate,
            Fingerprint())) for x in X
        ])

        ####################################################################
        #             Handle unknown and similar fingerprints              #
        ####################################################################

        # For unknown results assign nearest neighbour
        prediction = self.assign_nearest(X, prediction)
        # Merge similar fingerprints
        prediction = self.merge_fingerprints(prediction, self.similarity)

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
                Array of fingerprints.

            Returns
            -------
            result : np.array of shape=(n_flows,) and dtype=int
                Array of Fingerprints. Without any -1 labels.
            """
        ####################################################################
        #             Sort flows and fingerprints by timestamp             #
        ####################################################################

        # Sort flows by time
        sort_time = np.argsort(X)
        sort_orig = np.argsort(sort_time)

        # Sort by time
        X = X[sort_time]
        y = y[sort_time]
        # Get timestamps
        timestamps = np.asarray([x.time_start for x in X])

        ####################################################################
        #               Assign closest fingerprints in time                #
        ####################################################################

        # Get blocks of unassigned fingerprint indices
        blocks = list()
        block  = list()
        for i, fingerprint in enumerate(y):
            if fingerprint and block:
                blocks.append(np.asarray(block))
                block = list()
            elif not fingerprint:
                block.append(i)
        if block:
            blocks.append(np.asarray(block))

        # For each block of unassigned fingerprints compute new labels
        for block in blocks:
            # Get indices before and after block
            before = min(block) - 1
            after  = max(block) + 1
            # Get timestamps before and after block
            ts_before = X[before].time_start if before >= 0          else float('inf')
            ts_after  = X[after ].time_start if after  <  X.shape[0] else float('inf')
            # Get fingerprints before and after block
            fp_before = y[before] if before >= 0          else Fingerprint()
            fp_after  = y[after ] if after  <  X.shape[0] else Fingerprint()

            # Assign new fingerprints per block
            block_before = abs(timestamps[block] - ts_before) <\
                           abs(timestamps[block] - ts_after )
            y[block[ block_before]] = fp_before
            y[block[~block_before]] = fp_after

        # Return fingerprints in original order
        return y[sort_orig]


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
        ####################################################################
        #           Case default: all fingerprints are different           #
        ####################################################################
        result = np.asarray(fingerprints)

        # Retrieve unique fingerprints
        unique = sorted(set(fingerprints))

        ####################################################################
        #                Case 1: all fingerprints are equal                #
        ####################################################################
        if threshold <= 0:
            # Create one big merged fingerprint out of all unique fingerprints
            result[:] = Fingerprint(set().union(*unique))

        ####################################################################
        #         Case 2: Merge fingerprints by 0 < threshold < 1          #
        ####################################################################
        elif threshold < 1:
            # Initialise fingerprinting pairs to merge
            pairs = set([
                # Define pairs
                (fp1, fp2)
                # For each combination of pairs
                for fp1, fp2 in self.score_combinations(unique, threshold)
                # Where similarity >= threshold
                if fp1.compare(fp2) >= threshold
            ])

            # Create mapping of original fingerprint -> merged fingerprint
            mapping = dict()
            # Loop over all fingerprints to be merged
            for fp1, fp2 in pairs:
                # Create merged fingerprint
                fp_merged = mapping.get(fp1, fp1).merge(
                            mapping.get(fp2, fp2))
                # Set mappings
                mapping[fp1] = fp_merged
                mapping[fp2] = fp_merged

            # Apply mapping
            result = np.array([mapping.get(fp, fp) for fp in fingerprints])

        ####################################################################
        #                    Return merged fingerprints                    #
        ####################################################################
        return result


    def score_combinations(self, fingerprints, threshold):
        """Generator for combinations of fingerprints where can be > threshold.

            IMPORTANT: This method is purely for efficiency purposes. It is
            based on the observation that fingerprints of size N require at
            least M equal destinations to reach the threshold. Alternatively
            `itertools.combinations(fingerprints, 2)`` may be used.

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
        fingerprints = sorted(set(fingerprints))

        # Only compare fingerprints of similar lengths
        lengths = dict()
        # Loop over all fingerprints
        for fp in fingerprints:
            # Get fingerprint length
            length = len(fp)
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
            for key in fp:
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
            for key in fp:
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
            for key in fp:
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
            for key in fp:
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
