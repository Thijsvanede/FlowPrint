from collections import Counter
import numpy as np

class KNN(object):

    def __init__(self, k):
        """Fast KNN algorithm for jaccard distance."""
        # Set k
        self.k = k
        # Set initial samples
        self.samples = np.zeros(0)

    def fit(self, X, y=None):
        """Fit samples to model."""
        # Set samples
        self.samples = np.array([frozenset(x) for x in X])
        # Get mapping from element to sample
        self.mapping = dict()

        # Loop over all sets
        for index, s in enumerate(self.samples):
            # Loop over all items per set
            for item in s:
                # Get value for mapping
                value = set([(s, index)])
                # Add value to mapping
                self.mapping[item] = self.mapping.get(item, set()) | value

    def closest(self, X, similarity, verbose=False):
        """Get closest neighbour for all X."""
        # Get initial indices
        indices = np.zeros(X.shape[0], dtype=int) - 1
        result  = np.zeros(X.shape[0], dtype=object)

        # For each X, get closest element
        for i, x in enumerate(X):
            if verbose: print("{}/{}".format(i+1, X.shape[0]), end='\r')
            # Get all matches
            matches = Counter()
            # Loop over elements in x
            for element in x:
                # Check if element in mapping
                if element in self.mapping:
                    # Get all corresponding matches
                    matches.update(self.mapping[element])

            # Initialise closest match
            highest =  0
            b_match = -1

            # Loop over all matches
            for (match, index), count in matches.items():
                # Get score
                score = count / (len(match) + len(x) - count)
                # Check if score is highest
                if score > highest and score > similarity:
                    # Set new highscore
                    highest = score
                    b_match = index

            # Set index
            indices[i] = b_match

        # Get matches
        for index, value in enumerate(indices):
            result[index] = self.samples[value] if value != -1 else None

        # Return result
        return result

    def kneighbors(self, X, k=None, verbose=False):
        """Get kneighbors for X."""
        # Set k
        k = k or self.k
        # Get X as frozensets
        X = np.array([frozenset(x) for x in X])

        # Initialise results
        distances = np.zeros((X.shape[0], k), dtype=float)
        neighbors = np.zeros((X.shape[0], k), dtype=int  )

        # Loop over all elements
        for pos, element in enumerate(X):
            # If verbose print progress
            if verbose: print("{}/{}".format(pos+1, X.shape[0]), end='\r')
            # Get single elements
            distance, neighbor = self._kneighbors_(element, pos, k)
            # Set single element
            distances[pos] = distance
            neighbors[pos] = neighbor

        # Return results
        return distances, neighbors

    def _kneighbors_(self, X, index, k):
        """Get kneighbors for single element."""
        # Initialise possible matches
        matches = Counter()

        # Get possible matches, i.e. matches > 0
        for x in X:
            if x in self.mapping: matches.update(self.mapping[x])

        # Remove element X from mapping
        if (X, index) in matches: del matches[(X, index)]

        # Get matches as tuples
        matches = [(m, i, c) for (m, i), c in matches.items()]

        # If not enough matches found, fill with ones
        if len(matches) < k:
            matches += [(set(), i, 0) for i in range(k - len(matches))]

        # Compute score
        counts  = np.array([c      for m, i, c in matches])
        lengths = np.array([len(m) for m, i, c in matches])
        indices = np.array([i      for m, i, c in matches])
        scores  = 1 - (counts / (lengths - counts + len(X)))

        # Get k indices
        indices_k = np.argpartition(scores, k-1)[:k]

        # Return result
        return scores[indices_k], indices[indices_k]
