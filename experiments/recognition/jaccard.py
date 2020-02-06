import multiprocessing
import numpy as np
from collections import Counter
from tcluster.distances import precompute, jaccard

def closest(sets, items, verbose=False):
    """Find mapping from items to known sets.

        Parameters
        ----------
        sets : iterable of set
            Known sets, i.e. values of resulting mapping.

        items : iterable of set
            Unknown sets, i.e. values of resulting mapping.

        verbose : boolean, default=False
            If True, prints progress

        Returns
        -------
        result : dict of set -> set
            Dictionary of unknown items -> known set.
        """
    # Get unique
    sets  = np.unique(sets)
    items = np.unique(items)

    # Transform sets to dictionary
    sets_dict = dict()
    # Loop over all sets
    for s in sets:
        # Loop over all elements in set
        for elem in s:
            # Update dictionary
            sets_dict[elem] = sets_dict.get(elem, set()) | set([frozenset(s)])

    # Create mapping
    mapping = dict()

    # Loop over all items
    for pos, item in enumerate(items):
        # Print if verbose
        if verbose: print("{}/{}".format(pos, items.shape[0]), end='\r')
        # Create mapping
        mapping[item] = closest_single(sets_dict, item)

    # Return result
    return mapping

def closest_single(sets, item):
    """Test given item (set) against known sets.

        Parameters
        ----------
        sets : dict of object -> set
            Dictionary for each object to sets containing object.

        element : set
            Set to compare against known sets.

        Returns
        -------
        result : set
            Most closely matched set.
        """
    # Initialise possible matches
    matches = Counter()

    # Get possible matches, i.e. matches > 0
    for x in item:
        if x in sets:
            matches.update(sets[x])

    # If no matches found, return None
    if not matches: return None

    # Get matches as tuples
    matches = [(m, c) for m, c in matches.items()]
    # Compute score
    counts  = np.array([c      for m, c in matches])
    lengths = np.array([len(m) for m, c in matches])
    scores  = counts / (lengths - counts + len(item))
    # Return maximum score
    return matches[np.argmax(scores)][0]


if __name__ == "__main__":
    a = np.arange(1000)

    # Create random state
    rs = np.random.RandomState(42)


    for i in range(1000):
        train = np.array([frozenset(rs.choice(a, rs.randint(50, 200))) for _ in range(10)])
        test  = np.array([frozenset(rs.choice(a, rs.randint(50, 200))) for _ in range(10)])

        mapping  = closest (train, test)
        # mapping2 = closest2(train, test)
        # scores  = np.array([0 if v           is None else 1-jaccard(k, v          ) for k, v in sorted(mapping.items())])
        # scores2 = np.array([0 if mapping2[k] is None else 1-jaccard(k, mapping2[k]) for k, v in sorted(mapping.items())])
        #
        # if not np.all(scores == scores2):
        #     print("Error found at {}:".format(i))
        #
        #     for k, v in mapping.items():
        #         print("  {} --> {} ({:.4f})/{} ({:.4f})".format(
        #                         np.argwhere(test  == k)[0][0],
        #                         np.argwhere(train == v)[0][0],
        #                         1 - jaccard(k, v),
        #                         np.argwhere(train == mapping2[k])[0][0],
        #                         1 - jaccard(k, mapping2[k])
        #                         ))
        #
        #     exit()

    print("Completed!")
