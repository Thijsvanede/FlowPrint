from cluster import Cluster
from itertools import combinations
import networkx as nx
import numpy as np

class CrossCorrelationGraph(object):

    def __init__(self, cluster=None, window=5, cc_threshold=0):
        """"""
        # Set cluster
        self.cluster = cluster or Cluster()

        # Set correlation parameters
        self.window       = window
        self.cc_threshold = cc_threshold
        self.cc           = dict()
        self.graph        = nx.Graph()

    def fit(self, X):
        """Fit Cross Correlation Graph.

            Parameters
            ----------
            X : iterable of Flow
                Flows for which to create a cross-correlation graph.

            Returns
            -------
            result : self
                Returns self
            """
        # Fit cluster and return self
        pred = self.cluster.predict(X)
        # Get predictions as Clusters
        clusters = self.cluster.get_cluster_dict()
        pred = [clusters.get(x) for x in pred]

        # Compute cross correlation
        self.cc = self.cross_correlation(X)

        # In case of cc_threshold == 0
        if self.cc_threshold == 0:
            # Create a fully connected graph
            self.graph = nx.complete_graph(np.unique(pred))
            # Set all weights
            for u, v, d in self.graph.edges(data=True):
                d['weight'] = 1

        # In case of cc_threshold > 0
        else:
            # Create graph
            self.graph = nx.Graph()

            # Add nodes of graph
            self.graph.add_nodes_from(np.unique(pred))
            # Add edges of graph
            for (u, v), weight in self.cc.items():
                if weight >= self.cc_threshold:
                    self.graph.add_edge(u, v, weight=weight)

        # Return self for fit predict method
        return self

    def fit_graph(self, X):
        """Fit Cross Correlation Graph and return graph.

            Parameters
            ----------
            X : iterable of Flow
                Flows for which to create a cross-correlation graph.

            Returns
            -------
            result : networkx.Graph
                Weighted graph of cross-correlations.
            """
        return self.fit(X).graph

    ########################################################################
    #                      Compute cross correlation                       #
    ########################################################################

    def cross_correlation(self, X):
        """Compute cross correlation between clusters

            Parameters
            ----------
            X : iterable of Flow
                Flows from which to compute activity.

            Returns
            -------
            result : dict
                Dictionary of cross correlation values.
            """
        # Initialise result
        result = dict()

        # Get activity of samples
        activity, tuples = self.activity(X)

        # Compute cross correlation values
        for x, y in tuples:
            # Compute cardinality of union
            union = len(activity.get(x) & activity.get(y))
            # If a union exists add correlation
            if union:
                # Compute intersection
                intersection = len(activity.get(x) | activity.get(y))
                # Add correlation
                result[x, y] = union / intersection

        # Return result
        return result

    def activity(self, X):
        """Extracts sets of active clusters by time.

            Parameters
            ----------
            X : iterable of Flow
                Flows from which to compute activity.

            Returns
            -------
            activity : list of set
                Sets of simultaniously active clusters.

            combinations : set of tuple
                Sets of combinations that must be computed.
            """
        # Get clusters
        clusters     = self.cluster.predict(X)
        cluster_dict = self.cluster.get_cluster_dict()
        clusters     = [cluster_dict.get(c) for c in clusters]

        # Get all flows with timestamp in each window
        samples = list()

        # Loop over all clusters
        for flow, cluster in zip(X, clusters):
            # Get timestamps from flow
            timestamps = list(sorted(flow.timestamps))
            # Get first timestamp
            start = timestamps[0]

            # Initialise first entry and last entry to samples
            entries = [(start, flow, cluster)]

            # Add one timestamp for every window
            for timestamp in timestamps:
                # Check new window
                if timestamp > start + self.window:
                    # Add new entry
                    entries.append((timestamp, flow, cluster))
                    # Reset start
                    start = timestamp

            # Add entries to samples
            samples.extend(entries)

        # Sort by timestamp
        samples = list(sorted(samples))

        # Create activity sets of clusters
        activity = dict()
        tuples   = set()

        # Set start time
        start  = samples[0][0]
        active = set()
        i      = 0

        # Loop over all entries
        for ts, flow, cluster in samples:
            # Skip anomalies
            if cluster.get_id() != -1:
                # In case of next timeframe dump active set
                if ts > start + self.window:
                    # Reset last timestamp
                    start = ts

                    # Add activity to all existing clusters
                    for c in active:
                        activity[c] = activity.get(c, set()) | set([i])
                    # Add activity combinations to combinations
                    for x, y in combinations(active, 2):
                        tuples.add((min(x, y), max(x, y)))

                    # Increase activity frame
                    i += 1

                    # Reset active clusters
                    active = set()

                # Add current cluster to active
                active.add(cluster)

        # Add final set if any
        for c in active:
            activity[c] = activity.get(c, set()) | set([i])

        # Return full activity
        return activity, tuples
