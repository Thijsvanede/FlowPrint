from cluster import Cluster
from itertools import combinations
import networkx as nx
import numpy as np

class CrossCorrelationGraph3(object):

    def __init__(self, cluster=None, window=5, cc_threshold=0):
        """"""
        # Set cluster
        self.cluster = cluster or Cluster()

        # Set correlation parameters
        self.window       = window
        self.cc_threshold = cc_threshold
        self.cc           = dict()
        self.mapping      = dict()
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
        self.cc, self.mapping = self.cross_correlation(X)

        # In case of cc_threshold == 0
        if self.cc_threshold == 0:
            # Create a fully connected graph
            self.graph = nx.complete_graph(list(self.mapping.keys()))

        # In case of cc_threshold > 0
        else:
            # Create graph
            self.graph = nx.Graph()

            # Add nodes of graph
            self.graph.add_nodes_from(list(self.mapping.keys()))
            # Add edges of graph
            for (u, v), weight in self.cc.items():
                if weight >= self.cc_threshold:
                    self.graph.add_edge(u, v, weight=weight)

        # Return self for fit predict method
        return self

    def fit_cliques(self, X):
        """Fit Cross Correlation Graph and return cliques.

            Parameters
            ----------
            X : iterable of Flow
                Flows for which to create a cross-correlation graph.

            Returns
            -------
            result : Generator of cliques
                Generator of all cliques in the graph
            """
        # Get graph
        graph = self.fit(X).graph

        # Get cliques
        cliques = nx.find_cliques(graph)

        # Return generator over cliques
        return (set.union(*[self.mapping.get(n) for n in c]) for c in cliques)

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
        mapping = self.activity(X)

        # Compute cross correlation values
        for x, y in combinations(mapping, 2):
            # Compute cardinality of union
            union = len(x & y)
            # If a union exists add correlation
            if union:
                # Compute intersection
                intersection = len(x | y)
                # Add correlation
                result[x, y] = union / intersection

        # Return result
        return result, mapping

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

                    # Increase activity frame
                    i += 1

                    # Reset active clusters
                    active = set()

                # Add current cluster to active
                active.add(cluster)

        # Add final set if any
        for c in active:
            activity[c] = activity.get(c, set()) | set([i])

        # Get mapping of activities
        mapping = dict()
        # Loop over all flows
        for k, v in activity.items():
            mapping[frozenset(v)] = mapping.get(frozenset(v), set()) | set([k])

        # Return full activity
        return mapping
