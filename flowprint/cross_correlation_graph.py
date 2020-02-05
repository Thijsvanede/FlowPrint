from itertools import combinations
import networkx as nx

class CrossCorrelationGraph(object):

    def __init__(self, window=30, correlation=0.1):
        """CrossCorrelationGraph for computing correlation between clusters.

            IMPORTANT: The self.graph object is an optimised graph. Each node
            does not represent a network destination, but represents an activity
            fingerprint. E.g. when destinations A and B are both only active at
            time slices 3 and 7, then these destinations are represented by a
            single node. We use the self.mapping to extract the network
            destinations from each graph node.
                This is a huge optimisation for finding cliques as the number of
            different network destinations theoretically covers the entire IP
            space, whereas the number of activity fingerprints is bounded by
            2^(batch / window), in our work 2^(300/30) = 2^10 = 1024. If these
            parameters change, the complexity may increase, but never beyond the
            original bounds. Hence, this optimisation never has a worse time
            complexity.

            Parameters
            ----------
            window : float, default=30
                Threshold for the window size in seconds.

            correlation : float, default=0.1
                Threshold for the minimum required correlation
            """
        # Set parameters
        self.window       = window
        self.correlation  = correlation
        self.mapping      = dict()
        self.graph        = nx.Graph()

    def fit(self, cluster, y=None):
        """Fit Cross Correlation Graph.

            Parameters
            ----------
            cluster : Cluster
                Cluster to fit graph, cluster must be populated with flows

            y : ignored

            Returns
            -------
            result : self
                Returns self
            """
        # Compute cross correlations within cluster
        correlations, self.mapping = self.cross_correlation(cluster)

        # In case of correlation <= 0
        if self.correlation <= 0:
            # Create a fully connected graph
            self.graph = nx.complete_graph(list(self.mapping.keys()))

        # In case of correlation > 0
        else:
            # Create graph
            self.graph = nx.Graph()

            # Add nodes of graph
            self.graph.add_nodes_from(list(self.mapping.keys()))
            # Add edges of graph
            for (u, v), weight in correlations.items():
                if weight >= self.correlation:
                    self.graph.add_edge(u, v, weight=weight)

        # Return self for fit predict method
        return self

    def predict(self, X=None, y=None):
        """Fit Cross Correlation Graph and return cliques.

            Parameters
            ----------
            X : ignored

            y : ignored

            Returns
            -------
            result : Generator of cliques
                Generator of all cliques in the graph
            """
        # Get cliques
        cliques = nx.find_cliques(self.graph)
        # Return generator over cliques
        return (set.union(*[self.mapping.get(n) for n in c]) for c in cliques)

    def fit_predict(self, X, y=None):
        """Fit cross correlation graph with clusters from X and return cliques.

            Parameters
            ----------
            X : array-like of shape=(n_samples,)
                Flows for which to create a cross-correlation graph.

            y : ignored

            Returns
            -------
            result : Generator of cliques
                Generator of all cliques in the graph
            """
        # Perform fit and predict
        return self.fit(X).predict(X)

    ########################################################################
    #                      Compute cross correlation                       #
    ########################################################################

    def cross_correlation(self, cluster):
        """Compute cross correlation between clusters

            Parameters
            ----------
            cluster : Cluster
                Cluster to fit graph, cluster must be populated with flows

            Returns
            -------
            correlation : dict
                Dictionary of cross correlation values between each
                NetworkDestination inside cluster.

            mapping : dict
                Mapping of activity fingerprint -> clusters
            """
        # Initialise correlation
        correlation = dict()

        # Get activity of samples
        activity = self.activity(cluster)
        # Get inverted mapping
        mapping = dict()
        for destination, active in activity.items():
            mapping[frozenset(active)] =\
                mapping.get(frozenset(active), set()) | set([destination])

        # Compute cross correlation values
        for x, y in combinations(mapping, 2):
            # Compute cardinality of union
            union = len(x & y)
            # If a union exists add correlation
            if union:
                # Compute intersection
                intersection = len(x | y)
                # Add correlation
                correlation[x, y] = union / intersection

        # Return result
        return correlation, mapping

    def activity(self, cluster):
        """Extracts sets of active clusters by time.

            Parameters
            ----------
            cluster : Cluster
                Cluster to fit graph, cluster must be populated with flows

            Returns
            -------
            mapping : dict
                Dictionary of NetworkDestination -> activity
            """
        # Get samples
        X = cluster.samples
        # Compute start time
        start = min(x.time_start() for x in X)

        # Initialise mapping of NetworkDestination -> activity
        mapping = dict()

        # Loop over all network destinations
        for destination in cluster.clusters():
            # Loop over each flow in destination
            for flow in destination.samples:
                # Compute activity per flow
                activity = set()
                # Loop over all timestamps
                for timestamp in flow.timestamps:
                    # Compute activity for each timestamp
                    activity.add(int((timestamp - start) // self.window))

                # Add activity to mapping
                mapping[destination] = mapping.get(destination, set()) | activity

        # Return activity mapping
        return mapping
