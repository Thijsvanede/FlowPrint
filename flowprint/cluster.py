from collections import Counter
import itertools
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

################################################################################
#         Cluster object for clustering flows per network destination          #
################################################################################

class Cluster(object):

    def __init__(self, load=None):
        """Cluster flows by network destinations

            Parameters
            ----------
            load : string, default=None
                If given, load cluster from json file from 'load' path.
            """
        # Set samples
        self.samples = np.zeros((0))

        # Initialise counter
        self.counter = 0

        # Dictionaries of destination identifiers -> cluster
        self.dict_destination = dict()
        self.dict_certificate = dict()

        # Load cluster if necessary
        if load is not None:
            self.load(load)

    ########################################################################
    #                       Fit & prediction methods                       #
    ########################################################################

    def fit(self, X, y=None):
        """Fit the clustering algorithm with flow samples X.

            Parameters
            ----------
            X : array-like of shape=(n_samples, n_features)
                Flow samples to fit cluster object.

            y : array-like of shape=(n_samples,), optional
                If given, add labels to each cluster.

            Returns
            -------
            result : self
                Returns self
            """
        # Add X to samples
        self.samples = np.concatenate((self.samples, X))
        # Set y to empty if None
        y = np.zeros(len(X)) if y is None else y

        # Loop over all samples in X
        for sample, label in zip(X, y):

            # Extract values
            certificate = sample.certificate()
            destination = sample.destination()

            # Get the number of matching clusters
            clusters = [self.dict_certificate.get(certificate),
                        self.dict_destination.get(destination)]

            # Case 1: Multiple matches
            # Check for multiple matching slices
            if all(c is not None for c in clusters):

                # Case 1a: Destination and certificate -> same cluster
                if all(c == clusters[0] for c in clusters):
                    # Set the cluster to that one cluster
                    cluster = clusters[0]

                # Case 1b: Destination and certificate -> different clusters
                else:
                    # Create new cluster
                    cluster = self.new_cluster()
                    # For each match
                    for c in clusters:
                        # Add samples from old cluster to new cluster
                        cluster.merge(c)
                        # Reset dictionaries to point to new cluster
                        for value in c.certificates:
                            if value is not None:
                                self.dict_certificate[value] = cluster
                        for value in c.destinations:
                            if value is not None:
                                self.dict_destination[value] = cluster

            # Case 2: Single or no matches
            else:
                # Check for matching cluster or create a new one
                cluster = [c for c in clusters if c] or [self.new_cluster()]
                # Get matching cluster
                cluster = cluster[0]

            # Add datapoint to cluster
            cluster.add(sample, label)
            # Point dictionaries to new cluster
            if certificate is not None:
                self.dict_certificate[certificate] = cluster
            if destination is not None:
                self.dict_destination[destination] = cluster

        # Return result
        return self

    def predict(self, X):
        """Predict cluster labels of X.

            Parameters
            ----------
            X : array-like of shape=(n_samples, n_features)
                Samples for which to predict NetworkDestination cluster.

            Returns
            -------
            result : array-like of shape=(n_samples,)
                Labels of NetworkDestination cluster corresponding to cluster of
                fitted samples. Has a value of -1 if no cluster could be matched
            """
        # Predict each item and return
        return np.asarray([self.predict_single(x) for x in X])

    def predict_single(self, X):
        """Predict single flow X

            Parameters
            ----------
            X : Flow
                Flow sample for which to retrieve NetworkDestination cluster.

            Returns
            -------
            result : int
                Label of NetworkDestination cluster corresponding to flow or -1
                if no cluster could be matched.
            """
        # Get matching cluster or -1
        return self.dict_destination.get(X.destination(),
               self.dict_certificate.get(X.certificate(),
               NetworkDestination(-1))).identifier

    def fit_predict(self, X):
        """Fit and predict cluster with given samples.

            Parameters
            ----------
            X : array-like of shape=(n_samples, n_features)
                Samples to fit cluster object.

            Returns
            -------
            result : array-like of shape=(n_samples,)
                Labels of cluster corresponding to cluster of fitted
                samples. Has a value of -1 if no cluster could be matched.
            """
        return self.fit(X).predict(X)

    ########################################################################
    #                          Auxiliarry methods                          #
    ########################################################################

    def new_cluster(self):
        """Creates and returns new NetworkDestination cluster.

            Returns
            -------
            result : NetworkDestination
                New unique NetworkDestination for given cluster.
            """
        # Increment number of clusters
        self.counter += 1
        # Create cluster and return
        return NetworkDestination(self.counter - 1)

    def clusters(self):
        """Return a set of NetworkDestinations in the current cluster object.

            Returns
            -------
            result : set
                Set of NetworkDestinations in cluster.
            """
        clusters  = set(self.dict_certificate.values())
        clusters |= set(self.dict_destination.values())
        return clusters

    def cluster_dict(self):
        """Return a dictionary of id -> NetworkDestination.

            Returns
            -------
            result : dict
                Dict of NetworkDestination.identifier -> NetworkDestination
            """
        return {c.identifier: c for c in self.clusters()}

    ########################################################################
    #                        Import/export methods                         #
    ########################################################################

    def save(self, outfile):
        """Saves cluster object to json file.

            Parameters
            ----------
            outfile : string
                Path to json file in which to store the cluster object.
            """
        # Create output
        output = {"samples": self.samples.tolist()}

        # Write to json file
        with open(outfile, 'w') as outfile:
            json.dump(output, outfile)


    def load(self, infile):
        """Loads cluster object from json file.

            Parameters
            ----------
            infile : string
                Path to json file from which to load the cluster object.
            """
        with open(infile, 'r') as infile:
            result  = json.load(infile)
            samples = np.asarray(result.get("samples"))
            self.fit(samples)

    def copy(self):
        """Returns a (semi-deep) copy of self.
            The resulting cluster is a deep copy apart from the samples X.
            Has a tremendous speedup compared to copy.deepcopy(self)

            Returns
            -------
            result : Cluster
                Copy of self
            """
        # Initialise result
        result = Cluster()
        # Fit result with given samples
        result.fit(self.samples)
        # Return result
        return result


    ########################################################################
    #                             Plot methods                             #
    ########################################################################

    def plot(self, annotate=False):
        """Plot cluster NetworkDestinations.

            Parameters
            ----------
            annotate : boolean, default=False
                If True, annotate each cluster
            """
        # Get clusters
        clusters = [c.get_description()    for c in self.clusters()]
        sizes    = [20*len(c.samples)**0.7 for c in self.clusters()]

        # Create complete graph from clusters
        graph = nx.Graph()
        graph.add_nodes_from(clusters)
        graph.add_edges_from(itertools.combinations(clusters, 2))

        # Draw graph
        nx.draw_spring(graph,
            alpha=0.7,              # see through nodes
            edgelist    = list(),   # don't show edges
            node_size   = sizes,    # set node sizes
            with_labels = annotate, # don't show labels
        )

        # Plot graph
        plt.show()


    ########################################################################
    #                            String method                             #
    ########################################################################

    def __str__(self):
        """Returns string representation of self."""
        # Get all clusters as a set
        clusters = self.clusters()

        # Get predictions
        preds = self.predict(self.samples)

        # Return string
        return """Cluster
---------------------------------
  Flow samples                : {:>}
  Unique Network Destinations : {:>}
  Unique labels               : {:>}
  -------------------------------
  Unique certificates         : {}
  Unique ip destinations      : {}""".format(
    self.samples.shape[0],
    len(clusters),
    np.unique(preds[preds != -1]).shape[0],
    len(self.dict_certificate),
    len(self.dict_destination))





################################################################################
#       A NetworkDestination clustering flows into a single destination        #
################################################################################

class NetworkDestination(object):

    def __init__(self, identifier, samples=[]):
        """NetworkDestination object for flow samples.

            Parameters
            ----------
            identifier : object
                Identifier for NetworkDestination
                Important: identifier must be unique!

            samples : iterable of Flow
                Samples to store in this NetworkDestination.
            """
        # Initialise variables
        self.identifier   = identifier
        self.samples      = []
        self.destinations = set()
        self.certificates = set()
        self.labels       = Counter()

        # Add each datapoint
        for X in samples:
            self.add(X)

    ########################################################################
    #                         Add flows to cluster                         #
    ########################################################################

    def add(self, X, y=None):
        """Add flow X to NetworkDestination object.

            Parameters
            ----------
            X : Flow
                Datapoint to store in this NetworkDestination.

            y : object
                Label for datapoint
            """
        # Add datapoint
        self.samples.append(X)
        self.labels.update([y])
        # Update pointers
        self.destinations.add(X.destination())
        self.certificates |= X.certificates


    def merge(self, other):
        """Merge NetworkDestination with other NetworkDestination object.

            Parameters
            ----------
            other : NetworkDestination
                Other NetworkDestination object to merge with.
            """
        # Only merge in case other is NetworkDestination object
        if isinstance(other, NetworkDestination):
            # Merge two NetworkDestinations
            self.samples.extend(other.samples)
            # Merge pointers
            self.destinations |= other.destinations
            self.certificates |= other.certificates
            self.labels += other.labels

    ########################################################################
    #                           Get description                            #
    ########################################################################

    def get_description(self):
        """Returns human readable description of cluster"""
        # Get descriptions
        descr_cert = [X.certificate() for X in self.samples]
        descr_ip   = ["{}".format(X.destination()) for X in self.samples]
        # Remove None values
        descr_cert = [x for x in descr_cert if x is not None]
        descr_ip   = [x for x in descr_ip   if x is not None]
        # Get most common
        descr_cert = Counter(descr_cert).most_common(1)
        descr_ip   = Counter(descr_ip  ).most_common(1)
        # Return description
        try   : return descr_cert[0][0]
        except: return descr_ip[0][0]

    ########################################################################
    #                           Object overrides                           #
    ########################################################################

    def __str__(self):
        """Returns string presentation of self."""
        return "NetworkDestination [{:4}] [size={:4}] [IPs={}] [labels={}]".\
                format(self.identifier, len(self.samples),
                list(sorted(self.destinations)), self.labels)
