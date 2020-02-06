from collections import Counter
from sklearn.utils import check_random_state
import csv
import json
import numpy as np

from cryptography.x509 import oid

# TODO
import matplotlib.pyplot as plt
import matplotlib as mpl

class _Cluster_(object):

    def __init__(self, identifier, min_samples, samples=[]):
        """Cluster object for samples.

            Parameters
            ----------
            identifier : object
                Identifier for cluster

            min_samples : int
                Minimum number of samples to not be an anomaly

            samples : iterable of Flow
                Samples to store in this cluster.
            """
        # Initialise variables
        self.identifier  = identifier
        self.min_samples = min_samples
        self.samples  = []
        self.destinations = set()
        self.certificates = set()
        self.labels = Counter()

        # Add each datapoint
        for X in samples:
            self.add(X)

    def add(self, X, y=None):
        """Add datapoint X to cluster object.

            Parameters
            ----------
            X : Flow
                Datapoint to store in this cluster.

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
        """Merge cluster with other cluster object.

            Parameters
            ----------
            other : _Cluster_
                Other cluster object to merge with.
            """
        # Only merge in case other is _Cluster_ object
        if isinstance(other, _Cluster_):
            # Merge two clusters
            self.samples.extend(other.samples)
            # Merge pointers
            self.destinations |= other.destinations
            self.certificates |= other.certificates
            self.labels += other.labels

    def match(self, other):
        """Return whether clusters match."""
        return len(self.destinations & other.destinations) or\
               len(self.certificates & other.certificates)

    def get_id(self):
        """Returns identifier of cluster or -1 if anomalous,
           i.e. number of samples do not exceed min_samples."""
        if len(self.samples) >= self.min_samples:
            return self.identifier
        else:
            return -1

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
    #                             I/O methods                              #
    ########################################################################

    def show(self):
        """Prints verbose string representation of self."""
        print("""
  Cluster {} containing {} samples
  --------------------------------------
      Certificates :
          - {}
      Destinations :
          - {}
""".format(str(self.identifier),
           len(self.samples),
           "\n        - ".join([str(v) for v in self.certificates]),
           "\n        - ".join([str(v) for v in self.destinations])))

    def copy(self):
        """Returns a (semi-deep) copy of self.
            The resulting _Cluster_ is a deep copy apart from the samples X.
            Has a tremendous speedup compared to copy.deepcopy(self)

            Returns
            -------
            result : Cluster
                Copy of self.
            """
        # Initialise result
        result = _Cluster_(self.identifier, self.min_samples)
        # Fit result with given samples
        result.destinations = set([d for d in self.destinations])
        result.certificates = set([c for c in self.certificates])
        result.labels       = Counter() + self.labels
        # Return result
        return result

    ########################################################################
    #                           Object overrides                           #
    ########################################################################

    def __str__(self):
        """Returns string presentation of self."""
        return "Cluster [{:4}] [size={:4}] [IPs={}] [labels={}]".format(
                self.identifier, len(self.samples), self.destinations,
                self.labels)

    def __gt__(self, other):
        """Greater than object override"""
        return self.identifier >  other.identifier

    def __ge__(self, other):
        """Greater equals object override"""
        return self.identifier >= other.identifier

    def __lt__(self, other):
        """Less than object override"""
        return self.identifier <  other.identifier

    def __le__(self, other):
        """Less equals object override"""
        return self.identifier <= other.identifier

    def __eq__(self, other):
        """Equals object override"""
        return self.identifier == other.identifier

    def __hash__(self):
        """Hash cluster object"""
        return hash(self.identifier)


class Cluster(object):

    def __init__(self, min_samples=1, load=None):
        """Optimised version of Cluster. Only works for descrete_or distances
           but is linear in time complexity instead of quadratic.

            Parameters
            ----------
            min_samples : int, default=1
                Minimum number of samples before concidering a cluster.
                Otherwise will be classified as anomalous.

            load : string, default=None
                If given, load cluster from json file from 'load' path.
            """
        # Initialise DBSCAN object features
        self.min_samples = min_samples

        # Set samples
        self.samples = np.zeros((0))

        # Initialise counter
        self.counter = 0

        # For each slice: Dictionary of known slice -> cluster
        self.dict_destination = dict()
        self.dict_certificate = dict()

        if load is not None:
            self.load(load)

    ########################################################################
    #                       Fit & prediction methods                       #
    ########################################################################

    def fit(self, X, y=None):
        """Fit the clustering algorithm with samples X.

            Parameters
            ----------
            X : array-like of shape=(n_samples, n_features)
                Samples to fit cluster object.

            progress : boolean, default=False
                If true, shows progress of clustering

            Returns
            -------
            result : self
                Returns self as object
            """
        # Add X to samples
        self.samples = np.concatenate((self.samples, X))
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

                # Case 1a: All cases point to the same cluster
                if all(c == clusters[0] for c in clusters):
                    # Set the cluster to that one cluster
                    cluster = clusters[0]

                # Case 1b: Some cases point to a different cluster
                else:
                    # Create new cluster
                    cluster = self.new_cluster()
                    # For each slice
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
                Samples to predict with cluster object.

            Returns
            -------
            result : array-like of shape=(n_samples,)
                Labels of cluster corresponding to cluster of fitted
                samples. Has a value of -1 if no cluster could be matched.
            """
        # Predict each item and return
        return np.array([self.predict_single(x) for x in X])

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
        """Creates a new cluster and returns it"""
        result = _Cluster_(self.counter, self.min_samples)
        self.counter += 1
        return result

    def get_clusters(self):
        """Returns a set of clusters in the current cluster object."""
        clusters  = set(self.dict_certificate.values())
        clusters |= set(self.dict_destination.values())
        return clusters

    def get_cluster_dict(self):
        """Return a dictionary of id -> cluster."""
        return {c.identifier: c for c in self.get_clusters()}

    def get_cluster_info(self):
        """Return a dictionary with cluster info."""
        return {c.identifier: c.get_description() for c in self.get_clusters()}

    def predict_single(self, X):
        """Get prediction of datapoint X for given slice i."""
        # Get matching cluster destination
        cluster = self.dict_destination.get(X.destination())

        # If found, return
        if cluster is not None:
            return cluster.get_id()

        # Get matching cluster certificate
        cluster = self.dict_certificate.get(X.certificate())

        # Return cluster id or -1 if no match found
        return -1 if cluster is None else cluster.get_id()

    def get_cluster_single(self, X):
        """Get prediction of datapoint X for given slice i."""
        # Get matching cluster
        result = self.dict_certificate.get(X.certificate(),
                 self.dict_destination.get(X.destination()))
        # Return cluster identifier or -1 if no match found
        return -1 if result is None else result.identifier

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
        output = {"min_samples": self.min_samples,
                  "samples"    : self.samples.tolist()}

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
            result = json.load(infile)
            self.min_samples = result.get("min_samples")
            samples          = np.asarray(result.get("samples"))
            self.fit(samples)

    def copy(self):
        """Returns a (semi-deep) copy of self.
            The resulting cluster is a deep copy apart from the samples X.
            Has a tremendous speedup compared to copy.deepcopy(self)

            Returns
            -------
            result : Cluster
                Copy of self.
            """
        # Initialise result
        result = Cluster(self.min_samples)
        # Fit result with given samples
        result.fit(self.samples)
        # Return result
        return result


    ########################################################################
    #                             Plot method                              #
    ########################################################################

    def plot(self, y=None, random_state=42, annotate=False):
        """ Plot the current cluster and show labels according to different
            colours. Note that plotted features do not represent units due
            to the handling of discrete features.

            Parameters
            ----------
            y : np.array of shape=(n_samples,), optional
                Labels to assign to plot clusters.

            random_state : int, RandomState instance or None, optional, default:
                None If int, random_state is the seed used by the random number
                generator; If RandomState instance, random_state is the random
                number generator; If None, the random number generator is the
                RandomState instance used by `np.random`.

            annotate : boolean, default=False
                If True, annotate clusters with certificate / IP.
            """
        # Set random state
        rs = check_random_state(random_state)

        # Get clusters
        pred = np.array([self.get_cluster_single(x) for x in self.samples])
        # Create anomaly mask
        anomaly = self.predict(self.samples) == -1
        # Set norm to ensure colors are displayed correctly
        norm = mpl.colors.Normalize(vmin=pred.min(), vmax=pred.max())

        # Create cluster centres
        centres  = generate_random_points(np.unique(pred).shape[0], 10, rs)
        # Create point coordinates
        points   = rs.normal(size=(self.samples.shape[0], 2))
        centroid = dict()
        for i, p in enumerate(np.unique(pred)):
            # Add centre to point
            points[pred == p] += centres[i]

        # If y is given, annotate clusters with labels
        if y is not None:
            for i, p in enumerate(np.unique(pred)):
                centroid[tuple(centres[i])] = set(np.unique(y[pred ==p]))

        # If annotate is true, annotate clusters
        elif annotate:
            for i, cluster in enumerate(sorted(self.get_clusters(), key=lambda x: x.identifier)):
                try:
                    centroid[tuple(centres[i])] = cluster.get_description()
                except:
                    print("Warning: Could not annotate {}".format(cluster))

        # Set colour
        colour = 'black' if y is not None else pred[~anomaly]

        # Scatter clustered points in plot
        plt.scatter(points[~anomaly][:, 0], points[~anomaly][:, 1],
                    s=20, norm=norm, cmap="gist_rainbow",
                    c=colour)

        # Scatter anomalous points in plot
        plt.scatter(points[anomaly][:, 0], points[anomaly][:, 1],
                    s=5, c='black')

        # Annotate is needed
        for coordinates, txt in centroid.items():
            plt.annotate(str(txt), (coordinates[0], coordinates[1]))

        # Plot cluster
        plt.show()


    def plot_clusters(self, random_state=42, annotate=False, annotate_cap=1,
                  colors=None):
        """ Plot the current cluster and show labels according to different
            colours and sizes. Note that plotted features do not represent
            units due to the handling of discrete features.

            Parameters
            ----------
            random_state : int, RandomState instance or None, optional, default:
                None If int, random_state is the seed used by the random number
                generator; If RandomState instance, random_state is the random
                number generator; If None, the random number generator is the
                RandomState instance used by `np.random`.

            annotate : boolean, default=False
                If True, annotate clusters with certificate / IP.

            annotate_cap : int, default=1
                Only annotate values with an occurence >= annotate_cap.

            colors : dict(), default=None
                If given, color the clusters with the given color
                Dictionary should be of cluster -> color
            """
        # Set random state
        rs = check_random_state(random_state)

        # Get clusters
        clusters = list(sorted(self.get_clusters()))

        # Create cluster centres
        centres = generate_random_points(len(clusters), 10, rs)

        # Plot each centre according to the number of points
        sizes = 5*np.asarray([len(cluster.samples) for cluster in clusters])

        # Set colour
        colour = 'black'
        if colors is not None:
            colour = np.asarray([colors.get(cluster, 'black') for cluster in clusters])

        # Create point coordinates
        centroid = dict()

        # Get a list of all clusters
        all_clusters = {c.identifier: c for c in clusters}

        # If annotate is true, annotate clusters
        if annotate:
            for i, cluster in enumerate(clusters):
                # Set description
                description = "Unknown"
                # Try to extract description from IP
                destination = Counter([c.destination() for c in cluster.samples if c.destination() != None])
                if destination:
                    description = destination.most_common(1)[0][0]

                # Try to extract description from certificate
                certificates = Counter([c.certificate() for c in cluster.samples if c.certificate() != None])
                if certificates:
                    # Get subject from certificate
                    description = certificates.most_common(1)[0][0].subject
                    # Find subject common name
                    description = description.get_attributes_for_oid(
                                    oid.NameOID.COMMON_NAME)[0].value

                # Try to extract description from domains
                domains = Counter([c.domain for c in cluster.samples if c.domain != None])
                if domains:
                    description = domains.most_common(1)[0][0]

                if sizes[i] >= annotate_cap:
                    centroid[tuple(centres[i])] = description

        # Scatter all points
        for col in sorted(np.unique(colour), key=lambda x: {'yellow': 0, 'black': 1, 'blue': 2, 'red': 3, 'green': 4, 'cyan': 5}.get(x)):
            if col == 'cyan': continue
            # Get mask for each colour
            mask = colour == col
            # Get label for each colour
            label = {'yellow': "unknown", 'black': "first", 'blue': "CDN", 'red': "ads", 'green': "social", 'cyan': "DNS"}.get(col)
            # Plot each colour individually
            plt.scatter(centres[mask][:, 0], centres[mask][:, 1],
                        s=sizes[mask], alpha=0.5, c=col, label=label)

        # Annotate is needed
        for coordinates, txt in centroid.items():
            plt.annotate(str(txt), (coordinates[0], coordinates[1]))

        # Create legend
        if colors is not None:
            # Plot legend
            legend = plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
            # Set all legends to same size
            for handle in legend.legendHandles:
                handle._sizes = [50]

        # Plot cluster
        plt.show()

    ########################################################################
    #                       Object method overrides                        #
    ########################################################################

    def __str__(self):
        """Returns string representation of self."""
        # Get all clusters as a set
        clusters = self.get_clusters()

        # Get predictions
        preds = self.predict(self.samples)

        # Return string
        return """
  OptimisedCluster
  ---------------------------------
    Samples             : {:>}
    Unique clusters     : {:>}
    Unique labels       : {:>}
    Number of anomalies : {:>}

    Unique slices
    -------------------------------
    Unique certificates : {}
    Unique destinations : {}
""".format(self.samples.shape[0],
           len(clusters),
           np.unique(preds[preds != -1]).shape[0],
           np.sum(preds == -1),
           len(self.dict_certificate),
           len(self.dict_destination))

    ########################################################################
    #                           Ancillary methods                          #
    ########################################################################

def generate_random_points(n, min_distance, random_state):
    """Generate points with a minimum distance between eachother

        Parameters
        ----------
        n : int
            Number of points to generate.

        min_distance : float
            Minimum distance between generated points.

        random_state : int, RandomState instance or None, optional, default:
            None If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by `np.random`.
        """
    rs = check_random_state(random_state)
    # Compute required grid distance
    x = y = int((2*n)**0.55+3) * min_distance

    # Create first point
    points = rs.rand(1, 2) * [x, y]

    # Generate new points with minimum distance
    while points.shape[0] < n:
        # Generate random point
        point = rs.rand(1, 2) * [x, y]

        # Get distance of point to all other points
        distances = np.array([np.sqrt(np.sum((point - p)**2)) for p in points])

        # If the smallest distance is large enough
        if min(distances) >= min_distance:
            # Add point to list of points
            points = np.vstack((points, point))

    # Return result
    return points
