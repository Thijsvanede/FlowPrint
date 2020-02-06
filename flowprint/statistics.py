from cluster import Cluster
from collections import Counter
from itertools import combinations
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import numpy as np

class Statistics(object):

    def __init__(self, X, y, random_state=42):
        """Statistics object containing analyses reports."""
        # Initialise performed analyses
        self.analyses = [None, None]

        # Set random_state
        self.random_state = check_random_state(random_state)

        # Initialise cluster
        self.cluster  = Cluster()
        self.X        = X
        self.y        = y
        self.cluster.fit(X, y)

    ########################################################################
    #                       Create statistics report                       #
    ########################################################################

    def report(self, kfold=10):
        """Create full statistics report over given kfolds.

            Returns
            -------
            result : self
                Returns self
            """

        # Run and set all analyses
        self.analyses = [
            ("Cluster Uniqueness: #applications per cluster"  , self.cluster_uniqueness()),
            ("Cluster Destinations: #destinations per cluster", self.cluster_destinations()),
            ("Cluster Certificates: #certificates per cluster", self.cluster_certificates()),
        ]

        # Returns self
        return self

    ########################################################################
    #                          Get random indices                          #
    ########################################################################

    def k_fold(self, k):
        """Generator: selects applications using k-fold random selection.

            Parameters
            ----------
            k : int
                Number of splits to make.

            Yields
            ------
            result : np.array
                The indices for each selection
            """

        # Get unique applications
        applications = np.unique(self.y)

        # Loop over all application splits
        for apps in np.array_split(applications, k):
            # Create mask for each split
            yield np.isin(self.y, apps)

    def monte_carlo(self, k, size):
        """Generator: selects applications using monte carlo random selection.

            Parameters
            ----------
            k : int
                Number of splits to make.

            size : int
                Size of each monte_carlo selection

            Yields
            ------
            result : dict of application -> np.array of indices
                Dictionary of applications -> indices for each selection.
            """
        # Get unique applications
        applications = np.unique(self.y)

        # Yield for each of k selections
        for _ in range(k):

            # Initialise result
            result = dict()
            # Randomly select applications
            apps = self.random_state.choice(applications, size=size, replace=False)
            # Loop over all apps
            for app in apps:
                # Set all applications
                result[app] = self.y == app

            # yield apps
            yield result

    def monte_carlos(self, k, apps=False, sizes=None):
        """Generator: Creates a monte carlo random selection for each size.

            Parameters
            ----------
            k : int
                Number of splits to make.

            size : list of int, optional
                If given use given sizes. Otherwise use self.test_sizes().

            Yields
            ------
            result : list (k) of np.array
                Single monte carlo selection for each size
            """
        # Loop over all sizes
        for size in sizes or self.test_sizes(apps):
            # For each size create a monte carlo randomness.
            yield self.monte_carlo(k, size)

    def test_sizes(self, apps=False):
        """Generator: gets list of test sizes from 1, 2, 5, 10, 20, 50, ..."""
        # Initialise i counter
        i = 1
        # Increase periodically
        while i < np.unique(self.y).shape[0] if apps else self.X.shape[0]:
            yield i
            if str(i)[0] == '1' or str(i)[0] == '5':
                i *= 2
            elif str(i)[0] == '2':
                i = int(i*5/2)

        # Add complete set as a result as well
        yield np.unique(self.y).shape[0] if apps else self.X.shape[0]

    ########################################################################
    #                          Cluster statistics                          #
    ########################################################################

    def cluster_uniqueness(self):
        """Computes number of labels per cluster.

            Returns
            -------
            result : Counter
                Counter of #labels in cluster -> #clusters.
            """
        # Compute uniquness per cluster and return
        return Counter([len(c.labels) for c in self.cluster.get_clusters()])

    def cluster_destinations(self):
        """Computes number of destinations per cluster.

            Returns
            -------
            result : Counter
                Counter of #destinations in cluster -> #clusters.
            """
        # Compute uniquness per cluster and return
        return Counter([len(c.destinations) for c in self.cluster.get_clusters()])

    def cluster_certificates(self):
        """Computes number of destinations per cluster.

            Returns
            -------
            result : Counter
                Counter of #certificates in cluster -> #clusters.
            """
        # Compute uniquness per cluster and return
        return Counter([len(c.certificates) for c in self.cluster.get_clusters()])

    ########################################################################
    #                        Application statistics                        #
    ########################################################################

    def application_clusters(self):
        """Get a counter of clusters per application.

            Returns
            -------
            result : dict
                Dictionary of application -> Counter of clusters
            """
        # Initialise result
        result = dict()

        # Get clusters for all applications
        for app in np.unique(self.y):
            result[app] = Counter(self.cluster.predict(self.X[self.y == app]))

        # Return result
        return result

    def application_unique_cluster(self):
        """Get unique cluster for each application.

            Returns
            -------
            result : dict
                Dictionary of application -> set of unique cluster
            """
        # Initialise result
        result = {application: set() for application in np.unique(self.y)}

        # Loop over all clusters
        for c in self.cluster.get_clusters():
            # Check if cluster is unique to application
            if len(c.labels) == 1:
                # Get application
                app = c.labels.most_common(1)[0][0]
                # Add unique cluster to application
                result[app] = result.get(app, set()) | set([c.identifier])

        # Return result
        return result

    def application_unique_cluster_set(self):
        """Get unique cluster set of each application.

            Returns
            -------
            result : dict
                Dictionary of application -> set of unique cluster combinations.
            """
        # Get cluster combinations
        preds = {app: set(self.cluster.predict(self.X[self.y == app])) for app in np.unique(y)}
        # Store applications which are NOT unique
        not_unique = set()

        # Loop over all combinations
        for a, b in combinations(preds, 2):
            # Compute union of two sets
            union = preds.get(a) & preds.get(b)
            # If a or b is equal to union, it is NOT unique
            if len(preds.get(a)) == len(union):
                not_unique.add(a)
            if len(preds.get(b)) == len(union):
                not_unique.add(b)

        # Return result
        return {x: set() if x in not_unique else y for x, y in preds.items()}

    def application_detection_time(self):
        """Computes minimum required time to discern application.
            An application can be discerned if it shows a combination of flow
            destinations which is unique to that application. The time until the
            first unique combination of flows is returned for each application.

            Returns
            -------
            result : dict
                Dictionary of application -> time before combination of
                application destinations of flows was unique. If float('inf'),
                the application could not be labeled as unique.
            """
        # Initialise result
        result = dict()

        # Get clusters per application
        clusters = {k: set(v.keys()) for k, v in
                    self.application_clusters().items()}

        # Loop over all applications:
        for application in np.unique(self.y):
            # Get all possible clusters
            possible = {k: v for k, v in clusters.items()}
            # Get flows of application
            data = sorted(self.X[self.y == application])
            # Get start time
            start = data[0].time_start()
            end = float('inf')

            # Loop over all flows
            for flow, cluster in zip(data, self.cluster.predict(data)):
                # Retrace possible clusters
                possible = {k: v for k, v in possible.items() if cluster in v}
                if len(possible) == 1:
                    end = flow.time_start()
                    break

            # Set total time
            result[application] = end - start

        # Return result
        return result

    def app_detection_time(self, k=10, r=(0, 300), plot=False, tikz=None):
        """Create plot of application detection times.

            Parameters
            ----------
            k : int, default=10
                Number of k-fold cross validations.

            r : tuple, default=(0, 300)
                Range for plotting in seconds.

            plot : boolean, default=False
                If True, plot application detection times.

            tikz : string, optional
                Path to output tikz.tex image. If given, outputs a tikz image
                with the resulting plot.

            Returns
            -------
            result : dict
                Dictionary of n_apps -> list of % detected per second
                for given range r.
            """
        # Get size -> k-fold -> indices
        mcs = self.monte_carlos(k, apps=True)
        # Get predictions
        predictions = self.cluster.predict(self.X)

        # Initialise result
        result = dict()

        # Loop over all size
        for kfold in mcs:
            # Initialise averages
            total  = [0 for t in range(*r)]
            n_folds = 0
            n_apps  = 0

            # Loop over all apps per kfold
            for apps in kfold:
                # Get sets of clusters for each application
                sets = {a: set(predictions[i]) for a, i in apps.items()}
                n_apps = len(apps)
                # Initialise detection times
                detection_times = dict()

                # Loop over apps
                for app, idx in apps.items():
                    # Get start time
                    start = min(self.X[idx]).time_start()
                    end   = float('inf')
                    # Get possible applications
                    possible = {a: s for a, s in sets.items()}
                    # Loop over data and predictions for each app
                    for data, pred in sorted(zip(self.X[idx], predictions[idx])):
                        # Add current to visited clusters
                        possible = {a: s for a, s in possible.items() if pred in s}
                        # Check if possible only contains a single app
                        if len(possible) == 1:
                            end = data.time_start()
                            break
                    # Set detection times
                    detection_times[app] = end - start

                # Increase number of apps
                n_folds += 1
                for k, v in enumerate(total):
                    total[k] += sum([t <= k for t in detection_times.values()])

            # Normalise for k_folds and number of applications
            total = [v/(n_folds*n_apps) for v in total]

            # Add total to result
            result[n_apps] = total

        # Plot if necessary
        if plot:
            for n_apps, coordinates in sorted(result.items()):
                # Plot line
                plt.plot(coordinates, label=n_apps)
            # Plot legend
            plt.legend()
            # Show plot
            plt.show()

        # If tikz, write output coordinates as tikz
        if tikz:
            # Initialise each plot
            plots = list()
            # Loop over all plots
            for n_apps, coordinates in sorted(result.items()):
                # Initialise individual plot
                plot = "    \\addplot[color=blue]\n    coordinates{\n"
                # Add all coordinates
                for x, y in enumerate(coordinates):
                    # Add coordinate
                    plot += "    ({}, {})\n".format(x, y)
                # Close plot
                plot += "    };\n"
                plot += "    \\addlegendentry{{{} applications}}".format(n_apps)
                # Append plot
                plots.append(plot)

            # Create complete picture with all plots
            string = """\\begin{{tikzpicture}}
    \\begin{{axis}}[
    xlabel={{seconds}},
    ylabel={{\\% of uniquely identifiable applications}}]
    {}
    \\end{{axis}}
\\end{{tikzpicture}}""".format("\n".join(plots))

            # Write to outfile
            with open(tikz, 'w') as outfile:
                outfile.write(string)

        # Return result
        return result



    ########################################################################
    #                             Plot methods                             #
    ########################################################################

    def plot_application_detection_time(self, maximum=float('inf'), n_apps=None):
        """Plot ratio of detected applications after time.

            Parameters
            ----------
            maximum : float, default=float('inf')
                Optional cap in seconds at maximum time.

            n_apps : int, optional
                Number of applications to take into account.
                If given, take a random subsample of n_apps applications.

            average : int, optional
                If given, average over given runs.
            """
        # Compute application detection time
        adt = self.application_detection_time()

        # Check if we should select subsamples
        if n_apps is not None:
            # Select a given number of applications
            if n_apps > len(adt):
                print("Warning, could only select {} applications."
                      .format(len(adt)))
            # Randomly select n_apps applications
            apps = self.random_state.choice(list(adt.keys()),
                                            size=min(n_apps, len(adt)),
                                            replace=False)
            # Only keep selected applications
            adt = {k:v for k, v in adt.items() if k in apps}

        # Get timestamps for which to plot
        times = [x for x in adt.values() if x <= maximum and x != float('inf')]
        times = np.array(list(sorted(set(times))))

        res = np.zeros(times.shape[0])
        for i, t in enumerate(times):
            res[i] = len([k for k, v in adt.items() if v <= t]) / len(adt)

        plt.plot(times, res)
        plt.show()




    ########################################################################
    #                             Print report                             #
    ########################################################################

    def __str__(self):
        """Returns string representation of report."""
        result  = "Analysis Report\n"
        result += "--------------------------------------------------\n"
        result += "\n"

        for analysis, output in self.analyses:
            result += "  {}\n".format(analysis)
            result += "  ------------------------------------------------\n"
            if isinstance(output, Counter):
                for k, v in sorted(output.items()):
                    result += "    {:4}: {:>4}x\n".format(k, v)
            elif isinstance(output, dict):
                for k, v in sorted(output.items()):
                    result += "    {} : {}\n".format(k, sorted(v.items(), key=lambda x: -x[1]))
            else:
                result += "    non-counter"
            result += "\n"

        # Return result
        return result
