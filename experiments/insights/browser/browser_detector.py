from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

from os import path
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../src')))
from cluster_optimised import ClusterOptimised

class BrowserDetector(object):

    def __init__(self, columns, cluster=None, random_state=42):
        """Object for identifying browser traffic in network flows.

            Parameters
            ----------
            columns : dict
                Dictionary of column name -> column index.

            cluster : ClusterOptimised, optional
                Clustering method to use.
                If None is given, use the default clustering method, i.e.
                TLS certificate + destination with min_samples=5 and
                none_eq=False

            random_state : int, RandomState instance or None, optional, default:
                None If int, random_state is the seed used by the random number
                generator; If RandomState instance, random_state is the random
                number generator; If None, the random number generator is the
                RandomState instance used by `np.random`.
            """
        # Set columns
        self.column  = columns
        # Initialise Cluster for browser detector
        if cluster is None:
            self.cluster = ClusterOptimised(min_samples=5, none_eq=False,
                   slices=[[self.column["TLS certificate issuer name"],
                            self.column["TLS certificate signature"],
                            self.column["TLS certificate subject name"],
                            self.column["TLS certificate subject pk"],
                            self.column["TLS certificate subject pk algorithm"],
                            self.column["TLS certificate version"]],

                           [self.column['IP dst'],
                            self.column['TCP dport']]])
        else:
            self.cluster = cluster
        # Initialise classifier
        self.classifier = RandomForestClassifier(n_estimators=10,
                                                 random_state=random_state)


    def fit(self, X, y):
        """Fit the classifier with browser and non-browser traffic.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Features to fit the classifier with. Should be output of
                self.features().

            y : np.array of shape=(n_samples,)
                Array of labels, 0 for non-browser, 1 for browser.

            Returns
            -------
            result : self
                Returns self for fit_predict method.
            """
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        """Predict whether samples from X are browser: 1 or non_browser: 0.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Features to predict with the classifier. Should be output of
                self.features().

            Returns
            -------
            result : np.array of shape=(n_samples,)
                0 if sample from X is not from browser, 1 if sample
                from X is from browser.
            """
        return self.classifier.predict(X)

    def fit_predict(self, X, y):
        """Fit and predict the samples with the classifier as browser
            or non-browser traffic.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Features to fit the classifier with. Should be output of
                self.features().

            y : np.array of shape=(n_samples,)
                Array of labels, 0 for non-browser, 1 for browser.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                0 if sample from X is not from browser, 1 if sample
                from X is from browser.
            """
        return self.classifier.fit(X, y).predict(X)

    def features(self, flows):
        """Returns flow features for determining whether flows are browser.

            Parameters
            ----------
            flows : np.array of shape=(n_samples, n_features)
                Flows to analyse.

            Returns
            -------
            result : np.array of shape=(n_samples, n_features)
                Features for determining browser flows.
                Currently the features are [clusters', length incoming',
                length outgoing'] where the ' indicates the derivative.
            """
        # Get prediction according to clustering mechanism
        prediction = self.cluster.fit_predict(flows)

        # Create dataframe
        df = {c: flows[:, i] for c, i in self.column.items()}
        # Add prediction
        df["prediction"] = prediction
        # Create as pandas dataframe
        df = pd.DataFrame(df, index=pd.to_datetime(df["timestamp"], unit='s'))
        # Add an index to the flows and sort
        df["index"] = np.arange(df.shape[0])
        df = df.sort_index()

        # Add a column for total byte length of incoming and outgoing flows
        df['length incoming'] = df['TCP length incoming mean']*\
                                df['IP packets incoming']
        df['length outgoing'] = df['TCP length outgoing mean']*\
                                df['IP packets outgoing']

        # Compute metrics of unique clusters
        clusters = df['prediction'].rolling('5s').apply(
                   lambda x: np.unique(x).shape[0], raw=True)
        lengthsi = df['length incoming'].rolling('5s').apply(np.mean, raw=True)
        lengthso = df['length outgoing'].rolling('5s').apply(np.mean, raw=True)

        # Compute derivative of unique cluster metric
        clusterd = clusters.rolling('5s').apply(lambda x: x[-1]-x[0], raw=True)
        lengthdi = lengthsi.rolling('5s').apply(lambda x: x[-1]-x[0], raw=True)
        lengthdo = lengthso.rolling('5s').apply(lambda x: x[-1]-x[0], raw=True)
        fraction = lengthdi/lengthdo

        # Imput NaN values to 0
        clusterd[np.isnan(clusterd)] = 0
        lengthdi[np.isnan(lengthdi)] = 0
        lengthdo[np.isnan(lengthdo)] = 0
        fraction[np.isnan(fraction)] = 0
        fraction[fraction ==  float('inf')] = 0
        fraction[fraction == -float('inf')] = 0

        # Create an array
        X = np.vstack((clusterd.values, lengthdi.values, lengthdo.values, fraction.values)).T

        # Return array as originally indexed
        return X[df["index"]]

    def is_browser(self, timestamps, flows,
                         before=pd.Timedelta(seconds=10),
                          after=pd.Timedelta(seconds=10)):
        """For each flow determines whether it belongs to a browser or not.

            Parameters
            ----------
            flows : np.array of shape=(n_samples, n_features)
                Flows to analyse.

            before : pd.Timedelta, default=pd.Timedelta(seconds=5)
                Time before browser flow was detected that should be included
                as originating from a browser.

            after : pd.Timedelta, default=pd.Timedelta(seconds=60)
                Time after browser flow was detected that should be included
                as originating from a browser.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Boolean array that is True where flows are detected as browser.
            """
        # Retrieve timestamps
        timestamps = pd.to_datetime(timestamps, unit='s')

        # Apply browser detection
        browser = self.predict(flows).astype(bool)

        # Return all flows with a given spread in time
        return self.spread_period(timestamps, browser, before, after)

    def spread_period(self, timestamps, browser,
                            before=pd.Timedelta(seconds=10),
                             after=pd.Timedelta(seconds=60)):
        """Return the indices of timestamps in browser timeframe.

            Parameters
            ----------
            timestamps : np.array of shape=(n_samples,)
                Timestamps of flows.

            browser : np.array of shabe=(n_samples,)
                Boolean array indicating whether corresponding timestamp is of
                a browser (True) or of a non-browser (False).

            before : pd.Timedelta, default=pd.Timedelta(seconds=10)
                Time before browser flow was detected that should be included
                as originating from a browser.

            after : pd.Timedelta, default=pd.Timedelta(seconds=60)
                Time after browser flow was detected that should be included
                as originating from a browser.
            """
        # Initialise result as no browser found at all
        result = np.zeros(browser.shape[0], dtype=bool)

        # Loop over all flows with browser detection
        for timestamp in timestamps[browser]:
            # Set timeframe for each browser flow detected
            start = timestamp - before
            end   = timestamp + after
            result[np.logical_and(timestamps >= start,
                                  timestamps <= end   )] = True
        # Return result
        return result

    def plot(self, timestamps, flows,
                         before=pd.Timedelta(seconds=10),
                          after=pd.Timedelta(seconds=60)):
        """Create a plot for activity showing when something is marked as
            a browser.

            Parameters
            ----------
            flows : np.array of shape=(n_samples, n_features)
                Flows to analyse.

            before : pd.Timedelta, default=pd.Timedelta(seconds=10)
                Time before browser flow was detected that should be included
                as originating from a browser.

            after : pd.Timedelta, default=pd.Timedelta(seconds=60)
                Time after browser flow was detected that should be included
                as originating from a browser.
            """
        # Retrieve timestamps
        timestamps = pd.to_datetime(timestamps, unit='s')

        # Extract flow features
        clusterd   = flows[:, 0]
        lengthdi   = flows[:, 1]
        lengthdo   = flows[:, 2]

        # Apply browser detection
        browser = self.predict(flows).astype(bool)

        # Get filter for browser
        df = pd.DataFrame({'clusterd': clusterd,
                           'lengthdi': lengthdi,
                           'lengthdo': lengthdo},
                           index=timestamps).sort_index()

        if browser.sum() > 0:
            plt.scatter(df.loc[browser].index, np.zeros(browser.sum()), c='g')
        for point in df.loc[browser].index:
            plt.plot([point - before, point + after], [0, 0], c='g')
        plt.plot(df['clusterd'], alpha=0.5)
        plt.show()


    def __str__(self):
        """Return string representation of BrowserDetector."""
        return "BrowserDetector(id=0x{:x})".format(id(self))
