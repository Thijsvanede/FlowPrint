from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

try:
    from .cluster import Cluster
except:
    try:
        from cluster import Cluster
    except Exception as e:
        raise ValueError(e)


class BrowserDetector(object):
    """Detector for browser application

        Attributes
        ----------
        classifier : sklearn.ensemble.RandomForestClassifier
            Random forest classifier used for classifying individual datapoints

        before : float
            Time frame in seconds to remove before detected browser

        after : float
            Time frame in seconds to remove after detected browser
    """

    def __init__(self, before=10, after=10, random_state=42):
        """Detector for browser application

            Parameters
            ----------
            before : float, default = 10
                Time frame in seconds to remove before detected browser

            after : float, default = 10
                Time frame in seconds to remove after detected browser

            random_state : int, RandomState instance or None, optional, default:
                None If int, random_state is the seed used by the random number
                generator; If RandomState instance, random_state is the random
                number generator; If None, the random number generator is the
                RandomState instance used by `np.random`
            """
        # Initialise classifier
        self.classifier = RandomForestClassifier(
            n_estimators=10, random_state=random_state
        )
        # Initialise before and after seconds
        self.before = before
        self.after  = after

    ########################################################################
    #                         Fit/Predict methods                          #
    ########################################################################

    def fit(self, X, y):
        """Fit the classifier with browser and non-browser traffic

            Parameters
            ----------
            X : array-like of shape=(n_samples, n_features)
                Flows to fit the classifier with

            y : array-like of shape=(n_samples,)
                Array of labels, -1 for non-browser, 1 for browser

            Returns
            -------
            result : self
                Returns self for fit_predict method
            """
        # Fit classifier
        self.classifier.fit(self.features(X), y.astype(int))
        # Return self
        return self

    def predict(self, X, y=None):
        """Predict whether samples from X are browser: 1 or non_browser: -1

            Parameters
            ----------
            X : array-like of shape=(n_samples, n_features)
                Flows to predict with the classifier

            y : ignored

            Returns
            -------
            result : np.array of shape=(n_samples,)
                -1 if sample from X is not from browser, 1 if sample from X is
                from browser
            """
        # Get prediction from Random Forest
        predictions = self.classifier.predict(self.features(X))

        ################################################################
        #       Label temporally close flows as browser as well        #
        ################################################################

        # Get timestamps from flows
        timestamps  = np.asarray([x.time_start for x in X])

        # Loop over all predictions
        for i, prediction in enumerate(predictions):
            # Check if we found a browser
            if prediction == 1:
                # Get timestamp
                ts = timestamps[i]
                # Set previous and future timestamps
                for j in range(i, 0, -1):
                    predictions[j] = max(predictions[j], 0)
                    if timestamps[j] < ts-self.before: break
                for j in range(i, timestamps.shape[0]):
                    predictions[j] = max(predictions[j], 0)
                    if timestamps[j] > ts+self.after: break

        # Set detected by timeframe to -1
        predictions[predictions == 0] = -1

        # Return result
        return predictions

    def fit_predict(self, X, y):
        """Fit and predict the samples with the classifier as browser
            or non-browser traffic

            Parameters
            ----------
            X : array-like of shape=(n_samples, n_features)
                Flows to fit the classifier with

            y : array-like of shape=(n_samples,)
                Array of labels, -1 for non-browser, 1 for browser

            Returns
            -------
            result : np.array of shape=(n_samples,)
                -1 if sample from X is not from browser, 1 if sample from X is
                from browser
            """
        return self.fit(X, y).predict(X)



    ########################################################################
    #                          Feature extraction                          #
    ########################################################################

    def features(self, X):
        """Returns flow features for determining whether flows are browser

            Parameters
            ----------
            X : array-like of shape=(n_samples, n_features)
                Flows from which to extract features

            Returns
            -------
            result : np.array of shape=(n_samples, n_features)
                Features for determining browser flows.
                Currently the features are [clusters', length incoming',
                length outgoing', ratio incoming/outgoing']
                where the ' indicates the derivative
            """
        print("Creating features")
        # Compute dataframe of flow features
        df = pd.DataFrame({
            'outgoing' : [ sum(y for y in x.lengths if y > 0) for x in X],
            'incoming' : [-sum(y for y in x.lengths if y < 0) for x in X],
            'cluster'  : Cluster().fit_predict(X)
        }, index=pd.to_datetime([x.time_start for x in X], unit='s')
        ).sort_index()

        # Compute features as rolling changes
        cluster  = df['cluster' ].rolling('5s').apply(
            lambda x: np.unique(x).shape[0], raw=True).values
        incoming = df['incoming'].rolling('5s').apply(np.mean, raw=True).values
        outgoing = df['outgoing'].rolling('5s').apply(np.mean, raw=True).values

        # Compute derivatives
        cluster  = np.concatenate(([0], np.diff(cluster)))
        incoming = np.concatenate(([0], np.diff(incoming)))
        outgoing = np.concatenate(([0], np.diff(outgoing)))
        ratio    = incoming/outgoing

        # To numpy array
        result = np.asarray([cluster, incoming, outgoing, ratio]).T
        # Impute NaN
        result[np.isnan(result)       ] = 0
        result[result ==  float('inf')] = 0
        result[result == -float('inf')] = 0

        # Return result
        return result
