from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import check_random_state
import numpy as np

class Splitter(object):

    def __init__(self, random_state=42):
        """Splitter object for splitting flows.

            Parameters
            ----------
            random_state : int, default=42
                Random state for reproducable splitting.
            """
        self.rs = check_random_state(random_state)

    def split_labels(self, X, y, size_known=1/2):
        """Split labels into known and unknown.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Samples on which to split.

            y : np.array of shape=(n_samples,)
                Labels on which to split.

            size_known : float, default=1/2
                Proportion of samples to return as known_test indices.

            Return
            ------
            X_known : np.array of shape=(n_samples, n_features)
                Known data

            y_known : np.array of shape=(n_samples,)
                Known labels

            X_unknown: np.array of shape=(n_samples, n_features)
                Known undata

            y_unknown: np.array of shape=(n_samples,)
                Unknown labels
            """
        # Get data as numpy array
        y = np.asarray(y)
        X = np.asarray(X)

        # Randomly select known and unknown labels
        unknown, known = train_test_split(np.unique(labels),
                                          test_size=size_known,
                                          random_state=self.rs)

        # Return known and unknown labels
        return X[np.isin(y,   known)], y[np.isin(y,   known)],\
               X[np.isin(y, unknown)], y[np.isin(y, unknown)]

    def k_fold_label(self, X, y, k, rs=42, max_labels=None):
        """Generator: splits the data into different k-folds according to label.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Samples on which to split.

            y : np.array of shape=(n_samples,)
                Labels on which to split.

            k : int
                Number of splits to make.

            rs : int, default=42
                Random state

            max_labels : int, optional
                If given, return a maximum number of labels per k-fold.

            Yields
            ------
            X : np.array of shape=(n_samples,)
                Samples in given k-fold.

            y : np.array of shape=(n_samples,)
                Labels in given k-fold.
            """
        # Get random state
        rs = check_random_state(rs)

        # Get unique labels
        labels = np.unique(y)
        # Split into k-folds
        for train, test in self.kfold_safe(k, labels, random_state=rs):
            # get labels for k-fold
            labels_kfold = labels[train]
            # In case of max_labels, randomly select maximum number of labels.
            if max_labels is not None and max_labels <= labels_kfold.shape[0]:
                # Randomly select max_labels of labels
                labels_kfold = rs.choice(labels_kfold, size=max_labels, replace=False)

            # Get mask for given k-fold
            mask = np.isin(y, labels_kfold)
            # Yield data
            yield X[mask], y[mask]


    def kfold_safe(self, k, data, random_state):
        """A safe method for requesting K-folds, also covers k=1.

            Parameters
            ----------
            k : int
                Number of folds.

            data : np.array of shape=(n_samples, n_features)
                Data on which to perform kfold split.

            random_state : int, default=42
                Random state.

            Yields
            ------
            train : np.array of shape=(n_samples, n_features)
                Data """
        if k < 2:
            yield np.arange(data.shape[0]), np.zeros(0)
        else:
            for train, test in KFold(k, random_state=random_state).split(data):
                yield train, test


    def split(self, X, y, size_known=1/2, size_train=1/2):
        """ Get indices of train, known_test and unknown_test data.
            I.e. labels in known_test may appear in train data, whereas
            labels in unknown_test will not appear in train data. This
            implementation splits the data on a temporal level, i.e. all
            training instances occur before testing instances.

            Note that ratios are approximate as the numer of labels has
            influence on the final result.

            Parameters
            ----------
            X : np.array of shape=(n_samples, n_features)
                Samples on which to split.

            y : np.array of shape=(n_samples,)
                Labels on which to split.

            size_known : float, default=1/2
                Proportion of samples to return as known_test indices.

            size_unknown : float, default=1/4
                Proportion of samples to return as unknown_test indices.

            Returns
            -------
            indices_train : np.array
                Indices of samples and labels in training set.

            indices_known : np.array
                Indices of samples and labels in known testing set.

            indices_unknown : np.array
                Indices of samples and labels in unknown testing set.
            """
        # Get data as numpy array
        y = np.asarray(y)
        X = np.asarray(X)

        # Initialise results
        X_train     , y_train      = list(), list()
        X_test_known, y_test_known = list(), list()

        # Get unique labels
        labels = np.unique(y)
        # Split labels in known and unknown
        labels_unknown, labels_known = train_test_split(labels,
                                            test_size=size_known,
                                            random_state=self.rs)

        ####################################################################
        #                       Set unknown dataset                        #
        ####################################################################
        # Compute mask for unknown data
        mask_test_unknown = np.isin(y, labels_unknown)
        # Set unknown sets
        X_test_unknown = X[mask_test_unknown]
        y_test_unknown = y[mask_test_unknown]

        ####################################################################
        #                        Set known dataset                         #
        ####################################################################
        # Compute masks for known & unknown data
        mask_known   = np.isin(y, labels_known)

        # Split known data in train and test sets
        timestamps = np.array([min(x.timestamps) for x in X])

        # For each application in known, split data in train and test
        for application in labels_known:

            # Create mask for application
            mask_app = y == application
            # Get timestamp on which to split
            split = np.percentile(timestamps[mask_app], 100*size_train)
            # Set train and test data for given application
            train = timestamps[mask_app] <= split
            test  = ~train

            # Add train data and labels
            X_train.append(X[mask_app][train])
            y_train.append(y[mask_app][train])

            # Add test data and labels
            X_test_known.append(X[mask_app][test])
            y_test_known.append(y[mask_app][test])

        # Combine train and test data
        X_train      = np.concatenate(X_train)
        y_train      = np.concatenate(y_train)
        X_test_known = np.concatenate(X_test_known)
        y_test_known = np.concatenate(y_test_known)

        # Return result
        return X_train       , y_train,\
               X_test_known  , y_test_known,\
               X_test_unknown, y_test_unknown

    def split_normal(self, X, y, size_known=1/2, size_train=1/2):
        # Get data as numpy array
        y = np.asarray(y)
        X = np.asarray(X)

        # Initialise results
        X_train     , y_train      = list(), list()
        X_test_known, y_test_known = list(), list()

        # Get unique labels
        labels = np.unique(y)
        # Split labels in known and unknown
        labels_unknown, labels_known = train_test_split(labels,
                                            test_size=size_known,
                                            random_state=self.rs)

        # Get mask unknown
        mask_unknown = np.isin(y, labels_unknown)
        X_test_unknown = X[mask_unknown]
        y_test_unknown = y[mask_unknown]

        # Get mask known
        mask_known = ~mask_unknown
        # Split train test of known
        X_train, X_test_known, y_train, y_test_known = \
            train_test_split(X[mask_known], y[mask_known],
                             test_size=1-size_train, random_state=self.rs)

        # Return result
        return X_train       , y_train,\
               X_test_known  , y_test_known,\
               X_test_unknown, y_test_unknown
