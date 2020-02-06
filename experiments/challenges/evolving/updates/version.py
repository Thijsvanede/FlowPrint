from itertools import combinations, product
import argparse
import datetime
import glob
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../flowprint')))
from loader import Loader
from cluster import Cluster
from fingerprint import Fingerprint
from fingerprints import Fingerprints
from preprocessor import Preprocessor
from split import Splitter

################################################################################
#                          Load application versions                           #
################################################################################

def load(directory, dates):
    """Load application versions from directory.

        Parameters
        ----------
        directory : string
            Directory containing directory/<appname>/<version>.p.

        dates : dict
            Dictionary of (app, version) -> timestamp release.

        Returns
        -------
        X : np.array of shape=(n_samples,)
            Flows.

        y : np.array of shape=(n_samples,)
            Application labels.

        versions : np.array of shape=(n_samples,)
            Application versions.

        releases : np.array of shape=(n_samples,)
            Timestamps of app version release.
        """
    # Initialise results
    X = list()
    y = list()
    versions = list()
    releases = list()

    # Initialise loading object
    loader = Loader()

    # Loop over all applications in directory
    for application in os.listdir(directory):
        # Loop over all versions in directory
        for version in os.listdir(os.path.join(directory, application)):
            # Check if file is pickled
            if version.endswith(".p"):
                # Get full path to file
                path = os.path.join(directory, application, version)
                # Remove extension from version
                version = int(version.rsplit('.', 1)[0])
                # Load single file
                X_, y_ = loader.load_file(path)
                # Append to results
                X.extend(X_.tolist())
                y.extend([application]*X_.shape[0])
                versions.extend([version]*X_.shape[0])
                releases.extend([dates.get((application, version), 0)]*X_.shape[0])

    # Return results as numpy arrays
    return np.asarray(X), np.asarray(y), np.asarray(versions), np.asarray(releases)

def load_extended(directory, dates):
    """Load extended application versions from directory.

        Parameters
        ----------
        directory : string
            Directory containing directory/<appname>/<date>/<id>_<version>.p.

        dates : dict
            Dictionary of (app, version) -> timestamp release.

        Returns
        -------
        X : np.array of shape=(n_samples,)
            Flows.

        y : np.array of shape=(n_samples,)
            Application labels.

        versions : np.array of shape=(n_samples,)
            Application versions.

        releases : np.array of shape=(n_samples,)
            Timestamps of app version release.
        """
    # Initialise results
    X = list()
    y = list()
    versions = list()
    releases = list()

    # Initialise loading object
    loader = Loader()

    for path in glob.iglob(directory + '**/*.p', recursive=True):
        # Parse path
        parts = path.split('/')
        application = parts[-3]
        date        = parts[-2]
        id, version = parts[-1].split('_')
        version     = int(version[:-2])

        # Load data
        X_, y_ = loader.load_file(path)
        # Append to results
        X.extend(X_.tolist())
        y.extend([application]*X_.shape[0])
        versions.extend([version]*X_.shape[0])
        releases.extend([dates.get((application, version), 0)]*X_.shape[0])

    # Return results as numpy arrays
    return np.asarray(X), np.asarray(y), np.asarray(versions), np.asarray(releases)

def date(datefile):
    """Get the (approximate) date of a version release from version name.

        Parameters
        ----------
        datefile : string
            Path to file containing dates

        Returns
        -------
        result : dict
            Dictionary of (application, version) -> release date.
        """
    # Initialise result
    result = dict()

    # Load csv into pandas dataframe
    df = pd.read_csv(datefile)

    # Create dictionary
    for index, row in df.iterrows():
        # Get key from row
        key = (row['packageName'], int(row['versionCode']))
        # Get release date from row
        release_date = row['releaseDate']
        # Convert to datetime
        try:
            release_date = datetime.datetime.strptime(release_date, '%Y-%m-%d')
        except ValueError:
            release_date = datetime.datetime.strptime(release_date, '%m/%d/%y')

        # Convert to timestamp
        release_date = release_date.timestamp()
        # Add entry to result
        result[key] = release_date

    # Return result
    return result

################################################################################
#                                Fingerprinting                                #
################################################################################
def fingerprinter(batch=300, window=30, correlation=0.1, similarity=0.9):
    """Create new fingerprinter object.

        Parameters
        ----------
        batch : float, default=300
            Length of a single input batch in seconds.
            Inputs of X are first divided into batches of the given length
            and each batch is subsequently run through the system.

        window : float, default=30
            Length of an activity window in seconds.
            Each batch is subdivided into windows and activity correlation
            is computed when destination clusters are active in the same
            time window.

        correlation : float, default=0.1
            Minimum required correlation between destination clusters in
            order to be regarded as correlating. If they are correlating,
            clusters receive an edge between them in the cross-correlation
            graph.

        similarity : float, default=0.9
            Minimum required Jaccard similarity for fingerprints to be
            considered equivalent.

        Returns
        -------
        result : Fingerprints
            Returns new fingerprinting object.
        """
    # Create clustering object
    cluster = Cluster()
    # Create Fingerprinter and return
    return Fingerprints(cluster, batch, window, correlation, similarity)

################################################################################
#                Create fingerprints accross different versions                #
################################################################################
def fingerprints(X, y, versions, dates):
    """Generate fingerprints for each application and version.

        Parameters
        ----------
        X : np.array of shape=(n_samples,)
            Flows.

        y : np.array of shape=(n_samples,)
            Application labels.

        versions : np.array of shape=(n_samples,)
            Application versions.

        dates : dict
            Dictionary of (app, version) -> timestamp release.

        Returns
        -------
        result : dict
            Dictionary of application -> version -> set of fingerprints.
        """
    # Initialise result
    result = dict()

    # Loop over all different application versions
    for app, version, data, date in split(X, y, versions, dates):
        # print("\x1b[2KFingerprinting {}:{}".format(app, version), end='\r')
        # Get application dictionary
        app_versions = result.get(app, dict())
        if (version, date) in app_versions:
            raise ValueError("This should not happen")
        # Generate fingerprints of application version and store them
        app_versions[(version, date)] = set(fingerprinter().fit_predict_fingerprints(data))
        # Store dictionary
        result[app] = app_versions

    # Return result
    return result

def split(X, y, versions, dates):
    """Generator for splitting data between different applications and versions.

        Parameters
        ----------
        X : np.array of shape=(n_samples,)
            Flows.

        y : np.array of shape=(n_samples,)
            Application labels.

        versions : np.array of shape=(n_samples,)
            Application versions.

        dates : dict
            Dictionary of (app, version) -> timestamp release.

        Yields
        ------
        application : string
            Identifier of application

        version : string
            Identifier of application version

        data : np.array of shape=(n_app_version_samples,)
            Samples belonging to given application and version.

        date : float
        """
    # Loop over unique applications
    for application in np.unique(y):
        # Get data of given application
        X_ = X[y == application]
        y_ = y[y == application]
        versions_ = versions[y == application]
        # Get unique versions
        versions_u = np.unique(versions_)

        # Loop over sorted versions of that application
        for version in sorted(versions_u,
                              key=lambda x: dates.get((application, x), 0)):
            # Get data of given version
            data = X_[versions_ == version]
            # Yield result
            yield (application, version, data, dates.get((application, version), 0))

################################################################################
#                  Compare fingerprints of different versions                  #
################################################################################
def statistics(fingerprints, difference=1):
    """Get statistics for each application version."""
    # Initialise result
    result = dict()

    # Loop over all applications
    for application, versions in fingerprints.items():
        # Compute statistics and store
        result[application] = compare(versions, difference=difference)

    # Return result
    return result

def compare(versions, threshold=0.9, difference=1):
    """Compares fingerprints of different versions of applications.

        Parameters
        ----------
        versions : dict
            Dictionary of version -> fingerprints

        Returns
        -------
        result : np.array of shape=(n_fingerprints,)
            Table of comparison scores.
        """
    # Get a list of all fingerprints
    result = dict()

    result["version time diff"] = np.diff(np.asarray([k[1] for k in versions.keys() if k[1] != 0]))
    # Transform to versions
    versions = [v for k, v in sorted(versions.items(), key=lambda x: x[0][1])]

    # Set number of versions
    result["versions"]             = len(versions)
    result["fingerprints"]         = len(versions[0])
    result["fingerprint score"]    = 0
    result["fingerprint existing"] = 0
    result["fingerprint new"]      = 0
    result["version overlap"]      = 0
    result["version overlap size"] = 0

    # Common fingerprints between all sets
    fps_old = dict()

    # Loop over subsequent versions
    for i in range(0, len(versions)-difference):
        version_old = versions[i]
        version_new = versions[i+difference]

        # Add to total number of fingerprints
        result["fingerprints"] += len(version_new)

        # Add old fingerprints to set
        for fp in version_old:
            # Check if not yet in set
            if fp not in fps_old:
                # Add first observation and no matches
                fps_old[fp] = (i, 1, 0)

        # Check for overlap between versions
        overlap = set()

        # Loop over new fingerprints
        for fp in version_new:
            # Initialise highest score
            high_score = 0
            # Loop over known fingerprints of app
            for fp_old, entry in fps_old.items():
                # Compute score
                score = fp_old.compare(fp)
                # Set high_score if applicable
                if score > high_score:
                    # Set high score
                    high_score = score
            # Check if high_score is sufficient
            if high_score >= 1-threshold:
                # Add fingerprint to overlap
                overlap.add(fp)
                result["fingerprint existing"] += 1
            else:
                result["fingerprint new"] +=1

        # Add overlap
        result["version overlap"]      += int(len(overlap) > 0)
        result["version overlap size"] += len(overlap)


    # Loop over fingerprints
    for fp, (first, total, score) in fps_old.items():
        # Normalise score
        result["fingerprint score"] += score/total

    # Return result
    return result

################################################################################
#                              Display statistics                              #
################################################################################
def show(statistics, difference=1):
    """"""
    # Compute general statistics
    min_time_diff = np.asarray([v["version time diff"].min()  for k, v in statistics.items() if v["version time diff"].shape[0]])
    max_time_diff = np.asarray([v["version time diff"].max()  for k, v in statistics.items() if v["version time diff"].shape[0]])
    avg_time_diff = np.asarray([v["version time diff"].mean() for k, v in statistics.items() if v["version time diff"].shape[0]])
    med_time_diff = np.asarray([v["version time diff"].mean() for k, v in statistics.items() if v["version time diff"].shape[0]])
    # med_time_diff = np.concatenate([v["version time diff"]    for k, v in statistics.items() if v["version time diff"].shape[0]])
    std_time_diff = np.asarray([v["version time diff"].std()  for k, v in statistics.items() if v["version time diff"].shape[0]])

    n_apps     = len(statistics)
    n_versions = sum(v["versions"    ] for k, v in statistics.items())
    n_prints   = sum(v["fingerprints"] for k, v in statistics.items())
    n_app_mult = 0
    n_versions_mult = 0
    n_prints_mult   = 0
    fp_existing     = 0
    fp_new          = 0
    fp_scores       = 0
    version_overlap = 0
    version_overlap_s = 0

    # Loop over applications
    for application, stats in statistics.items():
        # Compute application statistics > 1 version
        if int(stats["versions"]) > difference:
            n_app_mult      += 1
            n_versions_mult += int(stats["versions"])
            n_prints_mult   += int(stats["fingerprints"])
            fp_existing     += stats["fingerprint existing"]
            fp_new          += stats["fingerprint new"]
            fp_scores       += stats["fingerprint score"]
            version_overlap += stats["version overlap"]
            version_overlap_s += stats["version overlap size"]

    print("""

Statistics
--------------------------------------------------
  Total number of applications    : {}
  Total number of versions        : {}
  Total number of fingerprints    : {}
  Avg versions     per app        : {:.4f}
  Avg fingerprints per app        : {:.4f}
  Avg fingerprints per version    : {:.4f}
--------------------------------------------------
  Number of apps multiple versions: {}
  Avg versions     per app        : {:.4f}
  Avg fingerprints per app        : {:.4f}
  Avg fingerprints per version    : {:.4f}
--------------------------------------------------
  Min time difference per version : {}
  Max time difference per version : {}
  Avg time difference per version : {}
  Med time difference per version : {}
  Std time difference per version : {}
--------------------------------------------------
  Shared fingerprints             : {}
  New    fingerprints             : {}
  Shared ratio                    : {:.4f}
  Avg matching score              : {:.4f}
--------------------------------------------------
  Overlapping     versions        : {}
  Non-overlapping versions        : {}
  Overlapping     ratio           : {:.4f}
  Avg overlapping fingerprints    : {:.4f}
""".format(n_apps, n_versions, n_prints,
           n_versions/n_apps, n_prints/n_apps, n_prints/n_versions,

           n_app_mult, n_versions_mult/n_app_mult,
           n_prints_mult/n_app_mult, n_prints_mult/n_versions_mult,

            min_time_diff, max_time_diff.max(), avg_time_diff.mean(), np.median(med_time_diff), std_time_diff.mean(),

           fp_existing, fp_new, fp_existing/(fp_new+fp_existing),
           fp_scores/n_prints_mult,

           version_overlap, n_versions_mult-version_overlap,
           version_overlap/n_versions_mult, version_overlap_s/version_overlap))


if __name__ == "__main__":
    # Get dates
    dates = date('../../../../../../data/ReCon/mt.csv')

    # Load data ReCon
    # X, y, version, date = load("../../../../../../data/ReCon/processed/", dates)

    # Load data ReCon extended
    # X, y, version, date = load_extended("../../../../../../data/ReCon/appversions/processed/", dates)

    # Load both ReCon and ReCon extended
    X1, y1, version1, date1 = load("../../../../../../data/ReCon/processed/", dates)
    X2, y2, version2, date2 = load_extended("../../../../../../data/ReCon/appversions/processed/", dates)

    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2))
    version = np.concatenate((version1, version2))

    # Create fingerprints
    fps = fingerprints(X, y, version, dates)
    # Compare fingerprints
    for difference in range(1, 50):
        print("Difference = {}".format(difference))
        show(statistics(fps, difference=difference), difference=difference)
