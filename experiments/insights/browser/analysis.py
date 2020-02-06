import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

# Import Flowprint
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import main
from cluster_optimised import ClusterOptimised
from utils             import get_indices

def plot(df, column, intervals, metric=lambda x: np.unique(x).shape[0], lines=[]):
    # For each interval show plot
    for interval in intervals:
        array = df[column].rolling(interval).apply(metric, raw=True)
        plt.plot(df.index, array, label=interval)

    # Plot vertical lines if given
    for line in lines:
        plt.axvline(x=np.array([line], dtype="datetime64[s]"), c='k')

    # Show plot
    plt.legend()
    plt.show()

def batchify(timestamps, diff):
    # Initialise result
    result = list()
    length = 0

    while length < timestamps.shape[0]:
        # Set mask for timstamps >= current and timeframe diff
        mask = np.logical_and(timestamps <  timestamps[length] + diff,
                              timestamps >= timestamps[length])
        # Get indices of batch
        batch = np.argwhere(mask).flatten()
        # Add batch to result
        result.append(batch)
        # Set next offset
        length += batch.shape[0]

    # Return result
    return result

def compute_stats(X, column, batch_size=float('inf')):
    """Computes several statistics about data.

        Parameters
        ----------
        X : np.array of shape=(n_samples, n_features)
            Data to fit to cluster

        column : dict
            Dictionary of 'feature_name' -> index
        """
    # Get timestamps
    timestamps = X[:, column["timestamp"]]
    # Sort everything based on timestamp
    sort_index = np.argsort(timestamps)
    timestamps = timestamps[sort_index]
    X          = X[sort_index]
    # Compute batch indices
    batches = batchify(timestamps, batch_size)

    # Initialise cluster
    cluster = ClusterOptimised(slices=[[column["TLS certificate issuer name"],
                                        column["TLS certificate signature"],
                                        column["TLS certificate subject name"],
                                        column["TLS certificate subject pk"],
                                        column["TLS certificate subject pk algorithm"],
                                        column["TLS certificate version"]],

                                      [column['IP dst'],
                                       column['TCP dport']]],
                                  min_samples=5,
                                  none_eq=False)
    # Get predictions
    predictions = cluster.fit_predict(X)

    # Loop over all batches
    for batch in batches:
        print(batch)



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate browser statistics.")
    parser.add_argument('-f', '--file', type=str, help='files to analyse.')
    args   = parser.parse_args()

    infile = args.file or 'browser.p'

    # Read browser file
    with open(infile, 'rb') as infile:
        X, columns = pickle.load(infile)

    # Get columns as indices
    column = get_indices(columns)

    #compute_stats(X, column, 10)
    #exit()

    # Create optimised cluster
    cluster = ClusterOptimised(slices=[[column["TLS certificate issuer name"],
                                        column["TLS certificate signature"],
                                        column["TLS certificate subject name"],
                                        column["TLS certificate subject pk"],
                                        column["TLS certificate subject pk algorithm"],
                                        column["TLS certificate version"]],

                                      [column['IP dst'],
                                       column['TCP dport']]],
                                  min_samples=5,
                                  none_eq=False)
    # Fit cluster with data
    cluster.fit(X)
    #cluster.to_csv('tmp.csv', index_ts=column["timestamp"])
    # Get timestamps from data
    ts = X[:, column["timestamp"]]

    # Generate flows
    flow_id = np.apply_along_axis(lambda x: hash(str(x)), 1, X[:,
                   [column["IP src"], column["TCP sport"],
                    column["IP dst"], column["TCP dport"]]])
    cluster_id = cluster.predict(X)
    size_packets = X[:, column["TCP stream length"]]
    size_bytes_in  = size_packets * X[:, column["TCP length incoming mean"]]
    size_bytes_out = size_packets * X[:, column["TCP length outgoing mean"]]

    # Create dataframe
    df = pd.DataFrame({
        "flow_id": flow_id,
        "cluster_id": cluster_id,
        "size_packets": size_packets,
        "size_bytes_in": size_bytes_in,
        "size_bytes_out": size_bytes_out
    }, index=pd.to_datetime(ts, unit='s')).sort_index()

    # Plot rolling averages - Number of unique clusters in past timeframes
    plot(df, 'cluster_id', ['1s', '5s', '10s'])
    plot(df, 'flow_id'   , ['1s', '5s', '10s'])
    exit()

    # Average number of flows per second
    # print(X.shape[0] / (ts.max() - ts.min()))
    # print(X.shape[0])
    # print((ts.max() - ts.min()))

    # Plot
    # plt.plot(np.sort(ts), np.arange(ts.shape[0]))
    # plt.axvline(x=1544541212) # tweakers.net
    # plt.axvline(x=1544541333) # ns.nl
    # plt.axvline(x=1544541383) # nu.nl
    # plt.axvline(x=1544541458) # sporcle.com
    # plt.axvline(x=1544541540) # funnyjunk.com
    # plt.show()
