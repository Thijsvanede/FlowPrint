import argparse
import os
import numpy as np

from flowprint.preprocessor             import Preprocessor
from flowprint.cluster                  import Cluster
from flowprint. cross_correlation_graph import CrossCorrelationGraph

if __name__ == "__main__":
    ########################################################################
    #                             Handle input                             #
    ########################################################################
    # Parse input
    parser = argparse.ArgumentParser("FlowPrint CrossCorrelationGraph export example", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--files'      , nargs='+', help='files to use as input')
    parser.add_argument('--dir'        , help='directory containing files to use as input')
    parser.add_argument('-b', '--batch'      , type=float, default=float('inf'), help='batch size in seconds creates separate cross correlation graph per batch (default=inf)')
    parser.add_argument('-c', '--correlation', type=float, default=0.1,          help='cross-correlation threshold,  used for cross correlation graph           (default=0.1)')
    parser.add_argument('-w', '--window'     , type=float, default=30 ,          help='window size in seconds     ,  used for cross correlation graph           (default= 30)')
    args = parser.parse_args()

    # Check if arguments were given
    if (args.files is None and args.dir is None) or\
       (args.files is not None and args.dir is not None):
        raise ValueError("Please specify either --files or --dir but not both.")

    # Get file names
    files = args.files or [args.dir+x for x in os.listdir(args.dir)]

    ########################################################################
    #                              Read data                               #
    ########################################################################
    # Create preprocessor
    preprocessor = Preprocessor(verbose=True)
    # Process all files
    X, y = preprocessor.process(files, files)

    ########################################################################
    #  Divide input into batches (also FingerprintGenerator.fit_predict()) #
    ########################################################################
    sort_time = np.argsort(X)
    X = X[sort_time]
    y = y[sort_time]

    # Divide X and y in batches of size batch
    sort_batch = np.array([x.time_start for x in X])
    # Compute number of required batches
    if sort_batch.shape[0]:
        batches = int(
            np.ceil((max(sort_batch) - min(sort_batch)) / args.batch)
        )
    else:
        batches = 0
    # Get edges of batches
    _, edges = np.histogram(sort_batch, bins=max(1, batches))
    # Sort indices into batches
    batches = np.digitize(sort_batch, edges[:-1])

    # Loop over all batches
    for i, batch in enumerate(np.unique(batches)):

        # Get data for given batch
        X_batch = X[batches == batch]
        y_batch = y[batches == batch]

        ################################################################
        #                   Create correlation graph                   #
        ################################################################
        # Create Cluster
        cluster = Cluster()
        # Fit cluster
        cluster = cluster.fit(X_batch, y_batch)

        # Create CrossCorrelationGraph from cluster
        ccg = CrossCorrelationGraph(
            window      = args.window,
            correlation = args.correlation
        ).fit(cluster)

        ################################################################
        #                   Export correlation graph                   #
        ################################################################

        # Export CrossCorrelationGraph for each batch
        ccg.export('test_batch_{}.gexf'.format(i), dense=True, format='gexf')
