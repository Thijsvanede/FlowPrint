import argparse
import os
import numpy as np

from flowprint.preprocessor  import Preprocessor
from flowprint.flowprint     import FlowPrint
from sklearn.metrics         import classification_report
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    ########################################################################
    #                             Handle input                             #
    ########################################################################
    # Parse input
    parser = argparse.ArgumentParser("FlowPrint recognition example")
    parser.add_argument('--files' , nargs='+', help='files to use as input')
    parser.add_argument('--dir'   , help='directory containing files to use as input')
    parser.add_argument('--ratio' , type=float, default=0.5, help='train ratio of data (default=0.5)')
    parser.add_argument('--random', action='store_true', help='split randomly instead of sequentially')
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
    #                              Split data                              #
    ########################################################################
    if args.random:
        # Perform random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.ratio, random_state=42)

    # Perform temporal split split
    else:
        # Initialise training and testing data
        X_train = list()
        y_train = list()
        X_test  = list()
        y_test  = list()

        # Loop over each different app
        for app in np.unique(y):
            # Extract flows relevant for selected app
            X_app = X[y == app]
            y_app = y[y == app]

            # Create train and test instances by split
            X_app_train = X_app[:int(X_app.shape[0]*args.ratio) ]
            y_app_train = y_app[:int(X_app.shape[0]*args.ratio) ]
            X_app_test  = X_app[ int(X_app.shape[0]*args.ratio):]
            y_app_test  = y_app[ int(X_app.shape[0]*args.ratio):]

            # Append to training/testing data
            X_train.append(X_app_train)
            y_train.append(y_app_train)
            X_test.append(X_app_test)
            y_test.append(y_app_test)

            # Print how we split the data
            print("Split {:40} into {} train and {} test flows".format(
                app, X_app_train.shape[0], X_app_test.shape[0]))

        # Concatenate
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_test  = np.concatenate(X_test )
        y_test  = np.concatenate(y_test )

    ########################################################################
    #                              Flowprint                               #
    ########################################################################
    # Create FlowPrint example
    flowprint = FlowPrint(
        batch       = 300,
        window      = 30,
        correlation = 0.1,
        similarity  = 0.9
    )

    # Fit FlowPrint with training data
    flowprint.fit(X_train, y_train)
    # Create test fingerprints
    fp_test = flowprint.fingerprinter.fit_predict(X_test)
    # Create prediction
    y_pred = flowprint.recognize(fp_test)

    ########################################################################
    #                           Print evaluation                           #
    ########################################################################
    print(classification_report(y_test, y_pred, digits=4))
