import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split


def fingerprint(flowprint, args):
    """Execute Flowprint in fingerprint mode"""
    ################################################################
    #                      Process input data                      #
    ################################################################
    # Initialise flows and labels
    X, y = list(), list()
    # Initialise preprocessor
    preprocessor = Preprocessor(verbose=True)

    # Parse files - if necessary
    if args.pcaps:
        # Process data
        X_, y_ = preprocessor.process(args.pcaps, args.pcaps)
        # Add data to datapoints
        X.append(X_)
        y.append(y_)

    # Load preprocessed data - if necessary
    if args.read:
        # Loop over all preprocessed files
        for infile in args.read:
            # Load each file
            X_, y_ = preprocessor.load(infile)
            # Add input file to data
            X.append(X_)
            y.append(y_)

    # Concatenate datapoints
    X = np.concatenate(X)
    y = np.concatenate(y)

    # Write preprocessed data - if necessary
    if args.write:
        # Save data
        preprocessor.save(args.write, X, y)

    ################################################################
    #               Split fingerprints if necessary               #
    ################################################################

    if args.split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.split, random_state=args.random)
        data = [('train', X_train, y_train), ('test', X_test, y_test)]
    else:
        data = [('', X, y)]

    # Loop over both sets
    for type, X, y in data:

        ################################################################
        #                     Create fingerprints                      #
        ################################################################

        # Fit fingerprints
        flowprint.fit(X, y)

        # Make dictionary of label -> fingerprints
        fingerprints = dict()
        # Fill fingerprints
        for fingerprint, label in flowprint.fingerprints.items():
            # Add fingerprints
            fingerprints[label] = fingerprints.get(label, []) + [fingerprint]

        ################################################################
        #                     Output fingerprints                      #
        ################################################################

        # Output to file
        if args.fingerprint:
            # Set output file
            outfile, ext = os.path.splitext(args.fingerprint)
            outfile = "{}{}{}".format(outfile, '.' + type if type else type, ext)
            # Store fingerprints
            flowprint.store(outfile)
            # Notify user fingerprints were saved
            print("Fingerprints saved to {}".format(outfile))

        # Output to terminal
        else:
            # Select type of fingerprints to output
            print("Output {}fingerprints:".format(type + ' ' if type else type))
            # Output fingerprints
            for label, fingerprint in sorted(fingerprints.items()):
                print("{}:".format(label))
                for fp in sorted(fingerprint):
                    # Get fingerprints as set
                    print("    {}".format(fp))
                print()




if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(
                prog="flowprint.py",
                description="Flowprint: Semi-Supervised Mobile-App\nFingerprinting on Encrypted Network Traffic",
                formatter_class=argparse.RawTextHelpFormatter)

    # Output arguments
    group_output = parser.add_mutually_exclusive_group(required=False)
    group_output.add_argument('--fingerprint', type=str  , nargs='?')
    group_output.add_argument('--detection'  , type=float           )
    group_output.add_argument('--recognition', action='store_true'  )

    # FlowPrint parameters
    group_flowprint = parser.add_argument_group("FlowPrint parameters")
    group_flowprint.add_argument('-b', '--batch'      , type=float, default=300)
    group_flowprint.add_argument('-c', '--correlation', type=float, default=0.1)
    group_flowprint.add_argument('-s', '--similarity' , type=float, default=0.9)
    group_flowprint.add_argument('-w', '--window'     , type=float, default=30 )

    # Flow data input/output agruments
    group_data_in = parser.add_argument_group("Flow data input/output")
    group_data_in.add_argument('-p', '--pcaps' , type=str,   nargs='+' )
    group_data_in.add_argument('-r', '--read'  , type=str,   nargs='+' )
    group_data_in.add_argument('-o', '--write' , type=str,             )
    group_data_in.add_argument('-l', '--split' , type=float, default= 0)
    group_data_in.add_argument('-a', '--random', type=int  , default=42)

    # Train/test input arguments
    group_data_fps = parser.add_argument_group("Train/test input")
    group_data_fps.add_argument('-t', '--train', type=str, nargs='+')
    group_data_fps.add_argument('-e', '--test' , type=str, nargs='+')

    # Set help message
    parser.format_help = lambda: \
"""usage: {} [-h]
                    (--detection [FLOAT] | --fingerprint [FILE] | --recognition)
                    [-b BATCH] [-c CORRELATION], [-s SIMILARITY], [-w WINDOW]
                    [-p PCAPS...] [-rp READ...] [-wp WRITE]

{}

Arguments:
  -h, --help                 show this help message and exit

FlowPrint mode (select up to one):
  --fingerprint [FILE]       run in raw fingerprint generation mode (default)
                             outputs to terminal or json FILE
  --detection   FLOAT        run in unseen app detection mode with given
                             FLOAT threshold
  --recognition              run in app recognition mode

FlowPrint parameters:
  -b, --batch       FLOAT    batch size in seconds       (default=300)
  -c, --correlation FLOAT    cross-correlation threshold (default=0.1)
  -s, --similarity  FLOAT    similarity threshold        (default=0.9)
  -w, --window      FLOAT    window size in seconds      (default=30)

Flow data input/output (either --pcaps or --read required):
  -p, --pcaps  PATHS...      path to pcap(ng) files to run through FlowPrint
  -r, --read   PATHS...      read preprocessed data from given files
  -o, --write  PATH          write preprocessed data to given file
  -i, --split  FLOAT         fraction of data to select for testing (default= 0)
  -a, --random FLOAT         random state to use for split          (default=42)

Train/test input (for --detection/--recognition):
  -t, --train PATHS...       path to json files containing training fingerprints
  -e, --test  PATHS...       path to json files containing testing fingerprints
""".format(
    # Usage Parameters
    parser.prog,
    # Description
    parser.description)

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                           Check arguments                            #
    ########################################################################

    # --fingerprint requires --pcaps or --read
    if not args.detection and\
       not args.recognition and\
       not args.pcaps and\
       not args.read:
        # Give help message
        print(parser.format_help())
        # Throw exception
        raise RuntimeError("--recognition requires input data, please specify "
                           "--pcaps or --read arguments.")

    # --detection or --recognition require --train and --test
    if (args.detection or args.recognition) and not (args.train and args.test):
        # Give help message
        print(parser.format_help())
        # Throw exception
        raise RuntimeError("--detection/--recognition require training and "
                           "testing fingerprints, please specify --train and "
                           "--test arguments.")

    ########################################################################
    #                           Create FlowPrint                           #
    ########################################################################

    # Create FlowPrint instance with given arguments
    flowprint = FlowPrint(
        batch       = args.batch,
        window      = args.window,
        correlation = args.correlation,
        similarity  = args.similarity
    )

    ########################################################################
    #                             Execute mode                             #
    ########################################################################
    # Fingerprint mode
    if not args.detection and not args.recognition:
        fingerprint(flowprint, args)
    # Detection/Recognition mode
    else:

        ################################################################
        #                      Load fingerprints                       #
        ################################################################
        # Load FlowPrint with train fingerprints
        flowprint.load(*args.train)
        # Load test fingerprints and labels from file
        X_test, y_test = zip(*
            flowprint.load(*args.test, store=False).items()
        )

        ################################################################
        #                         Execute mode                         #
        ################################################################
        # Detection mode
        if args.detection:
            prediction = flowprint.detect(X_test, threshold=args.detection)

        # Recognition mode
        elif args.recognition:
            prediction = flowprint.recognize(X_test)

        ################################################################
        #                         Show result                          #
        ################################################################

        y_current = None
        # Loop over all fingerprints sorted by input label
        for fp, y_test_, y_pred_ in sorted(zip(X_test, y_test, prediction),
                                            key=lambda x: list(x[1])):
            # Get label of fingerprint
            y_test_ = list(y_test_)[0]
            # Print label if new one is found
            if y_test_ != y_current:
                print('\n',y_test_)
                y_current = y_test_

            # Output result
            if args.recognition:
                y_pred_ = list(y_pred_)[0]
            print("    {} --> {}".format(fp, y_pred_))
