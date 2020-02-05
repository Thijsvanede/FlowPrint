import argparse
import numpy as np

from flowprint import FlowPrint
from preprocessor import Preprocessor

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
    group_output.add_argument('--fingerprint', action='store_true', help="run FlowPrint in raw fingerprint generation mode (default)")
    group_output.add_argument('--detection'  , action='store_true', help="run FlowPrint in unseen app detection mode")
    group_output.add_argument('--recognition', action='store_true', help="run FlowPrint in app recognition mode")

    # FlowPrint parameters
    group_flowprint = parser.add_argument_group("FlowPrint parameters")
    group_flowprint.add_argument('-b', '--batch'      , type=float, default=300, help="batch size in seconds       (default=300)")
    group_flowprint.add_argument('-c', '--correlation', type=float, default=0.1, help="cross-correlation threshold (default=0.1)")
    group_flowprint.add_argument('-s', '--similarity' , type=float, default=0.9, help="similarity threshold        (default=0.9)")
    group_flowprint.add_argument('-w', '--window'     , type=float, default=30 , help="window size in seconds      (default=30)")

    # Data agruments
    group_data_in = parser.add_argument_group("Data input")
    group_data_in.add_argument('-p', '--pcaps', nargs='+', help="path to pcap(ng) files to run through FlowPrint")
    group_data_in.add_argument('-r', '--read' , nargs='+', help="read preprocessed data from given files")
    group_data_in.add_argument('-t', '--write',            help="write preprocessed data to given file")

    # Set help message
    parser.format_help = lambda: \
"""usage: {} [-h]
                    (--detection | --fingerprint | --recognition)
                    [-b BATCH] [-c CORRELATION], [-s SIMILARITY], [-w WINDOW]
                    [-p PCAPS...] [-rp READ...] [-wp WRITE]

{}

Arguments:
  -h, --help                 show this help message and exit

FlowPrint mode (select up to one):
  --fingerprint              run in raw fingerprint generation mode (default)
  --detection                run in unseen app detection mode
  --recognition              run in app recognition mode

FlowPrint parameters:
  -b, --batch       FLOAT    batch size in seconds       (default=300)
  -c, --correlation FLOAT    cross-correlation threshold (default=0.1)
  -s, --similarity  FLOAT    similarity threshold        (default=0.9)
  -w, --window      FLOAT    window size in seconds      (default=30)

Data input (either --files or --read required):
  -p, --pcaps PATHS...       path to pcap(ng) files to run through FlowPrint
  -r, --read  PATHS...       read preprocessed data from given files
  -t, --write PATH           write preprocessed data to given file

""".format(
    # Usage Parameters
    parser.prog,
    # Description
    parser.description)

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                          Process input data                          #
    ########################################################################

    # Check if any input is given
    if not args.pcaps and not args.read:
        # Give help message
        print(parser.format_help())
        # Throw exception
        raise RuntimeError("No input data provided, please specify --pcaps or --read arguments.")

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

    ########################################################################
    #                            Run FlowPrint                             #
    ########################################################################

    # Create FlowPrint instance
    flowprint = FlowPrint()
    # Fit fingerprints
    flowprint.fit(X, y)

    # Run FlowPrint in given mode

    if args.detection:
        raise ValueError("Mode not implemented yet: Detection")

    elif args.recognition:
        raise ValueError("Mode not implemented yet: Recognition")

    else:
        # Make dictionary of label -> fingerprints
        fingerprints = dict()
        # Fill fingerprints
        for fingerprint, label in flowprint.fingerprints.items():
            # Add fingerprints
            fingerprints[label] = fingerprints.get(label, []) + [fingerprint]

        # Output fingerprints
        for label, fingerprint in fingerprints.items():
            print("{}:".format(label))
            for fp in fingerprint:
                print("    {}".format(fp))
            print()
