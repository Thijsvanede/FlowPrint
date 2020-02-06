import argparse
import json
import numpy as np
import os
from flowprint import FlowPrint
from flow import Flow

class IO(object):

    def __init__(self):
        """Input/Output handler for FlowPrint related objects."""
        self.flowprint = FlowPrint()

    ########################################################################
    #                               Flow I/O                               #
    ########################################################################

    def read_flows(self, infile):
        """Read flows and labels from input file.

            Parameters
            ----------
            infile : string
                Path to input file.

            Returns
            -------
            X : array-like of shape=(n_flows,)
                Flows read from input file.

            y : array-like of shape=(n_flows,)
                Labels read from input file.
            """
        # Read input
        with open(infile, 'r') as infile:
            data = json.load(infile)
        # Transform data to X and y
        X = np.asarray([Flow(d[0]) for d in data])
        y = np.asarray([     d[1]  for d in data])

        # Return result
        return X, y

    def write_flows(self, X, y, outfile):
        """Write flows and labels to output file.

            Parameters
            ----------
            X : array-like of shape=(n_flows,)
                Flows to write to output file.

            y : array-like of shape=(n_flows,)
                Labels to write to output file.

            outfile : string
                Path to output file.
            """
        # Create output
        output = [(flow.to_dict(), label) for flow, label in zip(X, y)]

        # Create directory of outfile if it does not yet exist
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        # Dump output to output file
        with open(outfile, 'w') as outfile:
            json.dump(output, outfile)

    def create_flows(self, infile, label=None, timeout=float('inf')):
        """Create flows and labels by reading pcaps from input file.

            Parameters
            ----------
            infile : string
                Path to pcap file to read.

            label : object
                Label corresponding to flows from input file.

            timeout : float, default=float('inf')
                Timeout to use for reading flows.

            Returns
            -------
            X : np.array of shape=(n_flows,)
                Flows read from infile.

            y : np.array of shape=(n_flows,)
                Labels corresponding to flows in X.
            """
        # Create flows from input file
        X = list(self.flowprint.extract(path=infile, timeout=timeout))
        # Create labels corresponding to flow
        y = [label]*len(X)

        # Return result as numpy arrays
        return np.asarray(X), np.asarray(y)


if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint I/O module.')

    # Input/Output data arguments
    parser.add_argument('-f', '--files'  , type=str, nargs='+', help="pcap(ng) files to parse into FlowPrint Flows.")
    parser.add_argument('-r', '--read'   , type=str, nargs='+', help="JSON Input  files for parsed FlowPrint Flows.")
    parser.add_argument('-w', '--write'  , type=str, nargs='+', help="JSON Output files for parsed FlowPrint Flows.")
    parser.add_argument('-t', '--timeout', type=str, help="Timeout for reading flows (default=inf).")

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                            Run FlowPrint                             #
    ########################################################################

    # Creat FlowPrint object
    io = IO()

    # In case of loading files
    if args.files:
        # Check whether write arguments are correctly specified
        if not args.write:
            raise ValueError("If '--files' specified, '--write' should also be "
                             "specified.")
        if len(args.write) != len(args.files):
            raise ValueError("Number of input files={} should equal number of "
                             "output files={}.".format(len(args.files),
                             len(args.write)))

        # Loop over all input-output pairs
        for infile, outfile in zip(args.files, args.write):
            # Create data
            X, y = io.create_flows(infile, infile, args.timeout or float('inf'))
            # Store data
            io.write_flows(X, y, outfile)

    # In case of reading files
    if args.read:
        # Loop over all input files
        for infile in args.read:
            X, y = io.read_flows(infile)
            for x in X:
                print(x)
