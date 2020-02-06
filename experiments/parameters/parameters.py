from evaluate import Evaluator
from itertools import product
import argparse
import logging
import numpy as np
import pandas as pd

from os import path
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../flowprint')))
from loader import Loader
from preprocessor import Preprocessor
from split import Splitter

class Hyperparameters(object):

    def __init__(self, k=10):
        """Evaluate FlowPrint with different hyperparameters.

            Parameters
            ----------
            k : int, default=10
                Number of k-fold cross validations to run.
            """
        # Create evaluation object
        self.evaluator = Evaluator()

        # Set number of cross validations
        self.k = k
        self.splitter = Splitter()

        # Set hyperparameter sets
        self.batches = (60, 300, 600, 1800, 3600, 10800, 21600, 43200, 5, 10, 30)
        self.windows = (1, 5, 10, 30, 60, 300, 600, 1800)
        self.correlations = (0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)
        self.similarities = (0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)

        # Set logging instance
        self.logger = logging.getLogger("Hyperparameters")

    def run_quick(self, X, y, max_labels=100, fragmented=False):
        """Analyse k-fold FlowPrint performance for each hyperparameter.

            Parameters
            ----------
            X : np.array of shape=(n_flows,)
                Array of flows to evaluate.

            y : np.array of shape=(n_flows,)
                Array of labels corresponding to flows.

            max_labels : int, default=100
                Maximum number of applications to check.

            fragmented : string, default=None
                If given, save analysis after each hyperparameter setting in
                given filename.

            Returns
            -------
            result : pandas.DataFrame
                Dataframe containing results for all hyperparameters and k-folds
                over given input data X and labels y.
            """
        # Initialise dataframe
        df = pd.DataFrame(columns=['batch', 'window', 'correlation',
                              'similarity', 'k', 'support', 'accuracy',
                              'precision', 'recall', 'f1-score', 'homogeneity',
                              'completeness', 'v-measure', 'ari'], dtype=float)

        # Set default values
        default_batch  = 3600
        default_window = 5
        default_correlation = 0.3
        default_similarity  = 0.5

        ####################################################################
        #                       Run through batches                        #
        ####################################################################
        for batch in self.batches:
            # Loop over various K-folds
            for k, (X_, y_) in enumerate(self.splitter.k_fold_label(X, y, self.k, max_labels=max_labels)):

                # Log current evaluation
                self.logger.info("Processing ({}, {}, {}, {}, {})".format(
                    batch, default_window, default_correlation, default_similarity, k))

                # Run single evaluation
                eval = self.evaluator.run(X_, y_, batch, default_window,
                                        default_correlation, default_similarity)

                # Get single evaluation entry
                entry = {
                    'batch'       : batch,
                    'window'      : default_window,
                    'correlation' : default_correlation,
                    'similarity'  : default_similarity,
                    'k'           : k,
                    'support'     : X_.shape[0],
                    'accuracy'    : eval.get('accuracy'),
                    'precision'   : eval.get('precision'),
                    'recall'      : eval.get('recall'),
                    'f1-score'    : eval.get('f1-score'),
                    'homogeneity' : eval.get('homogeneity'),
                    'completeness': eval.get('completeness'),
                    'v-measure'   : eval.get('v-measure'),
                    'ari'         : eval.get('ari')
                }
                # Add entry to dataframe
                df = df.append(entry, ignore_index=True)

                # Save if fragmented file given
                if fragmented is not None:
                    self.write(df, fragmented)

        # Set default_batch
        default_batch = self.get_optimal(df, 'batch', self.batches)

        ####################################################################
        #                       Run through windows                        #
        ####################################################################
        for window in self.windows:
            # Loop over various K-folds
            for k, (X_, y_) in enumerate(self.splitter.k_fold_label(X, y, self.k, max_labels=max_labels)):

                # Log current evaluation
                self.logger.info("Processing ({}, {}, {}, {}, {})".format(
                    default_batch, window, default_correlation, default_similarity, k))

                # Run single evaluation
                eval = self.evaluator.run(X_, y_, default_batch, window,
                                        default_correlation, default_similarity)

                # Get single evaluation entry
                entry = {
                    'batch'       : default_batch,
                    'window'      : window,
                    'correlation' : default_correlation,
                    'similarity'  : default_similarity,
                    'k'           : k,
                    'support'     : X_.shape[0],
                    'accuracy'    : eval.get('accuracy'),
                    'precision'   : eval.get('precision'),
                    'recall'      : eval.get('recall'),
                    'f1-score'    : eval.get('f1-score'),
                    'homogeneity' : eval.get('homogeneity'),
                    'completeness': eval.get('completeness'),
                    'v-measure'   : eval.get('v-measure'),
                    'ari'         : eval.get('ari')
                }
                # Add entry to dataframe
                df = df.append(entry, ignore_index=True)

                # Save if fragmented file given
                if fragmented is not None:
                    self.write(df, fragmented)

        # Set default window
        default_window = self.get_optimal(df, 'window', self.windows)

        ####################################################################
        #                     Run through correlations                     #
        ####################################################################
        for correlation in self.correlations:
            # Loop over various K-folds
            for k, (X_, y_) in enumerate(self.splitter.k_fold_label(X, y, self.k, max_labels=max_labels)):

                # Log current evaluation
                self.logger.info("Processing ({}, {}, {}, {}, {})".format(
                    default_batch, default_window, correlation, default_similarity, k))

                # Run single evaluation
                eval = self.evaluator.run(X_, y_, default_batch, default_window,
                                        correlation, default_similarity)

                # Get single evaluation entry
                entry = {
                    'batch'       : default_batch,
                    'window'      : default_window,
                    'correlation' : correlation,
                    'similarity'  : default_similarity,
                    'k'           : k,
                    'support'     : X_.shape[0],
                    'accuracy'    : eval.get('accuracy'),
                    'precision'   : eval.get('precision'),
                    'recall'      : eval.get('recall'),
                    'f1-score'    : eval.get('f1-score'),
                    'homogeneity' : eval.get('homogeneity'),
                    'completeness': eval.get('completeness'),
                    'v-measure'   : eval.get('v-measure'),
                    'ari'         : eval.get('ari')
                }
                # Add entry to dataframe
                df = df.append(entry, ignore_index=True)

                # Save if fragmented file given
                if fragmented is not None:
                    self.write(df, fragmented)

        # Set default_correlation
        default_correlation = self.get_optimal(df, 'correlation', self.correlations)

        ####################################################################
        #                     Run through similarities                     #
        ####################################################################
        for similarity in self.similarities:
            # Loop over various K-folds
            for k, (X_, y_) in enumerate(self.splitter.k_fold_label(X, y, self.k, max_labels=max_labels)):

                # Log current evaluation
                self.logger.info("Processing ({}, {}, {}, {}, {})".format(
                    default_batch, default_window, default_correlation, similarity, k))

                # Run single evaluation
                eval = self.evaluator.run(X_, y_, default_batch, default_window,
                                        default_correlation, similarity)

                # Get single evaluation entry
                entry = {
                    'batch'       : default_batch,
                    'window'      : default_window,
                    'correlation' : default_correlation,
                    'similarity'  : similarity,
                    'k'           : k,
                    'support'     : X_.shape[0],
                    'accuracy'    : eval.get('accuracy'),
                    'precision'   : eval.get('precision'),
                    'recall'      : eval.get('recall'),
                    'f1-score'    : eval.get('f1-score'),
                    'homogeneity' : eval.get('homogeneity'),
                    'completeness': eval.get('completeness'),
                    'v-measure'   : eval.get('v-measure'),
                    'ari'         : eval.get('ari')
                }
                # Add entry to dataframe
                df = df.append(entry, ignore_index=True)

                # Save if fragmented file given
                if fragmented is not None:
                    self.write(df, fragmented)

        # Set default_similarity
        default_similarity = self.get_optimal(df, 'similarity', self.similarities)

        # Return dataframe
        return df

    def run_full(self, X, y, fragmented=False):
        """Analyse k-fold FlowPrint performance over all hyperparameters.

            Parameters
            ----------
            X : np.array of shape=(n_flows,)
                Array of flows to evaluate.

            y : np.array of shape=(n_flows,)
                Array of labels corresponding to flows.

            fragmented : string, default=None
                If given, save analysis after each hyperparameter setting in
                given filename.

            Returns
            -------
            result : pandas.DataFrame
                Dataframe containing results for all hyperparameters and k-folds
                over given input data X and labels y.
            """
        # Initialise dataframe
        df = pd.DataFrame(columns=['batch', 'window', 'correlation',
                              'similarity', 'k', 'support', 'accuracy',
                              'precision', 'recall', 'f1-score', 'homogeneity',
                              'completeness', 'v-measure', 'ari'], dtype=float)

        # Loop over all hyperparameter configurations
        for batch, window, correlation, similarity in product(
            self.batches, self.windows, self.correlations, self.similarities):

            # Loop over various K-folds
            for k, (X_, y_) in enumerate(self.splitter.k_fold_label(X, y, self.k, max_labels=max_labels)):

                # Log current evaluation
                self.logger.info("Processing ({}, {}, {}, {}, {})".format(
                    batch, window, correlation, similarity, k))

                # Run single evaluation
                eval = self.evaluator.run(X_, y_, batch, window,
                                          correlation, similarity)
                # Get single evaluation entry
                entry = {
                    'batch'       : batch,
                    'window'      : window,
                    'correlation' : correlation,
                    'similarity'  : similarity,
                    'k'           : k,
                    'support'     : X_.shape[0],
                    'accuracy'    : eval.get('accuracy'),
                    'precision'   : eval.get('precision'),
                    'recall'      : eval.get('recall'),
                    'f1-score'    : eval.get('f1-score'),
                    'homogeneity' : eval.get('homogeneity'),
                    'completeness': eval.get('completeness'),
                    'v-measure'   : eval.get('v-measure'),
                    'ari'         : eval.get('ari')
                }
                # Add entry to dataframe
                df = df.append(entry, ignore_index=True)

            # Save if fragmented file given
            if fragmented is not None:
                self.write(df, fragmented)

        # Return dataframe
        return df

    def get_optimal(self, df, name, parameters, verbose=True):
        """Get optimal value of given parameter

            Parameters
            ----------
            df : pd.DataFrame
                Dataframe over which to compute optimal value.

            name : string
                Name of value for which to optimise.

            parameters : iterable
                Values of parameters over which we optimise.

            verbose : boolean, default=True
                If true, print optimal value that was found.

            Returns
            -------
            result : float
                Optimal value.
            """
        # Extract df_batch results
        df_test = df.tail(len(parameters)*self.k)
        # Average over all K folds
        df_test = self.average(df_test)
        # Get batch with maximum F1-score
        index = df_test['f1-score'].idxmax()
        # Set default_window
        result = df_test.loc[index, name]

        # If verbose, print result
        if verbose:
            print("Optimal {} = {}".format(name, result))

        # Return result
        return result

    def average(self, df):
        """Compute average performance by averaging all k-folds.

            Parameters
            ----------
            df : pandas.DataFrame
                Output of self.run or loaded dataframe.

            Returns
            -------
            result : pandas.DataFrame
                Values averaged by aggregating k-folds.
            """
        # Initialise dataframe
        result = pd.DataFrame(columns=['batch', 'window', 'correlation',
                              'similarity', 'support', 'accuracy', 'precision',
                              'recall', 'f1-score', 'homogeneity',
                              'completeness', 'v-measure', 'ari'], dtype=float)

        for (batch, window, correlation, similarity), frame in df.groupby(
            ['batch', 'window', 'correlation', 'similarity']):
            # Compute total support
            support = frame['support'].sum()

            # Create single entry
            entry = {
                'batch'       : batch,
                'window'      : window,
                'correlation' : correlation,
                'similarity'  : similarity,
                'support'     : support,
                'accuracy'    : (frame['accuracy'    ] * frame['support']).sum(
                                ) / support,
                'precision'   : (frame['precision'   ] * frame['support']).sum(
                                ) / support,
                'recall'      : (frame['recall'      ] * frame['support']).sum(
                                ) / support,
                'f1-score'    : (frame['f1-score'    ] * frame['support']).sum(
                                ) / support,
                'homogeneity' : (frame['homogeneity' ] * frame['support']).sum(
                                ) / support,
                'completeness': (frame['completeness'] * frame['support']).sum(
                                ) / support,
                'v-measure'   : (frame['v-measure'   ] * frame['support']).sum(
                                ) / support,
                'ari'         : (frame['ari'         ] * frame['support']).sum(
                                ) / support
            }
            # Add entry to result
            result = result.append(entry, ignore_index=True)

        # Return result
        return result

    def write(self, result, outfile):
        """Write result to csv file.

            Parameters
            ----------
            result : pandas.DataFrame
                Output of self.run

            outfile : string
                File to write output to.
            """
        # Write to output file
        result.to_csv(outfile)

    def read(self, infile):
        """Read analysis from csv file.

            Parameters
            ----------
            infile : string
                Input file from which to read csv.

            Returns
            -------
            result : pandas.DataFrame
                Dataframe containing the read analysis.
            """
        return pd.read_csv(infile)


if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint evaluator.')
    parser.add_argument('-f', '--files', nargs='+',
                        help='pcap files to run through FlowPrint. We use the '
                             'directory of each file as label.')
    parser.add_argument('-a', '--save',
                        help='Save preprocessed data to given file.')
    parser.add_argument('-l', '--load', nargs='+',
                        help='load preprocessed data from given file.')

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                            Setup logging                             #
    ########################################################################

    logging.basicConfig(filename='hyperparameters.log',
                        filemode='a',
                        datefmt='%Y-%m-%d %H:%M%S.%f',
                        level=logging.DEBUG)

    ########################################################################
    #                          Preprocessing step                          #
    ########################################################################

    # Initialise preprocessor
    preprocessor = Preprocessor(verbose=True)

    # Parse files if required
    if args.files:
        # Extract files and labels
        files  = args.files
        labels = extract_labels(args.files)

        # Load data
        X, y = preprocessor.process(files, labels)

        # Save preprocessed data if necessary
        if args.save:
            preprocessor.save(args.save, X, y)

    # Load data if necessary
    if args.load:
        if any('andrubis' in x.lower() for x in args.load):
            loader = Loader('../../data/Andrubis/appdict.p')
        else:
            loader = Loader()

        X, y = loader.load_files(args.load)

    logging.info("Data loaded")

    ########################################################################
    #                            Run Evaluation                            #
    ########################################################################

    # Create evaluator
    hyperparameters = Hyperparameters()
    analysis = hyperparameters.run_quick(X, y, fragmented='fragmented.csv', max_labels=200)
    hyperparameters.write(analysis, 'analysis.csv')
    analysis = hyperparameters.read('analysis.csv')
    average = hyperparameters.average(analysis)
    hyperparameters.write(average, 'average.csv')
