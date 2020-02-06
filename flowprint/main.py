from cc_graph import CrossCorrelationGraph
from collections import Counter
from cluster import Cluster
from fingerprints import Fingerprints
from loader import Loader
from preprocessor import Preprocessor
from statistics import Statistics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import completeness_score, homogeneity_score
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import LabelEncoder
from split import Splitter
import argparse
import numpy as np
import os
import pickle

def extract_labels(files):
    # Initialise result
    result = list()
    # Loop over all files
    for file in files:
        # Extract directory name
        result.append(os.path.split(os.path.dirname(file))[-1])
    # Return result
    return result

if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint.')
    parser.add_argument('-f', '--files', nargs='+', help='pcap files to run '
                                                   'through FlowPrint. We use '
                                                   'the  directory of each '
                                                   'file as label.')
    parser.add_argument('-s', '--save', help='Save preprocessed data to given '
                                       'file.')
    parser.add_argument('-l', '--load', nargs='+', help='load preprocessed data'
                                                        'from given file.')
    parser.add_argument('-r', '--random', type=int, help='Random state to use '
                                                 'for splitting data. '
                                                 '(default=42)')
    parser.add_argument('-t', '--test', type=float, help='Portion of data to be'
                                                   ' used for testing. All '
                                                   'other data is used for '
                                                   'training. (default=0.33)')
    parser.add_argument('-p', '--stats', action='store_true', help='If flag is '
                                                 'set, output data statistics.')

    # Parse given arguments
    args = parser.parse_args()

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
            exit()

    # Load data if necessary
    if args.load:
        if any('andrubis' in x.lower() for x in args.load):
            loader = Loader('../data/Andrubis/appdict.p')
        else:
            loader = Loader()

        X, y = loader.load_files(args.load)

    ########################################################################
    #                   Get data statistics if required                    #
    ########################################################################
    # Check if stats flag has been set
    if args.stats:

        # Check fingerprint stats
        with open('fingerprints.p', 'rb') as infile:
            fps, y = pickle.load(infile)

        # Create mapping
        mapping = dict()
        for fingerprint, label in zip(fps, y):
            mapping[fingerprint] = mapping.get(fingerprint, set()) | set([label])

        # Check uniqueness of fingerprints
        from itertools import combinations

        for fp1, fp2 in combinations(mapping.keys(), 2):
            score = fp1.compare(fp2)
            if score > 0.8:
                print(score)
                fp1.show()
                fp2.show()
                print('\n\n')

        with open('fingerprints_processed.p', 'wb') as outfile:
            pickle.dump((mapping, table), outfile)

        exit()


        # Create statistics object
        statistics = Statistics(X, y)
        statistics.app_detection_time(plot=True)
        print(statistics.plot_application_detection_time(500))
        print(statistics.report())

        # Output all statistics
        exit()

    ########################################################################
    #                  Remove low occurence applications                   #
    ########################################################################
    # from collections import Counter
    # occurences = Counter(y)
    #
    # indices = []
    # for app, count in occurences.items():
    #     if count >= 100:
    #         indices.extend(np.argwhere(y == app).flatten().tolist())
    #
    # X = X[indices]
    # y = y[indices]

    ########################################################################
    #                           Clustering step                            #
    ########################################################################

    # Create clustering object
    cluster = Cluster()

    ########################################################################
    #                         Fingerprinting step                          #
    ########################################################################

    # Create Fingerprinter
    fingerprints = Fingerprints(cluster)

    # Train and predict with fingerprinter
    y_pred_fingerprints = fingerprints.fit_predict_fingerprints(X, y)
    y_pred_labels       = np.array([fp.label for fp in y_pred_fingerprints])
    y_pred_fingerprints = np.array([hash(y_) for y_ in y_pred_fingerprints])

    ########################################################################
    #                               Evaluate                               #
    ########################################################################

    # Compute metrics
    accuracy     = accuracy_score     (y, y_pred_labels)
    homogeneity  = homogeneity_score  (y, y_pred_fingerprints)
    completeness = completeness_score (y, y_pred_fingerprints)
    v_measure    = v_measure_score    (y, y_pred_fingerprints)

    # Print metrics
    print(classification_report(y, y_pred_labels, digits=4))
    print("Accuracy    : {:.4f}".format(accuracy))
    print("Homogeneity : {:.4f}".format(homogeneity))
    print("Completeness: {:.4f}".format(completeness))
    print("V-Measure   : {:.4f}".format(v_measure))

    ########################################################################
    #                            Anomaly setup                             #
    ########################################################################

    # Create splitter instance
    splitter = Splitter(args.random or 42) # Default = 42 otherwise use random

    # Split in train, test_known, test_unknown sets
    X_train, y_train,\
    X_test_known, y_test_known,\
    X_test_unknown, y_test_unknown = splitter.split_normal(X, y, size_train=4/5)

    # Create Fingerprinter
    fingerprints = Fingerprints(cluster)

    # Train fingerprinter
    fp_train = fingerprints.fit_predict_fingerprints(X_train, y_train)

    # Create mappings of fingerprints for fast search
    # TODO: document
    fp_train_dest = dict()
    fp_train_cert = dict()
    for fp in fp_train:
        for destination in fp.destinations:
            dsts = fp_train_dest.get(destination, set())
            dsts.add(fp)
            fp_train_dest[destination] = dsts
        for certificate in fp.certificates:
            certs = fp_train_cert.get(certificate, set())
            certs.add(fp)
            fp_train_cert[certificate] = certs

    # Predict fingerprinter
    fp_test_known   = fingerprints.fit_predict_fingerprints(X_test_known)
    fp_test_unknown = fingerprints.fit_predict_fingerprints(X_test_unknown)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    threshold = 0.5

    count = 0
    length = len(set(fp_test_known))
    for test in set(fp_test_known):
        count += 1
        print("{}/{}".format(count, length), end='\r')

        possible = set()
        for destination in test.destinations:
            possible |= fp_train_dest.get(destination, set())
        for certificate in test.certificates:
            possible |= fp_train_cert.get(certificate, set())

        for train in possible:
            if test.compare(train) >= threshold:
                true_positives += np.sum(fp_test_known == test)
                break
        else:
            false_negatives += np.sum(fp_test_known == test)


    count = 0
    length = len(set(fp_test_unknown))
    for test in set(fp_test_unknown):
        count += 1
        print("{}/{}".format(count, length), end='\r')

        possible = set()
        for destination in test.destinations:
            possible |= fp_train_dest.get(destination, set())
        for certificate in test.certificates:
            possible |= fp_train_cert.get(certificate, set())

        for train in possible:
            if test.compare(train) >= threshold:
                false_positives += np.sum(fp_test_unknown == test)
                break
        else:
            true_negatives += np.sum(fp_test_unknown == test)

    print("TP: {}".format(true_positives))
    print("TN: {}".format(true_negatives))
    print("FP: {}".format(false_positives))
    print("FN: {}".format(false_negatives))

    # RESULTS RECON
    # TP : 3623
    # TN : 19818
    # FP : 0
    # FN : 2384
    # ACC: 0.9076863504356244

    # RESULTS ANDRUBIS 20*.p
    # TP : 80436
    # TN : 30676
    # FP : 28307
    # FN : 8209
    # ACC: 0.7526485490557346

    # RESULTS ANDRUBIS 20{1,2,3,4,5}.p
    # TP : 21289
    # TN : 13086
    # FP : 6486
    # FN : 2774
    # ACC: 0.7526485490557346
