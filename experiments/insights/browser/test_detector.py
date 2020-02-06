from browser_detector import BrowserDetector
from collections import Counter
import numpy as np
import pickle

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from cluster_optimised import ClusterOptimised
from loader import Loader
from splitter import Splitter
from utils import get_indices

if __name__ == "__main__":

    ########################################################################
    #                              Load data                               #
    ########################################################################

    # Load data
    loader = Loader(verbose=True)
    cols = ['file', 'timestamp',
            "TLS certificate issuer name",
            "TLS certificate signature",
            "TLS certificate subject name",
            "TLS certificate subject pk",
            "TLS certificate subject pk algorithm",
            "TLS certificate version",
            'IP dst', 'TCP dport',
            'TCP length incoming mean', 'IP packets incoming',
            'TCP length outgoing mean', 'IP packets outgoing']
    X_app, columns_app = loader.load("../../data/Andrubis/processed/2/", cols)
    # X_app, columns_app = loader.load("../../data/ReCon/depricated/processed/", cols)
    X_bro, columns_bro = loader.load("./alexa/data/samsung*.p", cols)

    # Get columns
    columns = columns_app
    column  = get_indices(columns)

    # Create labels for applications
    y_app = np.array([x.split('/')[-2] for x in X_app[:, column['file']]])
    y_bro = np.array(['browser']*X_bro.shape[0])

    # ########################################################################
    # # In case of Andrubis dataset get labels differently -- UNCOMMENT!
    # # Load labels
    # with open('../../data/Andrubis/appdict.p', 'rb') as infile:
    #     app_data = pickle.load(infile)
    # # Compute labels
    # y_app = np.array([str(app_data.get(x, [None, None, None])[0]) for x in y_app])
    # ########################################################################

    # Create timestamps for applications
    ts_app = X_app[:, column['timestamp']]
    ts_bro = X_bro[:, column['timestamp']]

    ########################################################################
    #                              Evaluation                              #
    ########################################################################

    # Initialise metrics
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    resurfaced    = 0
    removed       = 0
    removed_t     = 0
    c_browser     = 0
    c_non_browser = 0

    # Set parameters
    k = 10   # For k-fold cross validation
    n = 5000 # Number of selected applications

    ########################################################################
    #                     Randomly select applications                     #
    ########################################################################

    # Get all applications
    applications = np.unique([x for x in y_app if x != "None"])

    # Do this k times
    for rs in range(k):
        # Create random state
        rs = np.random.RandomState(rs)

        # Select 5k applications at random
        try:
            selected = rs.choice(applications, n, replace=False)
        except:
            selected = applications
            print("Warning: Could only select {} "
                  "applications.".format(selected.shape[0]))

        # Create mask for selected apps
        mask_selected = np.isin(y_app, selected)
        # Select corresponding data, labels and timestamps
        X_app_s  = X_app[mask_selected]
        y_app_s  = y_app[mask_selected]
        ts_app_s = ts_app[mask_selected]

        # Set timestamps to fall within same range
        ts_app_s = ts_app_s - min(ts_app_s)
        ts_bro   = ts_bro   - min(ts_bro  )

        # Split data
        splitter = Splitter(random_state=42)
        ita, ika, iua = splitter.split_temporal(X_app_s, y_app_s, ts_app_s,
                                                size_known=1/10, size_unknown=1/10)
        itb, ikb, iub = splitter.split_temporal(X_bro, y_bro, ts_bro,
                                                size_known=1/2, size_unknown=0)

        # ita, ika, iua = splitter.split(X_app_s, y_app_s, ts_app_s)
        # itb, ikb, iub = splitter.split(X_bro, y_bro, ts_bro,
        #                                         size_known=1/2, size_unknown=0)

        # Create train and test sets
        X_train     = np.concatenate(( X_app_s[ita],  X_bro[itb]))
        y_train     = np.concatenate(( y_app_s[ita],  y_bro[itb]))
        ts_train    = np.concatenate((ts_app_s[ita], ts_bro[itb]))

        # Known and unknown applications
        X_known    =  X_app_s[ika]
        y_known    =  y_app_s[ika]
        ts_known   = ts_app_s[ika]
        X_unknown  =  X_app_s[iua]
        y_unknown  =  y_app_s[iua]
        ts_unknown = ts_app_s[iua]

        # Browser
        X_b    =  X_bro[ikb]
        y_b    =  y_bro[ikb]
        ts_b   = ts_bro[ikb]

        ####################################################################
        #                       Cluster applications                       #
        ####################################################################

        # Create cluster class
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

        # Fit and predict cluster
        p_known   = cluster.fit_predict(np.concatenate(( X_train,  X_known)),
                                        np.concatenate((ts_train, ts_known)))
        p_unknown = cluster.fit_predict(np.concatenate(( X_train,  X_unknown)),
                                        np.concatenate((ts_train, ts_unknown)))
        p_browser = cluster.fit_predict(np.concatenate(( X_train,  X_b)),
                                        np.concatenate((ts_train, ts_b)))

        # Get predictions for tested data
        p_trk, p_tek = p_known  [:X_train.shape[0]], p_known  [X_train.shape[0]:]
        p_tru, p_teu = p_unknown[:X_train.shape[0]], p_unknown[X_train.shape[0]:]
        p_trb, p_teb = p_browser[:X_train.shape[0]], p_browser[X_train.shape[0]:]

        # Rewrite predictions to True if new cluster formed or False otherwise
        p_known   = np.logical_and(p_tek != -1, ~np.isin(p_tek, np.unique(p_trk)))
        p_unknown = np.logical_and(p_teu != -1, ~np.isin(p_teu, np.unique(p_tru)))
        p_browser = np.logical_and(p_teb != -1, ~np.isin(p_teb, np.unique(p_trb)))

        ####################################################################
        #                    Perform Browser Detection                     #
        ####################################################################

        # Run browser detector on test data
        bd = BrowserDetector(column)
        bd.fit(bd.features(X_train), y_train == 'browser')

        # Run browser detector to get predictions True if browser, False otherwise
        bka = bd.is_browser(ts_known  , bd.features(X_known))
        bua = bd.is_browser(ts_unknown, bd.features(X_unknown))
        bb  = bd.is_browser(ts_b      , bd.features(X_b))

        ####################################################################
        #                   Increment evaluation metrics                   #
        ####################################################################
        tp += bb.sum()
        fp += bka.sum() + bua.sum()
        fn += bb.shape[0] - bb.sum()
        tn += bka.shape[0] + bua.shape[0] - (bka.sum() + bua.sum())

        # TODO make nicer for computing
        bkc = Counter(p_tek[bka])
        buc = Counter(p_teu[bua])

        tekc = Counter(p_tek)
        teuc = Counter(p_teu)

        # Add removed clusters
        rem = sum(tekc.get(c) == count for c, count in bkc.items())
        removed    += rem
        resurfaced += len(bkc) - rem

        rem += sum(teuc.get(c) == count for c, count in buc.items())
        removed    += rem
        resurfaced += len(bkc) - rem

        c_non_browser += np.unique(p_tek).shape[0] + np.unique(p_teu).shape[0]
        c_browser     += np.unique(p_teb).shape[0]

        print("Current results: "
              "TP={}, TN={}, FP={}, FN={}, REM={}, RES={}, CB={}, CNB={}"
              .format(tp, tn, fp, fn, removed, resurfaced,
                      c_browser, c_non_browser))

    ####################################################################
    #                 Print complete evaluation metrics                #
    ####################################################################
    print("Analysis report Browser Detector Performance")
    print("------------------------------------------------------------")

    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)
    fpr = fp / (tn+fp)
    fnr = fn / (tp+fn)
    acc = (tp + tn) / (tp+tn+fp+fn)

    print("""
                 | browser | background
    -----------------------------------
        detected | {:>7} | {:>10}
    not detected | {:>7} | {:>10}

    TPR = {:.4f}
    TNR = {:.4f}
    FPR = {:.4f}
    FNR = {:.4f}
    ACC = {:.4f}
""".format(tp, fp, fn, tn, tpr, tnr, fpr, fnr, acc))

    ####################################################################
    #                    Print cluster resurfacing                     #
    ####################################################################

    print("Analysis report cluster resurfacing")
    print("------------------------------------------------------------")
    print("    {:4} / {:4} = {:.2f}% of removed clusters resurfaced.\n".format(
          resurfaced, removed+resurfaced, 100*resurfaced/(removed+resurfaced)))
    print()

    ####################################################################
    #                     Print number of clusters                     #
    ####################################################################

    print("Analysis report total number of clusters")
    print("------------------------------------------------------------")
    print("    {} total browser clusters.".format(c_browser))
    print("    {} total non-browser clusters.".format(c_non_browser))

    exit()

    # # Test if there is at least 1 application flow that gets detected as non-browser
    # y_known_s   = set(y_known)
    # y_unknown_s = set(y_unknown)
    # for x in set(y_known[np.logical_and(~bka, p_known)]):
    #     if x not in y_known_s:
    #         print(x, "remained undetected!!!")
    # print()
    # for x in set(y_unknown[np.logical_and(~bua, p_unknown)]):
    #     if x not in y_unknown_s:
    #         print(x, "remained undetected!!!")
    # print('Analysed all applications.\n')


    ########################################################################
    #                     Browser detector performance                     #
    ########################################################################
    from sklearn.metrics import classification_report

    print("Analysis report Browser Detector Performance")
    print("------------------------------------------------------------")

    y_pred = np.concatenate((bka, bua, bb))
    y_test = np.array([0]*bka.shape[0] + [0]*bua.shape[0] + [1]*bb.shape[0])
    print(classification_report(y_test, y_pred))

    tp = bb.sum()
    fp = bka.sum() + bua.sum()
    fn = bb.shape[0] - tp
    tn = bka.shape[0] + bua.shape[0] - fp

    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)
    fpr = fp / (tn+fp)
    fnr = fn / (tp+fn)
    acc = (tp + tn) / (tp+tn+fp+fn)

    print("""
                 | browser | background
    -----------------------------------
        detected | {:>7} | {:>10}
    not detected | {:>7} | {:>10}

    TPR = {:.4f}
    TNR = {:.4f}
    FPR = {:.4f}
    FNR = {:.4f}
    ACC = {:.4f}
""".format(tp, fp, fn, tn, tpr, tnr, fpr, fnr, acc))

    ########################################################################
    #                Browser detector + novelty performance                #
    ########################################################################
#     print("Analysis report Browser Novelty Detector Performance")
#     print("------------------------------------------------------------")
#
#     y_pred = np.concatenate((bka[p_known], bua[p_unknown], bb[p_browser]))
#     y_test = np.array([0]*p_known.sum() + [0]*p_unknown.sum() + [1]*p_browser.sum())
#     print(classification_report(y_test, y_pred))
#
#     # Get evaluation metrics
#     tp = bb[p_browser].sum()
#     fp = bka[p_known].sum() + bua[p_unknown].sum()
#     fn = p_browser.sum() - tp
#     tn = p_known.sum() + p_unknown.sum() - fp
#
#     tpr = tp / (tp+fn)
#     tnr = tn / (tn+fp)
#     fpr = fp / (tn+fp)
#     fnr = fn / (tp+fn)
#     acc = (tp + tn) / (tp+tn+fp+fn)
#
#     print("""
#                  | browser | background
#     -----------------------------------
#         detected | {:>7} | {:>10}
#     not detected | {:>7} | {:>10}
#
#     TPR = {:.4f}
#     TNR = {:.4f}
#     FPR = {:.4f}
#     FNR = {:.4f}
#     ACC = {:.4f}
# """.format(tp, fp, fn, tn, tpr, tnr, fpr, fnr, acc))

    ########################################################################
    #                Resurfacing of wrongly removed clusters               #
    ########################################################################
    print("Analysis report cluster resurfacing")
    print("------------------------------------------------------------")

    # TODO make nicer
    # For computing
    from collections import Counter
    bkc = Counter(p_tek[bka])
    buc = Counter(p_teu[bua])

    tekc = Counter(p_tek)
    teuc = Counter(p_teu)

    resurfaced = 0
    removed    = 0

    for cluster, count in bkc.items():
        if tekc.get(cluster) - count == 0:
            removed += 1
        else:
            resurfaced += 1

    for cluster, count in buc.items():
        if teuc.get(cluster) - count == 0:
            removed += 1
        else:
            resurfaced += 1

    print("    {:4} / {:4} = {:.2f}% of removed clusters resurfaced.\n".format(
          resurfaced, removed+resurfaced, 100*resurfaced/(removed+resurfaced)))
