import argparse
import numpy as np
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='pickled files to combine.')
    args = parser.parse_args()

    result_X = list()
    result_y = list()

    for infile in args.files:
        print(infile, end=' ')
        with open(infile, 'rb') as infile:
            X, y = pickle.load(infile)
            result_X.append(X)
            result_y.append(y)
            print(X.shape)

    X = np.concatenate(result_X)
    y = np.concatenate(result_y)

    print(X.shape)
    print(y.shape)

    with open('recon_full.p', 'wb') as outfile:
        pickle.dump((X, y), outfile)
