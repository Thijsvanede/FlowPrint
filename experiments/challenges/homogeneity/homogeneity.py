import argparse
import code
import csv
import numpy as np
import pandas as pd
import ipaddress

from collections import Counter
from pprint import pprint

from os import path
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '../../../flowprint')))
from fpio import IO
from cluster import Cluster


def create_dns_dict(infile):
    # Initialise result
    result = dict()

    # Open file
    with open(infile, 'r') as infile:
        # Create csv reader
        reader = csv.reader(infile)

        # Loop over all lines
        for line in reader:
            # Parse each line
            app          = line[0]
            dns          = line[1]
            dns_extended = line[2]
            label        = line[3]

            # Do not parse first line
            if app == "package":
                continue

            # Remove port number from dns_extended
            if dns_extended.endswith(":80"):
                dns_extended = dns_extended[:-3]
            if dns_extended.endswith(":443"):
                dns_extended = dns_extended[:-4]

            try:
                ip = ipaddress.ip_address(dns_extended)
                dns_extended = dns
            except ValueError:
                pass

            # Check if comain is correct
            if not dns_extended.endswith(dns):
                raise ValueError("Could not correctly identify domain")

            # Discect DNS
            dns = dns_extended.split(".")[::-1]

            # Add entry to result
            prev    = None
            current = result
            for i, subdomain in enumerate(dns):
                entry = current.get(subdomain, dict())
                current[subdomain] = entry
                prev = current
                current = entry

            # Add final entry
            current["leaf"] = current.get("leaf", set()) | set([(app, label)])

    # Return result
    return result

def query_dns(dictionary, query):
    """Query in DNS dictionary"""
    # Check if query is None
    if query is None:
        return None

    # Initialise result
    result = dict()
    # Transform query
    query = query.split(".")[::-1]

    # Perform query
    current = dictionary
    for i, subdomain in enumerate(query):
        try:
            current = current[subdomain]
        except:
            return None

    if 'leaf' in current:
        return current["leaf"]
    else:
        return None




if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create argument parser
    parser = argparse.ArgumentParser(description='Flowprint IP-TCP analyser.')
    parser.add_argument('-f', '--files', nargs='+',
                        help='pcap files to run through FlowPrint. We use the '
                             'directory of each file as label.')
    parser.add_argument('-l', '--load', nargs='+',
                        help='load preprocessed data from given file.')
    parser.add_argument('-p', '--min-flows', type=int, default=1,
                        help='Minimum number of flows required per application '
                         '(default=1).')

    # Parse given arguments
    args = parser.parse_args()

    ########################################################################
    #                          Preprocessing step                          #
    ########################################################################

    # First create dictionary of dns -> (app, type)
    dns_dict = create_dns_dict("../../../../../data/labels/labeled_pkg_tld.csv")

    io_module = IO()

    if args.load:
        # Store flows
        X = list()
        y = list()

        # Get preprocessed flows
        for file in args.load:
            X_, y_ = io_module.read_flows(file)
            X.extend(X_)
            y.extend(y_)

        # Transform labels
        y = [label.split("/")[-2] for label in y]

        domains = dict()
        for flow, label in zip(X, y):
            domains[flow.domain] = domains.get(flow.domain, set()) | set([label])

        # for domain in sorted({'googleads.g.doubleclick.net', 'play.googleapis.com', 'android.clients.google.com', 'lh4.googleusercontent.com', 's.yimg.com', 'infoc2.duba.net'}):
        #     print(domain, domains[domain])
        # exit()

        # Cluster flows
        cluster = Cluster()
        cluster.fit(X, y)

        # Get dictionary of cluster -> type
        cluster_types = dict()

        # Loop over all clusters
        for c in sorted(cluster.get_clusters()):
            # Get most likely type
            c_type = Counter()
            # Loop over each flow in cluster
            for flow in c.samples:
                # Perform query on DNS
                result = query_dns(dns_dict, flow.domain)
                # If result is None skip
                if result is None:
                    continue

                # Compute weighted result
                result_weighted = Counter()
                for a, b in result:
                    if a in c.labels.keys():
                        result_weighted[b] = result_weighted.get(b, 0) + c.labels[a]

                # Get possible types from query and increment counter
                if result_weighted:
                    c_type.update([result_weighted.most_common(1)[0][0]])

            # Set final type of cluster
            c_type = c_type.most_common(1)[0][0] if c_type else '1'
            # Check if type should be DNS
            if all(d[1] == 53 for d in c.destinations):
                c_type = '5'
            # Set cluster type
            cluster_types[c] = c_type

        # For each cluster, check type and whether cluster is shared among apps
        result_single = dict()
        result_shared = dict()
        td = {0: "unknown", 1: "first", 2: "content", 3: "ads", 4: "social", 5: "dns"}
        for c, t in cluster_types.items():
            # Get labels
            if len(c.labels) > 1:
                result_shared[t] = result_shared.get(t, set()) | set([c])
            else:
                result_single[t] = result_single.get(t, set()) | set([c])

        shared_apps = set()
        for x in result_shared.values():
            for c in x:
                shared_apps |= set(c.labels.keys())

        print("Total        apps        : {:>8}".format(len(set(y))))
        print("Total shared apps        : {:>8}".format(len(shared_apps)))
        print("Total        clusters    : {:>8}".format(len(cluster_types)))
        print("Total shared clusters    : {:>8}".format(sum([len(x) for x in result_shared.values()])))
        print("Total        flows       : {:>8}".format(len(X)))
        print("Total shared flows       : {:>8}".format(sum([sum(len(c.samples) for c in x) for x in result_shared.values()])))
        print()

        print("Statistics non-shared clusters")
        print("---------------------------------")
        for i in range(1, 6):
            print("Total shared clusters '{}': {:10.4f} ({})".format(i, len(result_single.get(str(i), set())), td[i]))
            labels  = np.asarray([len(r.labels) for r in result_single.get(str(i), set())], dtype=int)
            labelss = set()
            for c in result_single.get(str(i), set()):
                for app in c.labels.keys():
                    labelss.add(app)
            print("  total apps             : {:10.4f}".format(len(labelss)))
            try:
                print("    avg apps  per cluster: {:10.4f}".format(labels.mean()))
                print("    min apps  per cluster: {:10.4f}".format(labels.min()))
                print("    25% apps  per cluster: {:10.4f}".format(np.percentile(labels, 25)))
                print("    50% apps  per cluster: {:10.4f}".format(np.percentile(labels, 50)))
                print("    75% apps  per cluster: {:10.4f}".format(np.percentile(labels, 75)))
                print("    max apps  per cluster: {:10.4f}".format(labels.max()))
            except:
                print("Error")
            print("---------------------------------")
            flows  = np.asarray([len(r.samples) for r in result_single.get(str(i), set())], dtype=int)
            flowss = 0
            for c in result_single.get(str(i), set()):
                flowss += len(c.samples)
            print("  total flows            : {:10.4f}".format(flowss))
            try:
                print("    avg flows per cluster: {:10.4f}".format(flows.mean()))
                print("    min flows per cluster: {:10.4f}".format(flows.min()))
                print("    25% flows per cluster: {:10.4f}".format(np.percentile(flows, 25)))
                print("    50% flows per cluster: {:10.4f}".format(np.percentile(flows, 50)))
                print("    75% flows per cluster: {:10.4f}".format(np.percentile(flows, 75)))
                print("    max flows per cluster: {:10.4f}".format(flows.max()))
            except:
                print("Error")
            print("---------------------------------")
            print()


        print("Statistics shared clusters")
        print("---------------------------------")
        for i in range(1, 6):
            print("Total shared clusters '{}': {:10.4f} ({})".format(i, len(result_shared.get(str(i), set())), td[i]))
            labels  = np.asarray([len(r.labels) for r in result_shared.get(str(i), set())], dtype=int)
            labelss = set()
            for c in result_shared.get(str(i), set()):
                for app in c.labels.keys():
                    labelss.add(app)
            print("  total apps             : {:10.4f}".format(len(labelss)))
            try:
                print("    avg apps  per cluster: {:10.4f}".format(labels.mean()))
                print("    min apps  per cluster: {:10.4f}".format(labels.min()))
                print("    25% apps  per cluster: {:10.4f}".format(np.percentile(labels, 25)))
                print("    50% apps  per cluster: {:10.4f}".format(np.percentile(labels, 50)))
                print("    75% apps  per cluster: {:10.4f}".format(np.percentile(labels, 75)))
                print("    max apps  per cluster: {:10.4f}".format(labels.max()))
            except:
                print("Error")
            print("---------------------------------")
            flows  = np.asarray([len(r.samples) for r in result_shared.get(str(i), set())], dtype=int)
            flowss = 0
            for c in result_shared.get(str(i), set()):
                flowss += len(c.samples)
            print("  total flows            : {:10.4f}".format(flowss))
            try:
                print("    avg flows per cluster: {:10.4f}".format(flows.mean()))
                print("    min flows per cluster: {:10.4f}".format(flows.min()))
                print("    25% flows per cluster: {:10.4f}".format(np.percentile(flows, 25)))
                print("    50% flows per cluster: {:10.4f}".format(np.percentile(flows, 50)))
                print("    75% flows per cluster: {:10.4f}".format(np.percentile(flows, 75)))
                print("    max flows per cluster: {:10.4f}".format(flows.max()))
            except:
                print("Error")
            print("---------------------------------")
            print()

        # colour_dict = {0: 'yellow', 1: 'black', 2: 'blue', 3: 'red', 4: 'green', 5: 'cyan'}
        # colours = {c: colour_dict[int(t)] for c, t in cluster_types.items()}
        # cluster.plot_clusters(annotate=True, annotate_cap=100, colors=colours, random_state=43)
