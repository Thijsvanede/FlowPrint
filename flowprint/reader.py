from cryptography import x509
from cryptography.hazmat.backends import default_backend
import glob
import numpy as np
import os
import pyshark
import warnings
from subprocess import Popen, PIPE

class Reader(object):
    """Reader object for extracting features from .pcap files

        Attributes
        ----------
        verbose : boolean
            Boolean indicating whether to be verbose in reading
    """

    ########################################################################
    #                         Class initialisation                         #
    ########################################################################

    def __init__(self, verbose=False):
        """Reader object for extracting features from .pcap files

            Parameters
            ----------
            verbose : boolean, default=False
                Boolean indicating whether to be verbose in reading
            """
        # Set verbosity level
        self.verbose = verbose

    ########################################################################
    #                             Read method                              #
    ########################################################################

    def read(self, path):
        """Read TCP and UDP packets from .pcap file given by path.
            Automatically choses fastest available backend to use.

            Parameters
            ----------
            path : string
                Path to .pcap file to read.

            Returns
            -------
            result : np.array of shape=(n_packets, n_features)
                Where features consist of:

                0) Filename of capture
                1) Protocol TCP/UDP
                2) TCP/UDP stream identifier
                3) Timestamp of packet
                4) Length of packet
                5) IP packet source
                6) IP packet destination
                7) TCP/UDP packet source port
                8) TCP/UDP packet destination port
                9) SSL/TLS certificate if exists, else None

            Warning
            -------
            warning
                Method throws warning if tshark is not available.
            """

        # If verbose, print which file is currently being read
        if self.verbose:
            print("Reading {}...".format(path))

        # Check if we can use fast tshark read or slow pyshark read
        try:
            return self.read_tshark(path)
        except Exception as ex:
            warnings.warn("tshark error: '{}', defaulting to pyshark backend. "
                          "note that the pyshark backend is much slower than "
                          "the tshark backend."
                          .format(ex))
            return self.read_pyshark(path)


    def read_tshark(self, path):
        """Read TCP and UDP packets from file given by path using tshark backend

            Parameters
            ----------
            path : string
                Path to .pcap file to read.

            Returns
            -------
            result : np.array of shape=(n_packets, n_features)
                Where features consist of:

                0) Filename of capture
                1) Protocol TCP/UDP
                2) TCP/UDP stream identifier
                3) Timestamp of packet
                4) Length of packet
                5) IP packet source
                6) IP packet destination
                7) TCP/UDP packet source port
                8) TCP/UDP packet destination port
                9) SSL/TLS certificate if exists, else None
            """
        # Create Tshark command
        command = ["tshark", "-r", path, "-Tfields",
                   "-e", "frame.time_epoch",
                   "-e", "tcp.stream",
                   "-e", "udp.stream",
                   "-e", "ip.proto",
                   "-e", "ip.src",
                   "-e", "tcp.srcport",
                   "-e", "udp.srcport",
                   "-e", "ip.dst",
                   "-e", "tcp.dstport",
                   "-e", "udp.dstport",
                   "-e", "ip.len",
                   "-e", "ssl.handshake.certificate"]
        # Initialise result
        result = list()

        # Call Tshark on packets
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        # Get output
        out, err = process.communicate()

        # Give warning message if any
        if err:
            warnings.warn("Error reading file: '{}'".format(
                err.decode('utf-8')))

        # Read each packet
        for packet in filter(None, out.decode('utf-8').split('\n')):
            # Get all data from packets
            packet = packet.split()

            # Perform check on packets
            if len(packet) < 8: continue

            # Perform check on multiple ip addresses
            packet[3] = packet[3].split(',')[0]
            packet[5] = packet[5].split(',')[0]
            packet[7] = packet[7].replace(',', '')

            # Parse certificate
            if len(packet) > 8:
                # Get first certificate
                cert = packet[8].split(',')[0]
                # Transform to hex
                cert = bytes.fromhex(cert.replace(':', ''))
                # Read as certificate
                cert = x509.load_der_x509_certificate(cert, default_backend())
                # Set packet as serial number
                packet[8] = cert.serial_number
            else:
                packet.append(None)

            # Add packet to result
            result.append([path] + packet)

        # Get result as numpy array
        result = np.asarray(result)

        # Check if any items exist
        if not result.shape[0]:
            return np.zeros((0, 8), dtype=object)

        # Change protocol number to text
        protocols = {'17': 'udp', '6': 'tcp'}
        result[:, 3] = [protocols.get(x, 'unknown') for x in result[:, 3]]

        # Return in original order
        return result[:, [0, 3, 2, 1, 8, 4, 6, 5, 7, 9]]


    def read_pyshark(self, path):
        """Read TCP and UDP packets from file given by path
            using pyshark backend

            Parameters
            ----------
            path : string
                Path to .pcap file to read.

            Returns
            -------
            result : np.array of shape=(n_packets, n_features)
                Where features consist of:

                0) Filename of capture
                1) Protocol TCP/UDP
                2) TCP/UDP stream identifier
                3) Timestamp of packet
                4) Length of packet
                5) IP packet source
                6) IP packet destination
                7) TCP/UDP packet source port
                8) TCP/UDP packet destination port
                9) SSL/TLS certificate if exists, else None
            """
        # If verbose, print which file is currently being read
        if self.verbose:
            counter_a = 0
            counter_b = 0

        # Read pcap file
        pcap = iter(pyshark.FileCapture(path))

        # Initialise result
        result = list()

        # Loop over packets
        while True:
            try:
                packet = next(pcap)
            except Exception as ex:
                warnings.warn("Pyshark error: '{}'".format(ex))
                break

            if not ("TCP" in packet or "UDP" in packet):
                counter_b += 1
                continue

            if self.verbose:
                counter_a += 1
                counter_b += 1
                print("Reading {}... {}/{} packets".format(path, counter_a, counter_b), end='\r')

            # Get required packet data
            d = [path,
                 packet.layers[2].layer_name, # Get
                 packet.layers[2].stream,     # Get stream ID
                 packet.sniff_timestamp,      # Get packet timestamp
                 packet.length,               # Get packet length
                 packet.layers[1].src,        # Get source IP or IPv6 (fixed)
                 packet.layers[1].dst,        # Get destination IP or IPv6 (fixed)
                 packet.layers[2].srcport,    # Get source port
                 packet.layers[2].dstport,    # Get destination port
                 None]

            # Check whether SSL/TLS certificate is in packet
            if "SSL" in packet and\
               packet.ssl.get("handshake_certificate") is not None:
                # Get certificate
                cert = packet.ssl.get('handshake_certificate')
                # Parse cert to bytes
                cert = bytes.fromhex(cert.replace(':', ''))
                # Parse x509 certificate as DER
                cert = x509.load_der_x509_certificate(cert,
                                                      default_backend())
                # Get serial number - TODO extend with other features?
                d[-1] = cert.serial_number

            # Append data item to result
            result.append(d)

        # Close capture
        pcap.close()

        if self.verbose:
            print()

        # Return result as numpy array
        return np.array(result)
