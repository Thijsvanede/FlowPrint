from datetime import datetime
import ipaddress

################################################################################
#                              Single Flow object                              #
################################################################################

class Flow(object):
    """Flow object extracted from pcap file that can be used for fingerprinting

        Attributes
        ----------
        src : string
            Source IP

        sport : int
            Source port

        dst : string
            Destination IP

        dport : int
            Destination port

        source : tuple
            (Source IP, source port) tuple

        destination : tuple
            (Destination IP, destination port) tuple

        certificate : Object
            Certificate of flow, if any

        certificates : set
            Set of TLS certificates in flow

        lengths : list
            List of packet length for each packet in flow

        timestamps : list
            List of timestamps corresponding to each packet in flow
    """

    def __init__(self):
        """Initialise an empty Flow."""
        # Initialise flow endpoints
        self.src   = None
        self.sport = None
        self.dst   = None
        self.dport = None

        self.certificates = set()

        self.lengths    = list()
        self.timestamps = list()

    def add(self, packet):
        """Add a new packet to the flow.

            Parameters
            ----------
            packet : np.array of shape=(n_features,)
                Packet from Reader.
            """
        # Extract endpoints from packet
        ip_a  , ip_b   =     packet[5],      packet[6]
        port_a, port_b = int(packet[7]), int(packet[8])

        # Perform packet check
        if self.src is not None:
            if {self.src, self.dst} != {ip_a, ip_b} and {self.sport, self.dport} != {port_a, port_b}:
                raise ValueError("Packet {} incompatible with flow {}" .format(packet, self))
        # Set endpoints where smallest dport is destination
        elif port_a > port_b:
            self.src  , self.dst   = ip_a  , ip_b
            self.sport, self.dport = port_a, port_b
        else:
            self.src  , self.dst   = ip_b  , ip_a
            self.sport, self.dport = port_b, port_a

        # Add certificate if any
        if packet[9] is not None:
            self.certificates.add(packet[9])

        # Perform equivalence checks
        if len(self.certificates) > 1:
            raise ValueError("More than 1 certificate found: {}"
                             .format(self.certificates))

        # Set timestamps and lengths
        self.timestamps.append(float(packet[3]))
        self.lengths   .append( int(packet[4]) if packet[5] == self.src else
                               -int(packet[4]))

        # Return self
        return self

    ########################################################################
    #                    Source/Destination attributes                     #
    ########################################################################

    @property
    def source(self):
        """(source IP, source port)-tuple of Flow"""
        return (self.src, self.sport)

    @property
    def destination(self):
        """(destination IP, destination port)-tuple of Flow"""
        return (self.dst, self.dport)

    @property
    def certificate(self):
        """Certificate of Flow"""
        return list(self.certificates)[0] if self.certificates else None

    @property
    def time_start(self):
        """Returns start time of Flow."""
        return min(self.timestamps)

    @property
    def time_end(self):
        """Returns end time of Flow."""
        return max(self.timestamps)

    ########################################################################
    #                           Class overrides                            #
    ########################################################################

    def __len__(self):
        """Return length of Flow in packets."""
        return len(self.lengths)

    def __str__(self):
        """Return string representation of flow."""
        return "[Time {} to {}] {:>15}:{:<5} <-> {:>15}:{:<5} [Length {}]".format(
            datetime.fromtimestamp(min(self.timestamps)).strftime("%H:%M:%S.%f"),
            datetime.fromtimestamp(max(self.timestamps)).strftime("%H:%M:%S.%f"),
            self.src, self.sport, self.dst, self.dport,
            len(self))

    def __gt__(self, other):
        """Greater than object override"""
        return min(self.timestamps) >  min(other.timestamps)

    def __ge__(self, other):
        """Greater equals object override"""
        return min(self.timestamps) >= min(other.timestamps)

    def __lt__(self, other):
        """Less than object override"""
        return min(self.timestamps) <  min(other.timestamps)

    def __le__(self, other):
        """Less equals object override"""
        return min(self.timestamps) <= min(other.timestamps)
