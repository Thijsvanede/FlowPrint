from cryptography import x509
from cryptography.hazmat.backends import default_backend
from datetime import datetime
import ipaddress
import json

################################################################################
#                              Single Flow object                              #
################################################################################

class Flow(object):

    def __init__(self, load=dict(), check_dir=True):
        """Representation of a Flow.

            Parameters
            ----------
            load : dict(), optional
                Dictionary from which to load all flow values.

            check_dir : boolean, default=True
                If true, check direction of Flow and adapt accordingly.
            """
        # Initialise Flow variables
        self.protocol = None
        self.src = None
        self.dst = None
        self.sport = None
        self.dport = None

        self.lengths    = list()
        self.timestamps = list()
        self.direction  = list()

        self._certificates = set()
        self.domain        = None

        # If info, load from dict
        self.from_dict(load)

        # If check, perform checks:
        if check_dir and not self.check_dir():
            # Reverse flow direction if check failed
            self.reverse_dir()


    ########################################################################
    #                            Check methods                             #
    ########################################################################

    def check_valid(self):
        """Perform check on validity of object.

            Raises
            ------
            AssertionError
            """
        if len(self.lengths) != len(self.timestamps) or\
           len(self.lengths) != len(self.direction):
            raise AssertionError("Length of Lengths={}, Timestamp={} and "
                                 "Direction={} are not equal".format(
                                 len(self.lengths),
                                 len(self.timestamps),
                                 len(self.direction)))

    def check_dir(self):
        """Check if direction of flow is correct."""
        # Get source and destination IP
        src = ipaddress.ip_address(self.src)
        dst = ipaddress.ip_address(self.dst)

        # Check if destination is private and source is not private
        if dst.is_private and not src.is_private:
            return False
        if dst.is_private == src.is_private:
            return self.sport > self.dport
        return True


    ########################################################################
    #                             Add packets                              #
    ########################################################################

    def add(self, packet, certificate):
        """Add a new packet to the flow.

            Parameters
            ----------
            packet : np.array of shape=(n_features,)
                Packet from Reader.
            """
        # Add data to packet
        self.timestamps.append(float(packet.sniff_timestamp))
        self.lengths   .append(int(packet.length))
        self.direction .append(packet.layers[1].src == self.src and
                      int(packet.layers[2].srcport) == self.sport)

        # Add certificate if any
        if certificate is not None:
            self._certificates.add(certificate)


    ########################################################################
    #                           Class properties                           #
    ########################################################################

    @property
    def source(self):
        """Return source of Flow."""
        return (self.src, self.sport)

    @property
    def destination(self):
        """Return destination of Flow."""
        return (self.dst, self.dport)

    @property
    def certificate(self):
        """Returns certificate of Flow."""
        return list(self.certificates)[0] if self.certificates else None

    @property
    def start(self):
        """Returns timestamp of Flow."""
        return min(self.timestamps)

    @property
    def end(self):
        """Returns timestamp of Flow."""
        return max(self.timestamps)

    @property
    def certificates(self):
        """Returns certificates as cryptography.x509 Certificates."""
        return set([x509.load_der_x509_certificate(
                    bytes.fromhex(cert), default_backend())
                    for cert in self._certificates])

    ########################################################################
    #                    Method versions of properties                     #
    ########################################################################

    def certificate(self):
        """Returns certificate of Flow."""
        return list(self.certificates)[0] if self.certificates else None

    def destination(self):
        """Return destination of Flow."""
        return (self.dst, self.dport)


    ########################################################################
    #                          Auxiliary methods                           #
    ########################################################################

    def reverse_dir(self):
        """Reverse direction of flow, i.e. incoming messages are outgoing."""
        # Reverse IP directions
        self.src, self.dst = self.dst, self.src
        # Reverse port directions
        self.sport, self.dport = self.dport, self.sport
        # Reverse direction
        self.direction = [not d for d in self.direction]


    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_dict(self):
        """Return object as dictionary, compatibel with from_dict.

            Returns
            -------
            result : dict()
                Dictionary containing flow information.
            """
        return {
            'protocol'     : self.protocol,
            'src'          : self.src,
            'dst'          : self.dst,
            'sport'        : self.sport,
            'dport'        : self.dport,
            'lengths'      : self.lengths,
            'timestamps'   : self.timestamps,
            'direction'    : self.direction,
            'certificates' : list(self._certificates),
            'domain'       : self.domain,
        }

    def from_dict(self, flow=dict()):
        """Load flows from given flow dictionary.

            Parameters
            ----------
            flow : dict()
                Dictionary containing all flow information.
            """
        # Initialise Flow variables
        self.protocol = flow.get('protocol')
        self.src = flow.get('src')
        self.dst = flow.get('dst')
        self.sport = flow.get('sport')
        self.dport = flow.get('dport')

        self.lengths    = flow.get('lengths'   , list())
        self.timestamps = flow.get('timestamps', list())
        self.direction  = flow.get('direction' , list())

        self._certificates = set(flow.get('certificates', list()))
        self.domain        = flow.get('domain')

        # Check self
        self.check_valid()

    def to_json(self):
        """Convert Flow to a json string.

            Returns
            -------
            result : string
                Json version of self.
            """
        return json.dumps(self.to_dict())

    def from_json(self, string):
        """Load Flow from json string.

            Parameters
            ----------
            string : string
                Json string.

            Returns
            -------
            result : self
                Returns self.
            """
        # Load self from json string
        self.from_dict(json.loads(string))
        # Return self
        return self


    ########################################################################
    #                           Override methods                           #
    ########################################################################

    def __len__(self):
        """Return number of packets in flow."""
        return len(self.lengths)

    def __gt__(self, other):
        """Greater than object override"""
        return self.start >  other.start

    def __ge__(self, other):
        """Greater equals object override"""
        return self.start >= other.start

    def __lt__(self, other):
        """Less than object override"""
        return self.start <  other.start

    def __le__(self, other):
        """Less equals object override"""
        return self.start <= other.start

    def __str__(self):
        """Print flow as string."""
        # Get timestamp of flow
        if self.timestamps:
            timestamp = self.start
            timestamp = datetime.fromtimestamp(timestamp)
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
        else:
            timestamp = "---------- NONE ----------"

        # Return string
        return "{} flow [{}] from {:>15}:{:>5} (client) to {:>15}:{:>5} ({})"\
                .format(self.protocol,timestamp, self.src, self.sport,
                self.dst, self.dport, self.domain)
