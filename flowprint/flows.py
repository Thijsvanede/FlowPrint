from datetime import datetime
import ipaddress

########################################################################
#                          Single Flow object                          #
########################################################################

class Flow(object):

    def __init__(self):
        """Representation of a Flow."""
        self.ips = set()
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
        # Add flow identifiers
        self.ips.add((packet[5], packet[7]))
        self.ips.add((packet[6], packet[8]))
        # Add certificate if any
        if packet[9] is not None:
            self.certificates.add(packet[9])

        # Perform equivalence checks
        if len(self.ips) > 2:
            raise ValueError("More than 2 IPs found: {}".format(self.ips))
        if len(self.certificates) > 1:
            raise ValueError("More than 1 certificate found: {}"
                             .format(self.certificates))

        # Set timestamps and lengths
        self.timestamps.append(float(packet[3]))
        self.lengths   .append( int(packet[4]) if packet[5] == self.src() else
                               -int(packet[4]))

        # Return self
        return self

    ########################################################################
    #                      Source/Destinatino methods                      #
    ########################################################################

    def get_ips(self):
        """Return set of IP addresses"""
        return sorted([(str(ip), int(port)) for ip, port in self.ips])

    def src(self):
        """Return source IP of flow."""
        (x_ip, x_port), (y_ip, y_port) = self.get_ips()

        # If both IP addresses are private
        if ipaddress.ip_address(x_ip).is_private and\
           ipaddress.ip_address(y_ip).is_private:
            # Return address with highest port
            return x_ip if x_port > y_port else y_ip

        # Return non-private address
        return x_ip if ipaddress.ip_address(x_ip).is_private else y_ip

    def dst(self):
        """Return destination IP of flow."""
        (x_ip, x_port), (y_ip, y_port) = self.get_ips()

        # If both IP addresses are private
        if ipaddress.ip_address(x_ip).is_private and\
           ipaddress.ip_address(y_ip).is_private:
            # Return address with lowest port
            return y_ip if x_port > y_port else x_ip

        # Return non-private address
        return y_ip if ipaddress.ip_address(x_ip).is_private else x_ip

    def sport(self):
        """Return source port of flow."""
        (x_ip, x_port), (y_ip, y_port) = self.get_ips()

        # If both IP addresses are private
        if ipaddress.ip_address(x_ip).is_private and\
           ipaddress.ip_address(y_ip).is_private:
            # Return highest port
            return max(x_port, y_port)

        return x_port if ipaddress.ip_address(x_ip).is_private else y_port

    def dport(self):
        """Return destination port of flow."""
        (x_ip, x_port), (y_ip, y_port) = self.get_ips()

        # If both IP addresses are private
        if ipaddress.ip_address(x_ip).is_private and\
           ipaddress.ip_address(y_ip).is_private:
            # Return lowest port
            return min(x_port, y_port)

        return y_port if ipaddress.ip_address(x_ip).is_private else x_port

    def source(self):
        """Returns source of Flow."""
        return (self.src(), self.sport())

    def destination(self):
        """Returns destination of Flow."""
        return (self.dst(), self.dport())

    def certificate(self):
        """Returns certificate of Flow."""
        return list(self.certificates)[0] if self.certificates else None

    def time_start(self):
        """Returns start time of Flow."""
        return min(self.timestamps)

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
            self.src(), self.sport(), self.dst(), self.dport(),
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

########################################################################
#                         Flow combiner object                         #
########################################################################

class Flows(object):

    def combine(self, packets):
        """Combine individual packets into a flow representation.

            Parameters
            ----------
            packets : np.array of shape=(n_samples_packets, n_features_packets)
                Output from Reader.read

            Returns
            -------
            flows : np.array of shape=(n_samples_flows, n_features_flows)
                Numpy array grouped together as flows.
            """
        # Initialise result
        result = dict()

        # For each packet, add it to a flow
        for packet in packets:
            key = (packet[0], packet[1], packet[2])
            # Add packet to flow
            result[key] = result.get(key, Flow()).add(packet)

        # Return result
        return result
