from datetime import datetime

class Fingerprint(frozenset):

    def __new__(cls, *args):
        self = super(Fingerprint, cls).__new__(cls, *args)
        self._destinations = None
        self._certificates = None
        return self

    ########################################################################
    #                         Comparison functions                         #
    ########################################################################

    def merge(self, other):
        """Merge fingerprint with other fingerprint."""
        return Fingerprint(self | other)

    def compare(self, other):
        """Compare two fingerprints."""
        set_self  = self .certificates | self .destinations
        set_other = other.certificates | other.destinations
        return len(set_self & set_other) / max(len(set_self | set_other), 1)

    def contains(self, other):
        """Returns if flow is in Fingerprint."""
        set_self  = self .certificates | self .destinations
        set_other = other.certificates | other.destinations
        return len(set_self & set_other) / min(len(set_self), len(set_other))

    ########################################################################
    #                         Property definitions                         #
    ########################################################################

    @property
    def destinations(self):
        """Get all destinations related to this Fingerprint."""
        # Only compute destinations first time
        if self._destinations is None:
            # Initialise destinations
            destinations = set()
            # Loop over all clusters
            for cluster in self:
                destinations |= cluster.destinations

            # Cache destinations
            self._destinations = destinations

        # Return cached destinations
        return self._destinations

    @property
    def certificates(self):
        """Get all certificates related to this Fingerprint."""
        # Only compute certificates first time
        if self._certificates is None:
            # Initialise certificates
            certificates = set()
            # Loop over all clusters
            for cluster in self:
                certificates |= cluster.certificates

            # Cache certificates
            self._certificates = certificates

        # Return cached certificates
        return self._certificates

    @property
    def labels(self):
        """Get all labels related to this Fingerprint."""
        # Initialise destinations
        labels = set()
        # Loop over all clusters
        for cluster in self:
            labels |= set(cluster.labels.keys())

        # Return result
        return labels

    @property
    def label(self):
        """Get the label of this Fingerprint."""
        if len(self) == 0:
            return -1
        else:
            votes = dict()
            for cluster in self:
                vote = cluster.labels.most_common(1)[0][0]
                votes[vote] = votes.get(vote, 0) + 1
            return max(votes.items(), key=lambda x: x[1])[0]

    @property
    def time_start(self):
        """Get the start time of flows in fingerprint."""
        return min(min(c.samples).time_start() for c in self)

    @property
    def time_end(self):
        """Get the start time of flows in fingerprint."""
        return max(max(c.samples).time_end() for c in self)


    ########################################################################
    #                            Cast to object                            #
    ########################################################################

    def as_set(self):
        """Get fingerprint as a set."""
        return frozenset(self.certificates | self.destinations)

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def show(self):
        """String representation of fingerprint."""
        destinations = ",\n    ".join(str(d) for d in sorted(self.destinations))
        certificates = ",\n    ".join(str(c) for c in sorted(self.certificates))
        labels       = ", ".join(str(l) for l in sorted(self.labels))
        size_dsts    = len(self)
        size_flows   = sum(len(c.samples) for c in self)
        print("""Fingerprint {{{}}} [size={}, flows={}]
  Started: {}
  Ended  : {}
----------------------------------------
  Destinations={{
    {}
  }}
  Certificates={{
    {}
  }}""".format(labels if size_flows else "Empty", size_dsts, size_flows,
    datetime.fromtimestamp(self.time_start).strftime("%d-%m-%Y %H:%M:%S.%f") if size_flows else "-",
    datetime.fromtimestamp(self.time_end).strftime(  "%d-%m-%Y %H:%M:%S.%f") if size_flows else "-",
    destinations, certificates))

    def __str__(self):
        return "Fingerprint({}) {}".format(hex(id(self)), self.labels)

    ########################################################################
    #                              Overrides                               #
    ########################################################################

    def __eq__(self, other):
        """Equality method override."""
        return hash(self) == hash(other)

    def __hash__(self):
        """Hash method override."""
        # If empty, return identity
        if len(self) == 0:
            return id(self)
        # Else return destination & certificate hash
        return hash((frozenset(self.destinations),
                     frozenset(self.certificates)))
