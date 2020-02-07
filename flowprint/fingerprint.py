class Fingerprint(frozenset):

    def __new__(cls, *args):
        """FlowPrint fingerprint: a frozenset of NetworkDestinations."""
        self = super(Fingerprint, cls).__new__(cls, *args)
        self._destinations = None
        self._certificates = None
        self._n_flows    = None
        return self

    ########################################################################
    #                         Comparison functions                         #
    ########################################################################

    def merge(self, other):
        """Merge fingerprint with other fingerprint."""
        return Fingerprint(self | other)

    def compare(self, other):
        """Compare two fingerprints."""
        set_self  = self .as_set()
        set_other = other.as_set()
        return len(set_self & set_other) / max(len(set_self | set_other), 1)

    ########################################################################
    #                         Property definitions                         #
    ########################################################################

    @property
    def n_flows(self):
        """Get number of samples related to this Fingerprint."""
        # Only compute number of samples first time
        if self._n_flows is None:
            # Initialise number of samples
            self._n_flows = 0
            # Loop over all clusters
            for cluster in self:
                # Add number of samples per cluster
                self._n_flows += len(cluster.samples)

        # Return cached n_flows
        return self._n_flows

    ########################################################################
    #                            Cast to object                            #
    ########################################################################

    def as_set(self):
        """Get all destinations related to this Fingerprint."""
        # Initialise destinations and certificates
        destinations = set()
        certificates = set()
        # Loop over all clusters
        for cluster in self:
            destinations |= cluster.destinations
            certificates |= cluster.certificates

        """Get fingerprint as a set."""
        return frozenset(certificates | destinations)

    ########################################################################
    #                              Overrides                               #
    ########################################################################

    def __lt__(self, other):
        return (len(self), self.n_flows) < (len(other), other.n_flows)

    def __eq__(self, other):
        """Equality method override."""
        return hash(self) == hash(other)

    def __hash__(self):
        """Hash method override."""
        # Else return destination & certificate hash
        return hash(self.as_set())

    ########################################################################
    #                        String representation                         #
    ########################################################################

    def __str__(self):
        return "Fingerprint({}) [size={:4}]".format(hex(id(self)), len(self))
