class Fingerprint(frozenset):

    def __new__(cls, *args):
        """FlowPrint fingerprint: a frozenset of NetworkDestinations."""
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

    ########################################################################
    #                            Cast to object                            #
    ########################################################################

    def as_set(self):
        """Get fingerprint as a set."""
        return frozenset(self.certificates | self.destinations)

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
