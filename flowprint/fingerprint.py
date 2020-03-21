class Fingerprint(frozenset):
    """FlowPrint fingerprint: a frozenset of NetworkDestinations.

        Attributes
        ----------
        destinations : list
            (IP, port) destination tuples in fingerprint

            Note
            ----
            Only as getter, cannot be set

        certificates : list
            Certificates in fingerprint

            Note
            ----
            Only as getter, cannot be set

        n_flows : int
            Threshold for the window size in seconds
    """

    def __new__(cls, *args):
        """FlowPrint fingerprint: a frozenset of NetworkDestinations."""
        # Initialise attributes
        destinations = set()
        certificates = set()
        n_flows      = 0

        # Retrieve attributes from NetworkDestinations
        for cluster in set(*args):
            destinations |= cluster.destinations
            certificates |= cluster.certificates
            n_flows += len(cluster.samples)

        # Create frozenset of destination identifiers
        self = super(Fingerprint, cls).__new__(cls, destinations | certificates)

        # Set number of flows
        self.__setattr__('n_flows', n_flows)

        # Return frozenset
        return self

    ########################################################################
    #                         Comparison functions                         #
    ########################################################################

    def merge(self, *other):
        """Merge fingerprint with other fingerprint(s)

            Parameters
            ----------
            *other : Fingerprint
                One or more fingerprints to merge with given Fingerprint

            Returns
            -------
            result : Fingerprint
                Merged fingerprint
            """
        # Compute union of Fingerprints
        union = set().union(self, *other)
        # Create new fingerprint from union
        result = super(Fingerprint, type(self)).__new__(type(self), union)
        # Set n_flows to combination of self and other
        result.__setattr__('n_flows', self.n_flows + sum(o.n_flows for o in other))
        # Return result
        return result

    def compare(self, other):
        """Compare fingerprint with other fingerprint

            Parameters
            ----------
            other : Fingerprint
                Fingerprint to compare with

            Returns
            -------
            result : float
                Jaccard similarity between self and other
            """
        return len(self & other) / max(len(self | other), 1)

    ########################################################################
    #                              Attributes                              #
    ########################################################################

    @property
    def destinations(self):
        """(IP, port) destination tuples in fingerprint"""
        return sorted([list(x) for x in self if isinstance(x, tuple)])

    @property
    def certificates(self):
        """Certificates in fingerprint"""
        return sorted([x  for x in self if not isinstance(x, tuple)])

    ########################################################################
    #                            I/O functions                             #
    ########################################################################

    def to_dict(self):
        """Return fingerprint as dictionary object

            Returns
            -------
            result : dict
                Fingerprint as dictionary, may be used for JSON export
            """
        return {
            'certificates': self.certificates,
            'destinations': self.destinations,
            'n_flows': self.n_flows,
        }

    def from_dict(self, dictionary):
        """Load fingerprint from dictionary object

            Parameters
            ----------
            dictionary : dict
                Dictionary containing fingerprint object
                  'certificates' -> list of certificates
                  'destinations' -> list of destinations
                  'n_flows'      -> int specifying #flows in fingerprint.

            Returns
            -------
            result : Fingerprint
                Fingerprint object as read from dictionary
            """
        # Get destinations and certificates from dictionary
        dsts  = set([tuple(x) for x in dictionary.get('destinations', [])])
        certs = set([      x  for x in dictionary.get('certificates', [])])
        # Create frozenset of destination identifiers
        self = super(Fingerprint, type(self)).__new__(type(self), dsts | certs)
        # Set number of flows
        self.__setattr__('n_flows', dictionary.get('n_flows', 0))
        # Return self
        return self

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
        return hash(frozenset([x for x in self]))

    ########################################################################
    #                        String representation                         #
    ########################################################################

    def __str__(self):
        return "Fingerprint({}) [size={:4}]".format(hex(id(self)), len(self))
