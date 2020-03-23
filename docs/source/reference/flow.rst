.. _Flow:

Flow
====
The Flow class is FlowPrint's representation of each individual Flow in the network traffic.
A Flow object represents a TCP/UDP flow and all corresponding features that are used by FlowPrint to generate fingerprints.
We use the :ref:`FlowGenerator` class for generating Flow objects from all packets extracted by :ref:`Reader`.

.. autoclass:: flows.Flow

.. automethod:: flows.Flow.__init__

Add packets
-----------
Once created, a Flow is still empty and needs to be populated by packets.
We can add packets to a flow using the :py:meth:`flows.Flow.add` method.

.. automethod:: flows.Flow.add
