.. _Flow:

Flows
=====
A Flow object represents a TCP/UDP flow and all corresponding features that are used by FlowPrint to generate fingerprints.
We use the Flows class for generating Flow objects from all packets extracted by :ref:`Reader`.

Flow generation
^^^^^^^^^^^^^^^
To convert features from individual packets to Flows, we use the :py:meth:`flows.Flows.combine` method.

.. autoclass:: flows.Flows
    :members:

Flow
^^^^

The Flow class is FlowPrint's representation of each individual Flow in the network traffic.

.. autoclass:: flows.Flow

    .. automethod:: __init__

Add packets
-----------
Once created, a Flow is still empty and needs to be populated by packets.
We can add packets to a flow using the :py:meth:`flows.Flow.add` method.

.. automethod:: flows.Flow.add

Attributes
----------
Currently, the Flow object has methods to extract specific features from each flow.
To extract IP features

.. automethod:: flows.Flow.source

.. automethod:: flows.Flow.destination

.. automethod:: flows.Flow.src

.. automethod:: flows.Flow.dst

.. automethod:: flows.Flow.sport

.. automethod:: flows.Flow.dport

To extract the certificate of a flow

.. automethod:: flows.Flow.certificate

To extract temporal features from flow

.. automethod:: flows.Flow.time_start

.. automethod:: flows.Flow.time_end
