.. _NetworkDestination:

NetworkDestination
==================

A NetworkDestination represents a cluster of flows that communicate with the same destination.

.. autoclass:: cluster.NetworkDestination

    .. automethod:: __init__

Adding Flows
------------
We add new Flows using the :py:meth:`cluster.NetworkDestination.add` method.

.. automethod:: cluster.NetworkDestination.add

Merging destinations
--------------------
When merging two network destinations, we use the :py:meth:`cluster.NetworkDestination.merge` method.

.. automethod:: cluster.NetworkDestination.merge
