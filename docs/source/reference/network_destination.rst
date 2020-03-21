.. _NetworkDestination:

NetworkDestination
==================

A NetworkDestination represents a cluster of flows that communicate with the same destination.

.. autoclass:: network_destination.NetworkDestination

.. automethod:: network_destination.NetworkDestination.__init__

Adding Flows
------------
We add new Flows using the :py:meth:`network_destination.NetworkDestination.add` method.

.. automethod:: cluster.NetworkDestination.add

Merging destinations
--------------------
When merging two network destinations, we use the :py:meth:`network_destination.NetworkDestination.merge` method.

.. automethod:: cluster.NetworkDestination.merge
