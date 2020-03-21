.. _Reader:

Reader
======

The Reader object extracts raw features from .pcap files that can be turned into :ref:`Flow` using the :ref:`Preprocessor` class.

.. autoclass:: reader.Reader

.. automethod:: reader.Reader.__init__

Read data
^^^^^^^^^
Reader provides the read() method which reads flow features from a .pcap file.
This method automatically chooses the optimal available backend to use.

.. automethod:: reader.Reader.read

Cutsom Backend
^^^^^^^^^^^^^^

Alternatively, you can choose your own backend using one of the following methods.

.. automethod:: reader.Reader.read_tshark

.. automethod:: reader.Reader.read_pyshark
