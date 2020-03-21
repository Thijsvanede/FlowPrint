.. _Preprocessor:

Preprocessor
============

The Preprocessor object transforms data to from .pcap files to :ref:`Flow`.

.. autoclass:: preprocessor.Preprocessor

.. automethod:: preprocessor.Preprocessor.__init__

Process data
^^^^^^^^^^^^
The process method extracts all flows and labels (currently the file name) from a given input .pcap file.

.. automethod:: preprocessor.Preprocessor.process

I/O methods
^^^^^^^^^^^

As this process can take a long time, especially when using the pyshark backend (see :ref:`Reader`), the Preprocessor offers methods to save and load data through the means of pickling.

.. automethod:: preprocessor.Preprocessor.save

.. automethod:: preprocessor.Preprocessor.load
