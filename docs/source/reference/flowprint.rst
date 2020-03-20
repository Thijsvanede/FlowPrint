.. _FlowPrint:

FlowPrint
=========

The FlowPrint object that is used to generate :ref:`Fingerprint`'s.
Note that this is mainly a wrapper method, the actual Fingerprint generation is done in the :ref:`FingerprintGenerator`.

.. autoclass:: flowprint.FlowPrint

    .. automethod:: __init__

Generating fingerprints
^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: flowprint.FlowPrint.fit

.. automethod:: flowprint.FlowPrint.predict

.. automethod:: flowprint.FlowPrint.fit_predict

App Recognition
^^^^^^^^^^^^^^^
Once FlowPrint is trained using the :py:meth:`~flowprint.FlowPrint.fit`, you can use FlowPrint to label unknown Flows with known apps.

.. automethod:: flowprint.FlowPrint.recognize

Unseen app detection
^^^^^^^^^^^^^^^^^^^^
Once FlowPrint is trained using the :py:meth:`~flowprint.FlowPrint.fit`, you can use FlowPrint to detect if unknown Flows are in the set of known (trained) apps or if they are a previously unseen app.

.. automethod:: flowprint.FlowPrint.detect

I/O methods
^^^^^^^^^^^
FlowPrint provides methods to save and load a FlowPrint object, including its fingerprints to a json file.

.. automethod:: flowprint.FlowPrint.save

.. automethod:: flowprint.FlowPrint.load
