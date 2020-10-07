.. _FlowPrint:

FlowPrint
=========

The FlowPrint object that is used to generate :ref:`Fingerprint`'s.
Note that this is mainly a wrapper method, the actual Fingerprint generation is done in the :ref:`FingerprintGenerator`.

.. autoclass:: flowprint.FlowPrint

.. automethod:: flowprint.FlowPrint.__init__

Fitting and Predicting
^^^^^^^^^^^^^^^^^^^^^^
We train FlowPrint using the :py:meth:`~flowprint.FlowPrint.fit` method and can predict using the :py:meth:`~flowprint.FlowPrint.predict` method.

.. automethod:: flowprint.FlowPrint.fit

.. automethod:: flowprint.FlowPrint.predict

.. automethod:: flowprint.FlowPrint.fit_predict

Generating fingerprints
^^^^^^^^^^^^^^^^^^^^^^^
As opposed to the :py:meth:`~flowprint.FlowPrint.fit` and :py:meth:`~flowprint.FlowPrint.predict` methods, :py:meth:`~flowprint.FlowPrint.recognize` and :py:meth:`~flowprint.FlowPrint.detect` require :ref:`Fingerprint` objects as input instead of :ref:`Flow` objects. Therefore, we provide a simple method to transform :ref:`Flow` objects to their corresponding :ref:`Fingerprint`.

.. automethod:: flowprint.FlowPrint.fingerprint

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
