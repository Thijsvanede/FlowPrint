Overview
========
This section explains on a high level the different steps taken by FlowPrint to create fingerprints and compare them to recognize apps or detect unseen apps.

 1) `Flow extraction`_

 2) `Fingerprint generation`_

 3) `Fingerprint application`_

    a) `App recognition`_

    b) `Unseen app detection`_

Flow extraction
^^^^^^^^^^^^^^^
FlowPrint itself takes as input an array of :ref:`Flow` objects.
However, we need to extract these flows from the actual network traffic.
Currently, FlowPrint extracts these features from .pcap files using the :ref:`Preprocessor` object.
This module provides the function :py:meth:`preprocessor.Preprocessor.process` method in which you specify .pcap files and their lables as input and outputs :ref:`Flow` objects and their corresponding labels.
The :ref:`Preprocessor` class uses the :ref:`Reader` and :ref:`Flow` classes to produce :ref:`Flow` objects.
These :ref:`Flow` objects can be saved and loaded in files using the :py:meth:`preprocessor.Preprocessor.save` and :py:meth:`preprocessor.Preprocessor.load` methods respectively.
Figure 1 gives an overview of the flow extraction process.

.. figure:: ../_static/overview_processing.png

    Figure 1: Overview flow extraction.

Fingerprint generation
^^^^^^^^^^^^^^^^^^^^^^
After extracting Flows, FlowPrint generates :ref:`Fingerprint` objects.
We refer to our `paper`_ for a detailed overview.
The code implements this as described in Figure 2.
We see that the entire generation process takes place in the :ref:`FingerprintGenerator` object, which uses in order the following classes:

 1) :ref:`Cluster`

 2) :ref:`CrossCorrelationGraph`

 3) :ref:`Fingerprint`

.. figure:: ../_static/overview_generation.png

    Figure 2: Overview of fingerprint generation.

Fingerprint application
^^^^^^^^^^^^^^^^^^^^^^^
This library implements FlowPrint's app recognition and unseen app detection applications.

App recognition
---------------
To recognize known apps, we simply use :ref:`Flowprint`'s :code:`recognize(X)` method.
This method creates new :ref:`Fingerprint` objects for the given :ref:`Flow` objects :code:`X` and compares them to the fingerprints stored using the :code:`fit()` method.
It returns the closest matching fingerprint for each given :ref:`Flow` in :code:`X`.

Unseen app detection
--------------------
To detect unseen apps, we simply use :ref:`Flowprint`'s :code:`detect(X, threshold=0.1)` method.
This method creates new :ref:`Fingerprint` objects for the given :ref:`Flow` objects :code:`X` and compares them to the fingerprints stored using the :code:`fit()` method.
It returns :code:`+1` for each :ref:`Flow` in :code:`X` that matches a known fingerprint and :code:`-1` for each :ref:`Flow` that does not match known fingerprints.


.. _Flow extraction: #flow-extraction

.. _Fingerprint generation: #fingerprint-generation

.. _Fingerprint application: #fingerprint-application

.. _App recognition: #app-recognition

.. _Unseen app detection: #unseen-app-detection

.. _paper: https://dx.doi.org/10.14722/ndss.2020.24412
