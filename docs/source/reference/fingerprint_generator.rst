.. _FingerprintGenerator:

FingerprintGenerator
====================
This generator performs all steps to transform :ref:`Flow`'s into :ref:`Fingerprint`'s.
These steps include

  1) Batch data
  2) Clustering (also see :ref:`Cluster`)
  3) Cross correlation (also see :ref:`CrossCorrelationGraph`)
  4) Finding cliques (also see :ref:`CrossCorrelationGraph`)
  5) Transforming cliques into Fingerprints. (also see :ref:`Fingerprint`)

.. autoclass:: fingerprints.FingerprintGenerator

.. automethod:: fingerprints.FingerprintGenerator.__init__

Fingerprint generation
^^^^^^^^^^^^^^^^^^^^^^
The method :py:meth:`fingerprints.FingerprintGenerator.fit_predict` performs all steps required for fingerprint generation.

.. automethod:: fingerprints.FingerprintGenerator.fit_predict
