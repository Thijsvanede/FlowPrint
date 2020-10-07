Command line tool
=================
When FlowPrint is installed, it can be used from the command line.
The :code:`__main__.py` file in the :code:`flowprint` module implements this command line tool.
The command line tool provides a quick and easy interface to convert :code:`.pcap` files into :ref:`Flow` objects and use these objects to create :ref:`Fingerprint`'s.
Once generated, the :ref:`Fingerprint`'s can be used for app recognition and unseen app detection.
The full command line usage is given in its :code:`help` page:

.. code:: text

  usage: flowprint.py [-h] [--fingerprint [FINGERPRINT] | --detection DETECTION | --recognition] [-b BATCH]
                      [-c CORRELATION] [-s SIMILARITY] [-w WINDOW] [-p PCAPS [PCAPS ...]]
                      [-r READ [READ ...]] [-o WRITE] [-l SPLIT] [-a RANDOM] [-t TRAIN [TRAIN ...]]
                      [-e TEST [TEST ...]]

  Flowprint: Semi-Supervised Mobile-App
  Fingerprinting on Encrypted Network Traffic

  optional arguments:
  -h, --help                         show this help message and exit
  --fingerprint   [FINGERPRINT]      mode fingerprint generation [output to FILE] (optional)
  --detection     DETECTION          mode unseen app detection with THRESHOLD
  --recognition                      mode app recognition

  FlowPrint parameters:
  -b, --batch     BATCH              batch  size in seconds                       (default = 300)
  -c, --correlation CORRELATION      cross-correlation threshold                  (default = 0.1)
  -s, --similarity SIMILARITY        similarity        threshold                  (default = 0.9)
  -w, --window    WINDOW             window size in seconds                       (default =  30)

  Flow data input/output:
  -p, --pcaps     PCAPS [PCAPS ...]  pcap(ng) files to run through FlowPrint
  -r, --read      READ [READ ...]    read  preprocessed data from given files
  -o, --write     WRITE              write preprocessed data to   given file
  -l, --split     SPLIT              fraction of data to select for testing
  -a, --random    RANDOM             random state to use for split                (default =  42)

  Train/test input:
  -t, --train     TRAIN [TRAIN ...]  path to json training fingerprints
  -e, --test      TEST [TEST ...]    path to json testing  fingerprints

Examples
^^^^^^^^
Transform :code:`.pcap` files into `flows` and store them in a file.

.. code::

  python3 -m flowprint --pcaps <data.pcap> --write <flows.p>

Extract :code:`fingerprints` from :code:`flows`, split them into training and testing, and store the fingerprints into a file.

.. code::

  python3 -m flowprint --read <flows.p> --fingerprint <fingerprints.json>

Use FlowPrint to recognize apps or detect previously unknown apps

.. code::

  python3 -m flowprint --train <fingerprints.train.json> --test <fingerprints.test.json> --recognition
  python3 -m flowprint --train <fingerprints.train.json> --test <fingerprints.test.json> --detection 0.1
