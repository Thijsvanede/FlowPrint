Command line tool
=================
When FlowPrint is installed, it can be used from the command line.
The :code:`__main__.py` file in the :code:`flowprint` module implements this command line tool.
The command line tool provides a quick and easy interface to convert :code:`.pcap` files into :ref:`Flow` objects and use these objects to create :ref:`Fingerprint`'s.
Once generated, the :ref:`Fingerprint`'s can be used for app recognition and unseen app detection.
The full command line usage is given in its :code:`help` page:

.. code:: text

  usage: flowprint.py [-h]
                    (--detection [FLOAT] | --fingerprint [FILE] | --recognition)
                    [-b BATCH] [-c CORRELATION], [-s SIMILARITY], [-w WINDOW]
                    [-p PCAPS...] [-rp READ...] [-wp WRITE]

  Flowprint: Semi-Supervised Mobile-App
  Fingerprinting on Encrypted Network Traffic

  Arguments:
    -h, --help                 show this help message and exit

  FlowPrint mode (select up to one):
    --fingerprint [FILE]       run in raw fingerprint generation mode (default)
                               outputs to terminal or json FILE
    --detection   FLOAT        run in unseen app detection mode with given
                               FLOAT threshold
    --recognition              run in app recognition mode

  FlowPrint parameters:
    -b, --batch       FLOAT    batch size in seconds       (default=300)
    -c, --correlation FLOAT    cross-correlation threshold (default=0.1)
    -s, --similarity  FLOAT    similarity threshold        (default=0.9)
    -w, --window      FLOAT    window size in seconds      (default=30)

  Flow data input/output (either --pcaps or --read required):
    -p, --pcaps  PATHS...      path to pcap(ng) files to run through FlowPrint
    -r, --read   PATHS...      read preprocessed data from given files
    -o, --write  PATH          write preprocessed data to given file
    -i, --split  FLOAT         fraction of data to select for testing (default= 0)
    -a, --random FLOAT         random state to use for split          (default=42)

  Train/test input (for --detection/--recognition):
    -t, --train PATHS...       path to json files containing training fingerprints
    -e, --test  PATHS...       path to json files containing testing fingerprints

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
