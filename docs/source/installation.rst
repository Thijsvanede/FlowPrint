Installation
============
The most straigtforward way of installing FlowPrint is via pip

.. code::

  pip install flowprint

If you wish to stay up to date with the latest development version, you can instead download the `source code`_.
In this case, make sure that you have all the required `dependencies`_ installed.

.. note::

  Tshark should always be installed, see `tshark`_.

.. _source code: https://github.com/Thijsvanede/FlowPrint

.. _dependencies:

Dependencies
^^^^^^^^^^^^
FlowPrint requires the following python packages to be installed:

- Cryptography: https://pypi.org/project/cryptography/
- Matplotlib: https://matplotlib.org/
- NetworkX: https://networkx.github.io/
- Numpy: https://numpy.org
- Pandas: https://pandas.pydata.org/
- Pyshark: https://pypi.org/project/pyshark/
- Scikit-learn: https://scikit-learn.org/stable/index.html

All dependencies should be automatically downloaded if you install FlowPrint via pip. However, should you want to install these libraries manually, you can install the dependencies using the requirements.txt file

.. code::

  pip install -r requirements.txt

Or you can install these libraries yourself

.. code::

  pip install -U cryptography matplotlib networkx numpy pandas pyshark scikit-learn

.. _tshark:

Tshark
------
Tshark is required for both the raw tshark backend and the pyshark backend.
You can install tshark as a stand alone, but it also comes with the wireshark installation.
On ubuntu you can install tshark using

.. code::

  sudo apt install tshark

or

.. code::

  sudo apt install wireshark

To test whether tshark is active and in your path, please run

.. code::

  tshark --version

Which should output the current version you are running.

.. note::

  When tshark is not installed, FlowPrint will give a warning message because it tries to use tshark as a backend by default.
  If tshark cannot be found it falls back on pyshark, which is a lot slower.
