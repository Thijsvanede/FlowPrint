Getting Started
===============

Dependencies
^^^^^^^^^^^^
FlowPrint requires the following packages to be installed:

- Cryptography: https://pypi.org/project/cryptography/
- Matplotlib: https://matplotlib.org/
- NetworkX: https://networkx.github.io/
- Numpy: https://numpy.org
- Pyshark: https://pypi.org/project/pyshark/
- Scikit-learn: https://scikit-learn.org/stable/index.html

All dependencies should be automatically downloaded if you install FlowPrint via pip. However, should you want to install these libraries manually, you can install the dependencies using the requirements.txt file

.. code::

  pip install -r requirements.txt

Or you can install these libraries yourself

.. code::
  
  pip install -U cryptography matplotlib networkx numpy pyshark scikit-learn

Tshark
------
Although this is

Installation
^^^^^^^^^^^^
The most straigtforward way of installing FlowPrint is via pip

.. code::

  pip install flowprint

If you wish to stay up to date with the latest development version, you can instead download the `source code`_.
In this case, make sure that you have all the required `dependencies`_ installed.

.. _dependencies: #Dependencies
.. _source code: https://github.com/Thijsvanede/FlowPrint
