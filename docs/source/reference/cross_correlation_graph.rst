.. _CrossCorrelationGraph:

CrossCorrelationGraph
=====================
The CrossCorrelationGraph is used to compute correlations between different :py:obj:`cluster.NetworkDestination`'s and extract cliques.

.. autoclass:: cross_correlation_graph.CrossCorrelationGraph

.. automethod:: cross_correlation_graph.CrossCorrelationGraph.__init__

Graph creation
^^^^^^^^^^^^^^
We use the :py:meth:`cross_correlation_graph.CrossCorrelationGraph.fit` method to create the CrossCorrelationGraph.
Afterwards, we can detect cliques using the :py:meth:`cross_correlation_graph.CrossCorrelationGraph.predict` method.
Or do all in one step using the :py:meth:`cross_correlation_graph.CrossCorrelationGraph.fit_predict` method.

.. automethod:: cross_correlation_graph.CrossCorrelationGraph.fit

.. automethod:: cross_correlation_graph.CrossCorrelationGraph.predict

.. automethod:: cross_correlation_graph.CrossCorrelationGraph.fit_predict

Graph export
^^^^^^^^^^^^
The CrossCorrelationGraph can be exported using the export function.
This can be useful for further investigation using graphical tools such as Gephi_.

.. automethod:: cross_correlation_graph.CrossCorrelationGraph.export

.. _Gephi: https://gephi.org/
