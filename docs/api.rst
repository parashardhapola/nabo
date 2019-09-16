===
API
===

Dataset
-------

This is the primary class for processing the data once its saved in HDF5 format. It allows filtering of cells and genes, normalization and scaling of data, identification of highly variable genes and dimensionality reduction using PCA. This class has been designed to seamlessly work with HDF5 format such that only a minimal portion of data is ever loaded in memory. Also, the methods (functions) associated with this class have been designed in such a way that external data can be easily plugged in at any stage. For example, users can bring in normalization factors (cell size factors), a list of highly variable genes, etc.

.. autoclass:: nabo.Dataset
    :members:

Mapping
-------

This is the core class that performs reference graph building by calculating Euclidean distances between cells and then identifying shared nearest neighbours among cells. Cells from any number of target samples can then be mapped over this graph by calculating reference-target cell distances. All the results are saved as a HDF5 file of user's choice.

.. autoclass:: nabo.Mapping
    :members:

Graph
-----

This class allows users to interact with the reference graph containing projected target cells. It is possible to visualize the graph, perform clustering on the graph and generate statistics to assess the mappings.

.. autoclass:: nabo.Graph
    :members:

Marker
------

This module contains functions to identify marker genes (genes with significantly high expression) for sub-populations of interest.

.. autofunction:: nabo.run_de_test
.. autofunction:: nabo.find_cluster_markers

GraphPlot
---------

This class allows a highly customized graph visualization. It is created to work seamlessly with `Graph <api.html#graph>`_ class instances. When called it will, by default, automatically produce the graph visualization. This class requires that the `set_ref_layout <api.html#nabo.Graph.set_ref_layout>`_ method has been called on the `Graph <api.html#graph>`_ object.

.. autoclass:: nabo.GraphPlot
	:members:
