========
Glossary
========

.. glossary::
    :sorted:

    HVGs
        Highly variable genes

    Reference cells
        The cells from a population/sample that will be used as base/reference cells over which other cells are projected.

    Target cells
        The cells which would be individually mapped/projected onto reference cells

    Non zero mean
        Non zero mean of a gene is it's mean expression value calculated by considering only the cells where is was detected

    HDF5:
        It is a hierarchical data format for storing large amount of data. For example expression values for each cell can be saved as vectors which can be accessed using cell names. Once can load data for a cell without needing to load data for the rest of the cells.

    HDF5 group:
        A node in the HDF5 hierarchy under which a dataset for example PCA components or another node can be stored.

    Notebook:
        These are interactive programming interface. When we use this term we mean Jupyter (aka IPython) notebooks. Read more about them `here <http://jupyter.org/>`_.

    SNN graph:
        Graph created by identiyfing shared nearest neigbours between pair of cells.        

    GML format:
        GML (Graph Modeling Language) is a text file format for saving netwrok data.

    Graph modularity:
        After a grap is divided into clusters, it's modularity will be higher if there are high number connections between nodes from the same cluster and very few between nodes from different clusters.
        Modularity is represented on a scale 0-1 where 1 represents a perfectly modular graph.

    Mapping score:
        Mapping score is one of key metric used by Nabo describe cell mapping. This score is given to each reference cell and is directly proportional to the number of target cells mapping to a reference cell.

