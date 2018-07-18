======================
Examples and Workflows
======================

Majors steps in a usual Nabo workflow are:
    * scRNA-Seq data quality control
    * Data normalization
    * Identification of highly variable genes
    * Dimensionality reduction of reference and target population into same PCA space using highly variable genes
    * Creation of SNN graph for reference population by calculation Euclidean distance between each pair of cell.
    * Mapping target cell by calculating the distance between each target and reference cell using a modified Canberra metric.
    * Clustering and visualization of reference graph.
    * Identification of reference sub-populations with significant mapping.
    * Classification of target cells as per reference clusters
    * Identification of marker genes for reference cluster and for highly mapped (by target cells) reference cells.

.. toctree::
    :maxdepth: 2

    mll_enl.rst

