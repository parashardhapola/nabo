========
Overview
========

Nabo is a flexible Python package that allows projections of cells from one population to another. Nabo works by setting one of the populations as a reference' and then maps cells from other populations ('targets') onto it. Nabo provides data implicit methods of verifying mapping quality, this allows users to clearly infer similarities between sub-populations across samples.

Comparing two or more cell populations can be of interest due to multiple reasons:
    * Identifying cells of origin
    * Comparison across replicates
    * Testing population proportions change across conditions

Why use Nabo:
    * No assumptions with respect to nature of data. One can use raw counts or normalized data.
    * Optimized for speed. Nabo calculates distances using Numba's JIT compilers
    * Data persistence. All data is saved in HDF5 format files and data can be reanalyzed from any step.
    * Low memory footprint. Nabo uses out-of-core functions such that only a fraction of data is loaded into the memory at once. Hence, Nabo can easily be run on laptops with modest memory sizes.

Unique features of Nabo:
    * Allows mapping multiple samples over the same reference dataset
    * Visual and statistical comparison of different projections
    * Uses graph-optimized hierarchical clustering approach to identify cell groups in the reference population
    * Provides inbuilt methods to generate null expectation projections
    * Compatible with multiple data formats.
