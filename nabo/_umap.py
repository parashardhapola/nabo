import pandas as pd
import umap
import h5py

__all__ = ['make_umap']


def make_umap(pca_h5: str, use_comps: int, umap_dims: int, n_neighbors: int,
              spread: float, repulsion_strength: float,
              min_dist: float, n_epochs: int, data_group: str = 'data',
              index_suffix: str = '', verbose: bool = True) -> pd.DataFrame:
    """
    Takes a nabo generated pca file and performs UMAP clustering on cells.
    :param pca_h5: Name of pca H5 file
    :param data_group: key in h5 file under which cell arrays are placed
    :param use_comps: Number of PCA dimensions to use
    :param umap_dims: Number of UMAP dimensions to create
    :param n_neighbors: n_neighbours
    :param spread: spread
    :param repulsion_strength: repulsion_strength
    :param min_dist: min_dist
    :param n_epochs: n_epochs
    :param verbose: verbose
    :param index_suffix: Suffix to be appended to each cell name
    :return:
    """
    h5data = h5py.File(pca_h5, mode='r', swmr=True)

    df = pd.DataFrame(
        {x: h5data['data'][x][:use_comps] for x in h5data[data_group]}).T
    h5data.close()

    um = umap.UMAP(n_neighbors=n_neighbors, n_components=umap_dims,
                   repulsion_strength=repulsion_strength, n_epochs=n_epochs,
                   spread=spread, min_dist=min_dist, verbose=verbose)
    return pd.DataFrame(um.fit_transform(df).T,
                        index=['Dim'+str(x) for x in range(1, umap_dims+1)],
                        columns=[x+index_suffix for x in df.index]).T
