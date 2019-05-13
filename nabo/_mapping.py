import h5py
from typing import List, Dict
import os
import numba
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
import string

__all__ = ['Mapping']

tqdm_bar = '{l_bar} {remaining}'


@numba.jit('float64[:,:],float64[:,:],float64[:,:]')
def _euclidean_dist(x, y, d):
    n, g = y.shape
    m, g = x.shape
    for i in range(m):
        for j in range(n):
            td = 0.0
            for k in range(g):
                temp = x[i, k] - y[j, k]
                td += temp * temp
            d[i, j] = np.sqrt(td)


@numba.jit('float64[:,:],float64[:,:],float64[:,:],float64')
def _mod_canberra_dist(x, y, d, f):
    n, g = y.shape
    m, g = x.shape
    for i in range(m):
        for j in range(n):
            dist = 0.0
            for k in range(g):
                absx = abs(x[i, k])
                num = abs(x[i, k] - y[j, k])
                if num < f*absx:
                    absy = abs(y[j, k])
                    den = (absx + absy + 0.01)
                    dist += num/den
                else:
                    dist += 1
            d[i, j] = dist


def _calc_dist(target_fn: str, target_grp: str, ref_fn: str, ref_grp: str,
               ref_cells: List[str], out_fn: str, dist_grp: str,
               sorted_dist_grp: str, use_comps: int,
               chunk_size: int, intra_ref: bool, dist_factor: float,
               ignore_ref_cells: List[str],
               tqdm_msg1: str, tqdm_msg2: str) -> None:
    """
    Distance calculation

    :param target_fn: Target sample input HDF5 file
    :param target_grp:  Target sample input HDF5 group
    :param ref_fn: Reference sample input HDF5 file
    :param ref_grp:  Reference sample input HDF5 group
    :param ref_cells: List of reference cell names. This will be used to
                      reference cell indices will be saved based on order
                      of this list
    :param out_fn:  Output HDF5 file
    :param dist_grp: Output group for saving distances
    :param sorted_dist_grp:  Output group for saving sorted indices of
                             distances
    :param use_comps: Number of components to use from input dataset
    :param chunk_size: Number fo cells to process in one batch
    :param intra_ref: Set to True if comparing intra reference distances
    :param dist_factor: Distance factor. See documentation glossary
    :param ignore_ref_cells: Name of reference cells to ignore
    :param tqdm_msg1: Message to print while calculating distances
    :param tqdm_msg2: Message to print while sorting distances
    :return:None
    """
    target_h5: h5py.File = h5py.File(target_fn, mode='r')
    target_data: h5py.Group = target_h5[target_grp]
    target_cells: List[str] = [x for x in target_data]
    target_n_cells: int = len(target_cells)
    target_cell_cache: List[str] = []
    target_chunk: List[np.ndarray] = []

    ref_h5: h5py.File = h5py.File(ref_fn, mode='r')
    ref_data: h5py.Group = ref_h5[ref_grp]
    ref_n_cells: int = len(ref_cells)
    ref_cell_idx_cache: List[int] = []
    ref_chunk: List[np.ndarray] = []

    out_h5: h5py.File = h5py.File(out_fn, mode='a')
    if dist_grp in out_h5:
        del out_h5[dist_grp]
    if sorted_dist_grp in out_h5:
        del out_h5[sorted_dist_grp]
    dist_data: h5py.Group = out_h5.create_group(dist_grp)
    sorted_dist_data: h5py.Group = out_h5.create_group(sorted_dist_grp)

    for i in tqdm(range(target_n_cells + 1), total=target_n_cells,
                  bar_format=tqdm_bar, leave=False, desc=tqdm_msg1):
        if i < target_n_cells:
            target_c = target_cells[i]
            dist_data.create_dataset(target_c, shape=(ref_n_cells,),
                                     dtype=np.float64)
            target_cell_cache.append(target_c)
            target_chunk.append(target_data[target_c][:use_comps])
        if (((i + 1) % chunk_size == 0) or i == target_n_cells) and \
                len(target_chunk) > 0:
            target_chunk_array: np.ndarray = np.array(target_chunk)
            for j in range(ref_n_cells + 1):
                if j < ref_n_cells:
                    ref_cell_idx_cache.append(j)
                    ref_c = ref_cells[j]
                    ref_chunk.append(ref_data[ref_c][:use_comps])
                if (((j + 1) % chunk_size == 0) or j == ref_n_cells) and \
                        len(ref_chunk) > 0:
                    ref_chunk_array: np.ndarray = np.array(ref_chunk)
                    dist = np.empty((target_chunk_array.shape[0],
                                     ref_chunk_array.shape[0]), dtype=np.float)
                    if intra_ref:
                        _euclidean_dist(target_chunk_array, ref_chunk_array,
                                        dist)
                    else:
                        _mod_canberra_dist(target_chunk_array, ref_chunk_array,
                                           dist, dist_factor)
                    for k, target_c in zip(dist, target_cell_cache):
                        dist_data[target_c][ref_cell_idx_cache] = k
                    ref_cell_idx_cache = []
                    ref_chunk = []
            target_cell_cache = []
            target_chunk = []
    target_h5.close()
    ref_h5.close()
    out_h5.flush()

    ignored_cells_idx: List[int] = [x for x in range(ref_n_cells) if
                                    ref_cells[x] in ignore_ref_cells]
    mask_bool = np.zeros(ref_n_cells, dtype=bool)
    mask_bool[ignored_cells_idx] = True
    for cell in tqdm(dist_data, bar_format=tqdm_bar, leave=False,
                     desc=tqdm_msg2):
        a = np.ma.array(dist_data[cell][:], mask=mask_bool)
        if intra_ref:
            data: np.ndarray = np.argsort(a)[1:]
        else:
            data: np.ndarray = np.argsort(a)[:]
        sorted_dist_data.create_dataset(cell, data=data)
    out_h5.flush()
    out_h5.close()
    return None


def _calc_snn(fn: str, target_grp: str, target_suffix: str,
              ref_grp: str, ref_suffix: str, ref_cell_names: List[str],
              k: int, tqdm_msg: str) -> nx.Graph:
    """

    :param fn: Name of input hDF5 file containing sorted distance indices
    :param target_grp: Name of group containing sorted indices of reference
                       cells for distance to each target cell. Each dataset
                       in this group represents a target cell
    :param target_suffix: A label to be appended to each target cell in the
                          graph. This should be the name/label as used for
                          this cell during mapping.
    :param ref_grp: Name of group containing sorted indices of reference
                    cells for distance to other reference cells.
    :param ref_suffix: A label to be appended to each reference cell in the
                       graph. This should be the name/label as used for
                       this cell during mapping.
    :param ref_cell_names: A list of names of reference cells. Should be in
                           same order as used in distance calculation.
    :param k: Number of nearest neighbours to consider
    :param tqdm_msg: Message to print while creating SNN graph
    :return: SNN graph as a networkx graph
    """
    h5: h5py.File = h5py.File(fn, mode='r')
    if ref_grp not in h5:
        raise KeyError("ERROR: Please make sure that the distances between "
                       "reference cells has already been calculated")
    if target_grp not in h5:
        raise KeyError("ERROR: Please make sure that the distances between "
                       "reference and target cells has already been "
                       "calculated")
    target_data: h5py.Group = h5[target_grp]
    ref_data: h5py.Group = h5[ref_grp]
    g = nx.Graph()
    factor = 2 * (k - 1)
    for target_c in tqdm(target_data, bar_format=tqdm_bar,
                         leave=False, desc=tqdm_msg):
        # Adding node here such that isolates are also included
        # Adding a mock attribute to extract target cells later
        g.add_node(target_c + '_' + target_suffix, target=None)
        a = set(target_data[target_c][:k])
        for j in a:
            ref_c = ref_cell_names[j]
            snn = len(a.intersection(ref_data[ref_c][:k]))
            jaccard = round(snn / (factor - snn), 2)
            if snn > 0:
                g.add_edge(target_c + '_' + target_suffix,
                           ref_cell_names[j] + '_' + ref_suffix,
                           weight=jaccard)
    h5.close()
    return g


def _fix_disconnected_graph(g: nx.Graph, ref_cells: List[str], fn: str,
                            dist_grp: str, sorted_dist_grp: str, name: str,
                            weight: float) -> nx.Graph:
    """
    Connect graph components based on nearest component.

    :param g: Input disconnected graph
    :param ref_cells: Name of reference cells
    :param fn: HDF5 file containing distances
    :param dist_grp: Distance group name
    :param sorted_dist_grp: Sorted distance indices
    :param name: Reference cell name
    :param weight: Weight for connecting edge
    :return: Networkx graph
    """
    h5 = h5py.File(fn, mode='r')
    cells_map_idx = {x: n for n, x in enumerate(ref_cells)}
    comps = list(nx.connected_components(g))
    n_comps = len(comps)
    comp_lens = [len(x) for x in comps]
    sorted_lens_idx = np.argsort(comp_lens)
    valid_match_cell_idx = {x: [] for x in range(n_comps)}
    for i in range(n_comps):
        idx1 = sorted_lens_idx[i]
        for j in range(i + 1, n_comps):
            idx2 = sorted_lens_idx[j]
            if comp_lens[idx2] > comp_lens[idx1]:
                valid_match_cell_idx[idx1].extend(
                    [cells_map_idx[x.replace('_' + name, '')] for x in
                     comps[idx2]])
    for comp_n in valid_match_cell_idx:
        target_cells = {x: None for x in valid_match_cell_idx[comp_n]}
        if len(target_cells) > 0:
            dists, names1, names2 = [], [], []
            for cell in comps[comp_n]:
                cell = cell.rsplit('_', 1)[0]
                for i in h5[sorted_dist_grp][cell][:]:
                    if i in target_cells:
                        dists.append(h5[dist_grp][cell][i])
                        names1.append(cell)
                        names2.append(ref_cells[i])
                        break
            n1 = names1[np.argsort(dists)[0]] + '_' + name
            n2 = names2[np.argsort(dists)[0]] + '_' + name
            g.add_edge(n1, n2, weight=weight)
    h5.close()
    return g


def _dump_graph(g: nx.Graph, fn: str, out_grp: str) -> None:
    """
    Save networkx graph in HDF5 format

    :param g: Networkx graph
    :param fn: Output HDF5 file
    :param out_grp: Output group
    :return: None
    """
    h5 = h5py.File(fn, mode='a')
    if out_grp in h5:
        del h5[out_grp]
    out_data: h5py.Group = h5.create_group(out_grp)
    for i in g.nodes(data=True):
        if 'target' in i[1]:
            temp = []
            for j in g.edges(i[0], data=True):
                temp.append((j[1].encode('ascii'), j[2]['weight']))
            out_data.create_dataset(i[0], data=temp)
    h5.flush()
    h5.close()
    return None


def random_string(n: int = 30) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))


class Mapping:
    """
    This class encapsulates functions required to perform cell mapping.

    :param mapping_h5_fn: Output filename. Results will be written to this
                          HDF5 file. If the file already exists then it may
                          be used to read existing data
    :param ref_name: Label for reference samples.
    :param ref_pca_fn: Path to HDF5 file that contains input data. Ideally
                       this should be the transformed PCA data file generated
                       by Nabo's Dataset class.
    :param ref_pca_grp_name: Name of the group in the inout HDF5 file
                             wherein the data exists
    :param overwrite: Deletes all the data saved in the mapping_h5_fn to
                      start from scratch (default: False)
    """
    def __init__(self, mapping_h5_fn: str, ref_name: str, ref_pca_fn: str,
                 ref_pca_grp_name: str, overwrite: bool = False):

        self._h5Fn: str = mapping_h5_fn
        self.refName: str = ref_name
        self._refPcaFn: str = ref_pca_fn
        self._refPcaGrp: str = ref_pca_grp_name
        if self._h5Fn == self._refPcaFn:
            raise ValueError(
                "ERROR: Input HDF5 and output HDF5 file cannot be same")
        self._check_h5(self._refPcaFn, self._refPcaGrp)
        self.refCells: List[str] = []
        self._nameStash: Dict[str, str] = {}
        self._check_preload(overwrite)

        self._refDistGrp: str = self._nameStash[self.refName] + '_dist'
        self._refSortedDistGrp: str = \
            self._nameStash[self.refName] + '_sortedDist'
        self._refGraphGrpName = self._nameStash[self.refName] + '_graph'
        self._useComps = None
        self._k = None
        self._distFactor = None
        self._chunkSize = None

    @staticmethod
    def _check_h5(fn: str, group: str) -> bool:
        """
        Verifies that HDF5 file and group exist

        :param fn: Name of file
        :param group: Name of group
        :return: True if filename and group exist
        """
        if os.path.exists(fn) is False:
            raise ValueError("File %s doesn't exist" % fn)
        h5 = h5py.File(fn, mode='r')
        if group not in h5:
            raise ValueError('Group %s does not exist in file %s' % (
                group, fn))
        h5.close()
        return True

    def _create_metadata(self, h5):
        keys = list(h5.keys())
        for i in keys:
            del h5[i]

        grp = h5.create_group('name_stash')
        rs = random_string(30)
        self._nameStash[self.refName] = rs
        grp.create_dataset('ref_name', data=[
            self.refName.encode('ascii'), rs.encode('ascii')])

        self.refCells = self._load_ref_cells()
        grp = h5.create_group('ref_cells')
        grp.create_dataset(
            'ref_cells', data=[x.encode('ascii') for x in self.refCells])
        return True

    def _check_preload(self, overwrite: bool):
        h5 = h5py.File(self._h5Fn, mode='a')
        if overwrite is True:
            self._create_metadata(h5)
        else:
            if 'ref_cells' in h5 and 'ref_cells' in h5['ref_cells'] and \
               'name_stash' in h5 and 'ref_name' in h5['name_stash']:

                ref_name = h5['name_stash/ref_name'][0].decode('UTF-8')
                if ref_name != self.refName:
                    raise ValueError(
                        "ERROR: A different ref_name was used before for "
                        "this mapping file. Please set overwrite=True if "
                        "you want to overwrite all the saved data.")
                else:
                    if 'target_names' in h5['name_stash']:
                        for i in h5['name_stash/target_names']:
                            self._nameStash[i[0].decode('UTF-8')] = \
                                i[1].decode('UTF-8')
                    self._nameStash[self.refName] = \
                        h5['name_stash/ref_name'][1].decode('UTF-8')
                saved_cells = [x.decode('UTF-8') for x in
                               h5['ref_cells/ref_cells'][:]]
                pca_cells = self._load_ref_cells()
                intersect = set(saved_cells).intersection(pca_cells)
                if len(saved_cells) == len(pca_cells) == len(intersect):
                    self.refCells = saved_cells
                else:
                    raise ValueError(
                        "ERROR: Cell names in PCA file does not match those "
                        "used before in this mapping file. Please set "
                        "overwrite=True if you want to overwrite all the "
                        "saved data.")
            else:
                self._create_metadata(h5)
        h5.close()

    def _load_ref_cells(self) -> List[str]:
        """
        Loads names of cells from input HDF5 file saves them into
        'ref_cells/ref_cells' point in the HDF5 output file. If this point
        already exists then will cell names from there.

        :return: List containing cell names
        """

        pca_h5 = h5py.File(self._refPcaFn, mode='r')
        ref_cells = [x for x in pca_h5[self._refPcaGrp]]
        pca_h5.close()
        return ref_cells

    def calc_dist(self, target_fn: str, target_grp: str, dist_grp: str,
                  sorted_dist_grp: str, ignore_ref_cells: List[str]) -> None:
        """
        Calculates euclidean distance between each pair of reference cell
        or a modified canberra distance between each pair of reference and
        target cell.

        :param target_fn: input HDF5 file for target sample. If the
                          distances are being calculated for reference sample
                          then this is reference file name
        :param target_grp: group name within HDF5 file wherein data is located
        :param dist_grp: Name of output group name where distances will be
                         saved
        :param sorted_dist_grp: Name of group where distance sorted cell
                                indices will be saved
        :param ignore_ref_cells: List of names of reference cells to which
                                 distance should not be calculated.
        :return: None
        """
        if self._useComps is None or self._chunkSize is None or  \
                self._distFactor is None:
            raise ValueError('ERROR: Please set the parameters first using '
                             '"set_parameters" method')
        if ignore_ref_cells is None:
            ignore_ref_cells = []
        if target_fn == self._refPcaFn and target_grp == self._refPcaGrp:
            intra_ref = True
            msg1 = 'Calculating ref-ref distances  '
            msg2 = 'Sorting ref-ref distances      '
        else:
            intra_ref = False
            msg1 = 'Calculating target-ref dists   '
            msg2 = 'Sorting target-ref distances   '
        _calc_dist(target_fn, target_grp,
                   self._refPcaFn, self._refPcaGrp, self.refCells, self._h5Fn,
                   dist_grp, sorted_dist_grp, self._useComps, self._chunkSize,
                   intra_ref, self._distFactor, ignore_ref_cells, msg1, msg2)

    def calc_snn(self, target_sorted_dist_grp: str, target_name: str,
                 graph_grp: str, fix_graph_attempts: int = 5,
                 fix_weight: float = None) -> None:
        """
        Creates a shared nearest neighbour graph based on distances
        calculated by 'calc_dist' method.

        :param target_sorted_dist_grp: Name of HDF5 group wherein the
                                       indices of distance sorted cells are
                                       saved.
        :param target_name: A label for target sample. This will be appended to
                            each target cell name in the graph
        :param graph_grp: Name of output group where graph will be saved
        :param fix_graph_attempts: Number of attempts to connect a
                                   disconnected graph. This parameter will be
                                   soon be removed.
        :param fix_weight: Weight of edges used to connect disconnected
                           components of graph (default: 0.5/(2*(k-1))-0.5)
        :return: None
        """
        if self._k is None:
            raise ValueError('ERROR: Set parameters first')
        if target_name == self.refName:
            msg = 'Constructing reference graph   '
        else:
            msg = 'Constructing target graph      '
        g = _calc_snn(self._h5Fn, target_sorted_dist_grp,
                      target_name,
                      self._refSortedDistGrp, self.refName,
                      self.refCells, self._k, msg)
        if target_name == self.refName:
            if nx.is_connected(g) is False:
                if fix_weight is None:
                    fix_weight = 0.5 / ((2 * (self._k - 1)) - 0.5)
                print('INFO: Reference graph is disconnected. Trying to fix..')
                # TODO: monitor the progress and terminate if no progress.
                for i in range(fix_graph_attempts):
                    if nx.is_connected(g) is False:
                        g = _fix_disconnected_graph(
                             g, self.refCells, self._h5Fn, self._refDistGrp,
                             self._refSortedDistGrp, self.refName, fix_weight)
                    else:
                        print(
                            'INFO: Reference graph is no longer disconnected.')
                        break
            if nx.is_connected(g) is False:
                print('WARNING: Output graph is disconnected.')
        _dump_graph(g, self._h5Fn, graph_grp)

    def set_parameters(self, use_comps: int, k: int,
                       dist_factor: float, chunk_size: int) -> None:
        """
        Set run parameters for mapping

        :param use_comps: Number of input dimensions to use. In typical
                          usage this would mean number of PCA components
                          starting from first.
        :param k: Number of nearest neighbours to consider.
                  This is same as k in kNN.
        :param dist_factor: In a given dimension i, if value of target cell
                            is x_i and value for reference cell is y_i then the
                            distance will be saved only if
                            abs(x_i- y_i)/abs(x_i) <  dist_factor; otherwise
                            distance would be given the highest value i.e 1.
        :param chunk_size: Number of cells to load in memory in one go. For
                           smaller RAM usage set a smaller value.
        :return: None
        """
        self._useComps = use_comps
        self._k = k
        try:
            float(dist_factor)
            assert(dist_factor > 0)
        except (ValueError, AssertionError):
            raise ValueError('ERROR: "dist_factor" must be a non-zero float '
                             'value')
        self._distFactor = dist_factor
        self._chunkSize = chunk_size
        return None

    def make_ref_graph(self, use_stored_distances: bool = False):
        """
        A wrapper to run `calc_dist` and `calc_snn` methods creating the
        reference graph

        :param: use_stored_distances: If True then distance between
                                      reference cells is not calculated and
                                      Nabo will try to use the stored
                                      distance matrix. (Default: False)
        :return: None
        """
        if use_stored_distances is False:
            self.calc_dist(self._refPcaFn, self._refPcaGrp, self._refDistGrp,
                           self._refSortedDistGrp, [])
        self.calc_snn(self._refSortedDistGrp, self.refName,
                      self._refGraphGrpName)

    def _stash_target_name(self, target: str):
        h5 = h5py.File(self._h5Fn, mode='a')
        if 'target_names' in h5['name_stash']:
            del h5['name_stash/target_names']
        self._nameStash[target] = random_string(30)
        stash_data = []
        for i in self._nameStash:
            if i != self.refName:
                stash_data.append(
                    [i.encode('ascii'), self._nameStash[i].encode('ascii')]
                )
        h5['name_stash'].create_dataset('target_names', data=stash_data)
        h5.close()

    def map_target(self, target_name: str, target_pca_fn: str,
                   target_pca_grp_name: str,
                   ignore_ref_cells: List[str] = None,
                   use_stored_distances: bool = False,
                   overwrite: bool = False) -> None:
        """
        A wrapper to run calc_dist and calc_snn function for mapping target
        cells onto reference graph. If same target name is provided twice
        then the data is overwritten.

        :param target_name: Label/name of target sample
        :param target_pca_fn: Filename of input data for target. In typical
                              usage this would be the HDF5 file generated
                              using Nabo's Dataset class.
        :param target_pca_grp_name: Name of group containing data in HDF5 file
        :param ignore_ref_cells: List of reference cell names to be excluded
                                 from mapping
        :param use_stored_distances: If True then distance between target and
                                     reference cells is not calculated and
                                     Nabo will try to use the stored
                                     distance matrix. (Default: False)
        :param overwrite: Overwrite the target data
        :return: None
        """
        if target_pca_fn == self._refPcaFn:
            if target_pca_grp_name == self._refPcaGrp:
                raise ValueError(
                    'ERROR: Target PCA file name and group name '
                    'can not be same as that of reference')
        if target_pca_fn == self._h5Fn:
            raise ValueError(
                "ERROR: Input HDF5 and output HDF5 file cannot be same")
        if target_name == self.refName:
            raise ValueError('ERROR: Target name cannot be same as reference '
                             'name. Please provide a different name.')
        if ignore_ref_cells is None:
            ignore_ref_cells = []
        if use_stored_distances is True:
            if target_name not in self._nameStash:
                print("WARNING: Target data not saved. use_stored_distances "
                      "will have no effect")
            else:
                if overwrite is True:
                    print("WARNING: overwrite has no effect as "
                          "use_stored_distances is set to True")
                self.calc_snn(
                    self._nameStash[target_name] + '_sortedDist',
                    target_name, self._nameStash[target_name] + '_graph')
                return None
        else:
            if overwrite is False and target_name in self._nameStash:
                raise ValueError(
                    "ERROR: Data with this target name exists. Please set "
                    "overwrite=True if you want to map this target again.")
        self._stash_target_name(target_name)
        self._check_h5(target_pca_fn, target_pca_grp_name)
        self.calc_dist(target_pca_fn, target_pca_grp_name,
                       self._nameStash[target_name]+'_dist',
                       self._nameStash[target_name]+'_sortedDist',
                       ignore_ref_cells)
        self.calc_snn(self._nameStash[target_name] + '_sortedDist',
                      target_name, self._nameStash[target_name] + '_graph')
        return None
