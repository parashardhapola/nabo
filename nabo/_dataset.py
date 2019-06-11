import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Generator, Tuple
from ._plotting import plot_summary_data, plot_mean_var
import pandas as pd
import re
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import IncrementalPCA
import numba

__all__ = ['Dataset']

tqdm_bar = '{l_bar} {remaining}'


@numba.jit()
def clr(X):
    g = np.exp(np.log(X + 1).mean())
    return np.log((X + 1) / g)


class ExpDict(dict):
    """
    Class with attribute auto-complete feature. Will be useful
    """

    def __init__(self, genes, exp_func, suffix):
        super().__init__({x: None for x in genes})
        self.genes = {x: None for x in genes}
        self.expFunc = exp_func
        self.suffix = suffix

    def __dir__(self):
        return self.genes

    def __getattr__(self, name):
        if name in ['genes', 'expFunc', 'suffix']:
            return self.__getattribute__(name)
        else:
            return self.__getitem__(name)

    def __getitem__(self, key):
        if key not in self.genes:
            raise KeyError('Invalid gene name')
        return self.expFunc(key, as_dict=True, key_suffix=self.suffix)

    def __repr__(self):
        return "ExpDict for Dataset"


class Dataset:
    """
    Class for perform filtering, normalization and dimensionality
    reduction.

    :param h5_fn: Path to input HDF5 file
    :param mito_patterns: Pattern to grep mitochondrial gene names
    :param ribo_patterns: Pattern to grep ribosomal gene names
    :param force_recalc: If set to True then all the saved data from
                         previous instance of this class will be deleted
    """

    def __init__(self, h5_fn: h5py.File, mito_patterns: List[str] = None,
                 ribo_patterns: List[str] = None,
                 force_recalc: bool = False):

        self.h5Fn = h5_fn
        self._mito_patterns: List[str] = mito_patterns
        self._ribo_patterns: List[str] = ribo_patterns
        self._recalc: bool = force_recalc
        self.cells: List[str] = None
        self.genes: List[str] = None
        self.rawNCells: int = None
        self.rawNGenes: int = None
        self.cellIdx: Dict[str, int] = None
        self.geneIdx: Dict[str, int] = None
        self.keepCellsIdx: List[int] = None
        self.keepGenesIdx: List[int] = None
        self.mitoGenes: List[str] = None
        self.riboGenes: List[str] = None
        self.sf: np.ndarray = None
        self.geneStats: pd.DataFrame = None
        self.geneBinsMin = None
        self.varCorrectionFactor = None
        self.hvgList = None
        self.ipca = None
        self._load_info()
        self.exp = ExpDict(self.genes, self.get_norm_exp, '')

    def _load_info(self):
        h5: h5py.File = h5py.File(self.h5Fn, mode='a', libver='latest')
        try:
            # Never sort these! This order is crucial
            self.cells = [x.decode('UTF-8') for x in h5['names']['cells']]
            self.genes = [x.decode('UTF-8') for x in h5['names']['genes']]
        except KeyError:
            self.cells, self.genes = [], []
            raise IOError("FATAL ERROR: Could not extract gene/cell names "
                          "from the H5 file. Please make sure that file was "
                          "generated using Nabo's IO functions ")
        self.rawNCells = len(self.cells)
        self.rawNGenes = len(self.genes)
        self.cellIdx: Dict[str, int] = {x: n for n, x in enumerate(self.cells)}
        self.geneIdx: Dict[str, int] = {x: n for n, x in enumerate(self.genes)}
        if self._recalc is True and 'processed_data' in h5:
            del h5['processed_data']
        if 'processed_data' in h5:
            pd_grp = h5['processed_data']
        else:
            pd_grp = h5.create_group('processed_data')
        if 'keep_cells_idx' in pd_grp and 'keep_genes_idx' in pd_grp:
            self.keepCellsIdx = np.array(list(pd_grp['keep_cells_idx'][:]))
            self.keepGenesIdx = np.array(list(pd_grp['keep_genes_idx'][:]))
            print("INFO: Cached filtered gene and cell names loaded",
                  flush=True)
        else:
            self.keepCellsIdx = np.array(list(range(self.rawNCells)))
            self.keepGenesIdx = np.array(list(range(self.rawNGenes)))
        if 'sf' in pd_grp:
            self.sf = pd_grp['sf'][:]
            print("INFO: Cached cell size factors loaded", flush=True)
        else:
            self.sf = np.ones(self.rawNCells, dtype=np.float32)
        if 'hvg_list' in pd_grp:
            self.hvgList = [x.decode('UTF-8') for x in pd_grp['hvg_list']]
            print("INFO: Loaded cached HVG names", flush=True)
        h5.close()
        if self._mito_patterns is None:
            self._mito_patterns = ['^MT-']
        self.mitoGenes = self.get_genes_by_pattern(self._mito_patterns)
        if self._ribo_patterns is None:
            self._ribo_patterns = ['^RPS', '^RPL', '^MRPS', '^MRPL']
        self.riboGenes = self.get_genes_by_pattern(self._ribo_patterns)
        return True

    def update_exp_suffix(self, suffix: str, delimiter: str = '_') -> None:
        """
        Set the suffix for cell names obtained from `exp` attribute

        :param suffix: Add this string to end of cell names
        :param delimiter: delimiter character/string separating cell name
                          from suffix
        :return:
        """
        if suffix == '':
            self.exp.suffix = ''
        else:
            self.exp.suffix = delimiter + suffix
        return None

    def _get_empty_gene_array(self) -> np.ndarray:
        """

        :return: zero array of shape (rawNGenes,)
        """
        return np.zeros(self.rawNGenes, dtype=np.float32)

    def _get_empty_cell_array(self) -> np.ndarray:
        """

        :return: zero array of shape (rawNCells,)
        """
        return np.zeros(self.rawNCells, dtype=np.float32)

    def get_norm_exp(self, gene: str, as_dict: bool = False,
                     key_suffix: str = '', only_valid_cells: bool = False):
        """
        Get normalized expression of a gene across all the cells.

        :param gene: Valid name of a gene. Gene name will be converted to
                     upper case.
        :param as_dict: if True, returns a dictionary a dictionary with cell
                        names as keys and normalized expression as values
                        otherwise returns a list of normalized expression
                        values. The order of values is same as the cell
                        names in attribute `cells` (default: False)
        :param key_suffix: A character/string to append to each cell name (
                           default: '')
        :param only_valid_cells: If True then returns only valid cells i.e.
                                 cells removed during filtering step are not
                                 included. (default: False)
        :return:
        """
        h5: h5py.File = h5py.File(self.h5Fn, mode='r', libver='latest')
        try:
            d = h5['gene_data'][gene.upper()][:]
        except KeyError:
            raise KeyError("ERROR: This gene symbol does not exist in the "
                           "dataset.")
        h5.close()
        a = self._get_empty_cell_array()
        a[d['idx']] = d['val']
        a = a * self.sf

        if as_dict:
            if only_valid_cells:
                return {self.cells[i] + key_suffix: float(a[i])
                        for i in self.keepCellsIdx}
            else:
                return {self.cells[i] + key_suffix: float(a[i])
                        for i in range(len(a))}
        else:
            if only_valid_cells:
                return a[self.keepCellsIdx]
            else:
                return a

    def get_cum_exp(self, genes: List[str], report_missing: bool = False) -> \
            np.ndarray:
        """
        Calculates cumulative expression of provided genes for each cell.

        :param genes: List of gene names
        :param report_missing: if True then will print a warning message if
                               gene is not found otherwise will remain silent (
                               default: False)
        :return: numpy.ndarray of shape rawNCells with each element as total
                 expression value of param genes in a cell.
        """
        exp = self._get_empty_cell_array()
        h5: h5py.File = h5py.File(self.h5Fn, mode='r', libver='latest')
        for gene in [x.upper() for x in genes]:
            try:
                d = h5['gene_data'][gene]
            except KeyError:
                if report_missing:
                    print('WARNING: Gene %s not found!' % str(gene),
                          flush=True)
            else:
                if d.shape[0] > 0:
                    exp[d['idx']] += d['val']
        h5.close()
        return exp

    def get_genes_per_cell(self) -> np.ndarray:
        """

        :return: ndarray where each element is number of expressed genes
                 from a cell. Order is same as in attribute `cells`.
        """
        a = self._get_empty_cell_array()
        h5: h5py.File = h5py.File(self.h5Fn, mode='r', libver='latest')
        for i in tqdm(range(self.rawNCells), bar_format=tqdm_bar,
                      leave=False, desc='Calculating genes per cell     '):
            a[i] = h5['cell_data'][self.cells[i]].shape[0]
        h5.close()
        return a

    def get_total_exp_per_cell(self) -> np.ndarray:
        """

        :return: ndarray where each element is total expression values/counts
                 from a cell. Order is same as in attribute `cells`.
        """
        a = self._get_empty_cell_array()
        h5: h5py.File = h5py.File(self.h5Fn, mode='r', libver='latest')
        for i in tqdm(range(self.rawNCells), bar_format=tqdm_bar,
                      leave=False, desc='Calculating total exp per cell '):
            a[i] = h5['cell_data'][self.cells[i]]['val'].sum()
        h5.close()
        return a

    def get_gene_abundance(self) -> np.ndarray:
        """

        :return: ndarray where each element is the number of cells
                 where a gene is expressed. Order is same as in attribute
                 `genes`.
        """
        a = self._get_empty_gene_array()
        h5: h5py.File = h5py.File(self.h5Fn, mode='r', libver='latest')
        for i in tqdm(range(self.rawNGenes), bar_format=tqdm_bar,
                      leave=False, desc='Calculating gene abundance     '):
            a[i] = h5['gene_data'][self.genes[i]].shape[0]
        h5.close()
        return a

    def get_genes_by_pattern(self, patterns: List[str]) -> List[str]:
        """
        Get names of genes that match the pattern

        :param patterns: List of Regex pattern
        :return: List of gene names matching the pattern
        """
        genes = []
        for sp in patterns:
            genes.extend(
                [x for x in self.genes if re.match(sp, x) is not None])
        return sorted(set(genes))

    def export_as_dataframe(self, genes: List[str], normalized: bool = True,
                            clr_normed: bool = False,
                            clr_axis: int = 0) -> pd.DataFrame:
        """
        Export data for given genes. Data is exported only for cells
        that are present in keepCellsIdx attribute.
        :param genes: Genes to be exported
        :param normalized: Perform library size normalization (default: True)
        :param clr_normed: Perform CLR normalization (default: False)
        :return: Pandas dataframe with cells as rows and genes as columns
        """

        h5: h5py.File = h5py.File(self.h5Fn, mode='r', libver='latest')
        values = []
        saved_genes = []
        for gene in genes:
            try:
                d = h5['gene_data'][gene.upper()][:]
            except KeyError:
                continue
            a = self._get_empty_cell_array()
            a[d['idx']] = d['val']
            if normalized is True and clr_normed is False:
                a = a * self.sf
            values.append(a[self.keepCellsIdx])
            saved_genes.append(gene)
        h5.close()
        df = pd.DataFrame(
            values, index=saved_genes,
            columns=[self.cells[x] for x in self.keepCellsIdx]).T
        if clr_normed:
            return df.apply(clr, axis=clr_axis)
        else:
            return df

    def get_cell_lib(self, hto_patterns: List[str],
                     min_ratio: float = 0.5) -> pd.Series:
        """
        Get cell library based on hashtag ratios
        :param hto_patterns: Pattern to search for hastags
        :param min_ratio: Minimum ratio (0.5)
        :return: Pandas series
        """
        hto_names = self.get_genes_by_pattern(hto_patterns)
        hto = self.export_as_dataframe(hto_names, normalized=False)
        ratio = (hto.T / hto.sum(axis=1)).T
        valid_cells = (ratio > min_ratio).sum(axis=1) == 1
        ratio = ratio[valid_cells]
        cell_lib = ratio.idxmax(axis=1)
        print("INFO: Predicted library for %d/%d cells" %
              (len(cell_lib), len(self.keepCellsIdx)))
        return cell_lib

    def filter_data(self, min_exp: int = 1000, max_exp: int = np.inf,
                    min_ngenes: int = 100, max_ngenes: int = np.inf,
                    min_mito: int = -1, max_mito: int = 101,
                    min_ribo: int = -1, max_ribo: int = 101,
                    min_gene_abundance: int = 10,
                    rm_mito: bool = True, rm_ribo: bool = True,
                    verbose: bool = True) -> None:
        """
        Filter cells and genes

        :param min_exp: Minimum total expression value for each cell (if
                        count data then these would be minimum number of
                        reads or UMI per cell)
        :param max_exp: Maximum total expression value for each cell.
        :param min_ngenes: Minimum number of genes expressed per cell.
        :param max_ngenes: Maximum number of genes expressed per cell.
        :param min_mito: Minimum percentage of mitochondrial genes' expression
        :param max_mito: Maximum percentage of mitochondrial genes' expression
        :param min_ribo: Minimum percentage of ribosomal genes' expression
        :param max_ribo: Maximum percentage of ribosomal genes' expression
        :param min_gene_abundance: Minimum total expression of gene
        :param rm_mito: if True, exclude mitochondrial genes (default: True)
        :param rm_ribo: if True, exclude mitochondrial genes (default: True)
        :param verbose: if True then report the number of cell/genes removed
                        using each cutoff.(default: True)
        :return: None
        """
        tot_exp_per_cell = self.get_total_exp_per_cell()
        umi_low_cells = list(np.where(tot_exp_per_cell < min_exp)[0])
        umi_high_cells = list(np.where(tot_exp_per_cell > max_exp)[0])

        genes_per_cell = self.get_genes_per_cell()
        ngenes_low_cells = list(np.where(genes_per_cell < min_ngenes)[0])
        ngenes_high_cells = list(np.where(genes_per_cell > max_ngenes)[0])

        percent_mito = (100 * self.get_cum_exp(self.mitoGenes) /
                        tot_exp_per_cell)
        percent_ribo = (100 * self.get_cum_exp(self.riboGenes) /
                        tot_exp_per_cell)

        mito_low_cells = list(np.where(percent_mito < min_mito)[0])
        mito_high_cells = list(np.where(percent_mito > max_mito)[0])
        ribo_low_cells = list(np.where(percent_ribo < min_ribo)[0])
        ribo_high_cells = list(np.where(percent_ribo > max_ribo)[0])
        remove_cells = (umi_low_cells + umi_high_cells +
                        mito_low_cells + mito_high_cells +
                        ribo_low_cells + ribo_high_cells +
                        ngenes_low_cells + ngenes_high_cells)
        remove_cells = set(remove_cells)
        self.keepCellsIdx = np.array(
            sorted(set(self.keepCellsIdx).difference(remove_cells)))
        if min_gene_abundance < 0:
            print("'min_gene_abundance' should be greater than or equal to 0")
            print("Resetting 'min_gene_abundance' to 0", flush=True)
            min_gene_abundance = 0
        gene_abundance = self.get_gene_abundance()
        remove_genes = list(np.where(gene_abundance < min_gene_abundance)[0])
        if rm_mito:
            remove_genes += [self.geneIdx[x] for x in self.mitoGenes]
        if rm_ribo:
            remove_genes += [self.geneIdx[x] for x in self.riboGenes]
        remove_genes = set(remove_genes)
        self.keepGenesIdx = np.array(
            sorted(set(self.keepGenesIdx).difference(remove_genes)))
        if verbose:
            print("UMI filtered  : Low: %d High: %d" % (
                len(umi_low_cells), len(umi_high_cells)), flush=True)
            print("Gene filtered : Low: %d High: %d" % (
                len(ngenes_low_cells), len(ngenes_high_cells)), flush=True)
            print("Mito filtered : Low: %d High: %d" % (
                len(mito_low_cells), len(mito_high_cells)), flush=True)
            print("Ribo filtered : Low: %d High: %d" % (
                len(ribo_low_cells), len(ribo_high_cells)), flush=True)

        h5: h5py.File = h5py.File(self.h5Fn, mode='a', libver='latest')
        grp = h5['processed_data']
        if 'keep_cells_idx' in grp:
            del grp['keep_cells_idx']
        grp.create_dataset('keep_cells_idx', data=self.keepCellsIdx)
        if 'keep_genes_idx' in grp:
            del grp['keep_genes_idx']
        grp.create_dataset('keep_genes_idx', data=self.keepGenesIdx)
        h5.flush(), h5.close()
        return None

    def remove_cells(self, cell_names: List[str], update_cache: bool = False,
                     verbose: bool = False) -> None:
        """
        Remove list of cells by providing their names. Note that no data is
        actually deleted from the dataset but just the keepCellsIdx
        attribute is modified.

        :param cell_names: List of cell names to remove
        :param verbose: Print message about number of cells removed (
                        Default: False)
        :param update_cache: If True then the 'keep_cells_idx' dataset in
                             the H5 file is updated. This will override the
                             saved list of cells (keepCellsIdx) when the
                             dataset is loaded in the future.
        :return:
        """
        rem_cells = [self.cellIdx[i] for i in cell_names if i in self.cellIdx]
        num_keep_cells = len(self.keepCellsIdx)
        self.keepCellsIdx = np.array(sorted(
            set(list(self.keepCellsIdx)).difference(rem_cells)))
        if verbose:
            diff = num_keep_cells - len(self.keepCellsIdx)
            print("%d cells removed" % diff)
        if update_cache:
            h5: h5py.File = h5py.File(self.h5Fn, mode='a', libver='latest')
            grp = h5['processed_data']
            if 'keep_cells_idx' in grp:
                del grp['keep_cells_idx']
            grp.create_dataset('keep_cells_idx', data=self.keepCellsIdx)
            h5.flush(), h5.close()
        return None

    def remove_genes(self, gene_names: List[str], update_cache: bool = False,
                     verbose: bool = False) -> None:
        """
        Remove genes by providing their names. Note that no data is
        actually deleted from the dataset but just the keepGenesIdx
        attribute is
        modified.
        :param gene_names: List of gene names to remove
        :param verbose: Print message about number of cells removed (
                        Default: False)
        :param update_cache: If True then the 'keep_genes_idx' dataset in
                             the H5 file is updated. This will override the
                             saved list of cells (keepCellsIdx) when the
                             dataset is loaded in the future.
        :return:
        """
        rem_genes = [self.geneIdx[i] for i in gene_names if i in self.geneIdx]
        num_keep_genes = len(self.keepGenesIdx)
        self.keepGenesIdx = np.array(sorted(
            set(list(self.keepGenesIdx)).difference(rem_genes)))
        if verbose:
            diff = num_keep_genes - len(self.keepGenesIdx)
            print("%d genes removed" % diff)
        if update_cache:
            h5: h5py.File = h5py.File(self.h5Fn, mode='a', libver='latest')
            grp = h5['processed_data']
            if 'keep_genes_idx' in grp:
                del grp['keep_genes_idx']
            grp.create_dataset('keep_genes_idx', data=self.keepGenesIdx)
            h5.flush(), h5.close()
        return None

    def plot_raw(self, color: str = 'skyblue',
                 display_stats: bool = True,
                 savename: str = None, showfig: bool = True) -> None:
        """
        Plot total expression, genes/cell, % mitochondrial expression and
        % ribosomal expression fro each cell from raw data

        :return: None
        """
        tot_exp_per_cell = self.get_total_exp_per_cell()
        genes_per_cell = self.get_genes_per_cell()
        percent_mito = (100 * self.get_cum_exp(self.mitoGenes) /
                        tot_exp_per_cell)
        percent_ribo = (100 * self.get_cum_exp(self.riboGenes) /
                        tot_exp_per_cell)
        plot_names = ['Total exp. per cell', 'Genes per cell',
                      '% mitochondrial genes', ' % ribosomal genes']
        plot_summary_data([tot_exp_per_cell, genes_per_cell,
                           percent_mito, percent_ribo], plot_names, color,
                          display_stats, savename, showfig)
        print("The dataset contains: %d cells and %d genes"
              % (self.rawNCells, self.rawNGenes), flush=True)
        return None

    def plot_filtered(self, color: str = 'coral',
                      display_stats: bool = True,
                      savename: str = None, showfig: bool = True) -> None:
        """
        Plot total expression, genes/cell, % mitochondrial expression and
        % ribosomal expression for each cell from filtered data

        :param color:
        :param display_stats:
        :param savename:
        :param showfig:
        :return: None
        """

        tot_exp_per_cell = self.get_total_exp_per_cell()[self.keepCellsIdx]
        genes_per_cell = self.get_genes_per_cell()[self.keepCellsIdx]
        percent_mito = (100 * self.get_cum_exp(
            self.mitoGenes)[self.keepCellsIdx] / tot_exp_per_cell)
        percent_ribo = (100 * self.get_cum_exp(
            self.riboGenes)[self.keepCellsIdx] / tot_exp_per_cell)
        plot_names = ['Total exp. per cell', 'Genes per cell',
                      '% mitochondrial genes', ' % ribosomal genes']
        plot_summary_data([tot_exp_per_cell, genes_per_cell,
                           percent_mito, percent_ribo], plot_names, color,
                          display_stats, savename, showfig)
        print("The dataset contains: %d cells and %d genes"
              % (len(self.keepCellsIdx), len(self.keepGenesIdx)), flush=True)
        return None

    def set_sf(self, sf: Dict[str, float] = None, size_scale: float = 1000.0,
               all_genes: bool = False) -> None:
        """
        Set size factor for each cell. Updates `sf` attribute

        :param sf: size factor dict with keys same as the dataset names in
                   'cell_data' group of H5 file and values as size factor for
                   the corresponding cell to be used for normalization.
        :param size_scale: Values are scaled to this factor after
                           normalization using size factor (default: 1000,
                           set to None to disable size scaling).
        :param all_genes: Use expression values from all genes, even those
                          which were filtered out, to calculate size factor.
        :return: None
        """
        try:
            size_scale = float(size_scale)
        except TypeError:
            raise TypeError("ERROR: size_scale parameter should have a float"
                            " value. E.x. not 1 but 1.0")
        if sf is not None:
            for i in sf:
                self.sf[self.cellIdx[i]] = size_scale / sf[i]
        else:
            self.sf: np.ndarray = np.ones(self.rawNCells, dtype=np.float32)
            h5: h5py.File = h5py.File(self.h5Fn, mode='r', libver='latest')
            for i in tqdm(range(self.rawNCells), bar_format=tqdm_bar,
                          leave=False, desc='Setting cell size factors      '):
                temp = self._get_empty_gene_array()
                d = h5['cell_data'][self.cells[i]]
                temp[d['idx']] = d['val']
                if all_genes:
                    temp_sf = temp.sum()
                else:
                    temp_sf = temp[self.keepGenesIdx].sum()
                if temp_sf == 0:
                    temp_sf = 1  # Better check for this upstream
                self.sf[i] = size_scale / temp_sf
            h5.close()
        h5: h5py.File = h5py.File(self.h5Fn, mode='a', libver='latest')
        grp = h5['processed_data']
        if 'sf' in grp:
            del grp['sf']
        grp.create_dataset('sf', data=self.sf)
        h5.flush(), h5.close()
        return None

    def set_gene_stats(self) -> None:
        """
        Calculates the gene-wise expression mean and variance values.
        Sets geneStats attribute as a pandas DataFrame of shape (
        nGeneEntries, 4)

        :return: None
        """
        stats = {}
        keep_genes = {x: 0 for x in self.keepGenesIdx}
        h5: h5py.File = h5py.File(self.h5Fn, mode='r', libver='latest')
        for i in tqdm(range(self.rawNGenes), bar_format=tqdm_bar, leave=False,
                      desc='Calculating gene-wise stats    '):
            gene = self.genes[i]
            if i in keep_genes:
                d = h5['gene_data'][gene][:]
                temp = self._get_empty_cell_array()
                temp[d['idx']] = d['val']
                # Size factor used here to normalize the data
                temp = temp[self.keepCellsIdx] * self.sf[
                    self.keepCellsIdx]
                idx = temp > 0
                if idx.sum() == 0:
                    stats[gene] = {'valid_gene': False}
                else:
                    stats[gene] = {
                        'm': temp.mean(),
                        'nzm': temp[idx].mean(),
                        'variance': temp.var(),
                        'valid_gene': True,
                        'ncells': (temp > 0).sum()
                    }
            else:
                stats[gene] = {'valid_gene': False}
        h5.close()
        # Setting nan values to min values from respective columns
        # This is essential for round function to work which in itself is very
        # non-essential.
        stats = pd.DataFrame(stats).T
        stats.m = stats.m.fillna(stats.m.min())
        stats.nzm = stats.nzm.fillna(stats.nzm.min())
        stats.variance = stats.variance.fillna(stats.variance.min())
        stats.ncells = stats.ncells.fillna(0)
        self.geneStats = stats
        return None

    def correct_var(self, n_bins: int = 100, lowess_frac: float = 0.4) -> None:
        """
        Removes mean-variance trend in the dataset and adds corrected
        variance to as 'fixed_var' column to geneStats.

        :param n_bins: Number of bins for expression values. Larger number
                       of bins will provide a better fit for outliers but
                       may also result in overfitting.
        :param lowess_frac: value for parameter `frac` in statsmodels'
                            `lowess` function
        :return: None
        """
        if self.geneStats is None:
            self.set_gene_stats()
        stats = self.geneStats[self.geneStats.valid_gene].drop(
            columns=['valid_gene']).apply(np.log)
        bin_edges = np.histogram(stats.m, bins=n_bins)[1]
        bin_edges[-1] += 0.1  # For including last gene
        bin_genes = []
        for i in range(n_bins):
            idx: pd.Series = (stats.m >= bin_edges[i]) & \
                             (stats.m < bin_edges[i + 1])
            if sum(idx) > 0:
                bin_genes.append(list(idx[idx].index))
        bin_vals = []
        for genes in bin_genes:
            temp_stat = stats.reindex(genes)
            temp_gene = temp_stat.idxmin().variance
            bin_vals.append(
                [temp_stat.variance[temp_gene], temp_stat.m[temp_gene]])
        bin_vals = np.array(bin_vals).T
        bin_cor_fac = lowess(bin_vals[0], bin_vals[1], return_sorted=False,
                             frac=lowess_frac, it=100).T
        fixed_var = {}
        for bcf, genes in zip(bin_cor_fac, bin_genes):
            for gene in genes:
                fixed_var[gene] = np.e ** (stats.variance[gene] - bcf)
        self.geneStats['fixed_var'] = pd.Series(fixed_var)
        self.geneStats.fixed_var = self.geneStats.fixed_var.fillna(
            self.geneStats.fixed_var.min())
        self.geneBinsMin = bin_vals[1]
        self.varCorrectionFactor = bin_cor_fac
        return None

    def find_hvgs(
            self, var_min_thresh: float = None, nzm_min_thresh: float = None,
            var_max_thresh: float = np.inf, nzm_max_thresh: float = np.inf,
            min_cells: int = 0, plot: bool = True,
            use_corrected_var: bool = False,
            update_cache: bool = False) -> None:
        """
        Identifies highly variable genes using cutoff provided for corrected
        variance and non-zero mean expression. Saves the result in attribute
        `hvgList`.

        NOTE: Input threshold values are considered to be in log scale
              if use_corrected_var is True.

        :param var_min_thresh: Minimum corrected variance
        :param nzm_min_thresh: Minimum non-zero mean
        :param var_max_thresh: Maximum corrected variance
        :param nzm_max_thresh: Minimum non-zero mean
        :param min_cells: Minimum number of cells where a gene should have
                        non-zero value
        :param plot: if True then a scatter plots of mean and variance are
                     displayed highlighting the gene datapoints that were
                     selected as HVGs in blue.
        :param use_corrected_var: if True then uses corrected variance
                                  variance (default: True)
        :param update_cache: If true then Dump HVG list to the HDF5 (
                             default: False)
        :return: None
        """
        if use_corrected_var is True and 'fixed_var' not in self.geneStats:
            raise ValueError('ERROR: "use_corrected_var" parameter is set to '
                             'True. Either  run "correct_var" method first '
                             'or set "use_corrected_var" to False')
        if self.geneStats is None:
            self.set_gene_stats()
        gene_stats = self.geneStats[self.geneStats['valid_gene']].drop(
            columns=['valid_gene'])
        if use_corrected_var:
            temp = gene_stats['ncells'].copy()
            gene_stats = gene_stats.apply(np.log)
            gene_stats['ncells'] = temp
        if nzm_min_thresh is None:
            nzm_min_thresh = np.percentile(gene_stats['nzm'], 5)
        if var_min_thresh is None:
            if use_corrected_var:
                var_min_thresh = np.percentile(gene_stats['fixed_var'], 95)
            else:
                var_min_thresh = np.percentile(gene_stats['variance'], 95)
        if use_corrected_var:
            hvg_candidates = (gene_stats['fixed_var'] > var_min_thresh) & \
                             (gene_stats['nzm'] > nzm_min_thresh) & \
                             (gene_stats['fixed_var'] < var_max_thresh) & \
                             (gene_stats['nzm'] < nzm_max_thresh) & \
                             (gene_stats['ncells'] > min_cells)
        else:
            vmr = gene_stats['variance'] / gene_stats['m']
            hvg_candidates = (vmr > var_min_thresh) & \
                             (gene_stats['nzm'] > nzm_min_thresh) & \
                             (vmr < var_max_thresh) & \
                             (gene_stats['nzm'] < nzm_max_thresh) & \
                             (gene_stats['ncells'] > min_cells)
        if plot and use_corrected_var:
            plot_mean_var(gene_stats, hvg_candidates, var_min_thresh,
                          nzm_min_thresh, var_max_thresh, nzm_max_thresh,
                          [self.geneBinsMin, self.varCorrectionFactor])
        self.hvgList = hvg_candidates[hvg_candidates].index
        if update_cache:
            self.dump_hvgs(self.hvgList)
        print('%d highly variable genes found' % len(self.hvgList), flush=True)

    def dump_hvgs(self, hvgs: List[str]) -> None:
        """
        Save HVGs to HDF5 file

        :param hvgs: List of highly variable gens to save in the HDF5 file
        :return:
        """
        h5: h5py.File = h5py.File(self.h5Fn, mode='a', libver='latest')
        grp = h5['processed_data']
        if 'hvg_list' in grp:
            del grp['hvg_list']
        grp.create_dataset('hvg_list',
                           data=[x.encode("ascii") for x in hvgs])
        h5.flush(), h5.close()

    def get_lvgs(self, nzm_cutoff: float = None, log_nzm_cutoff: float = None,
                 n: int = None, use_corrected_var: bool = False) -> list:
        """
        Get name of names with least corrected variance.

        :param nzm_cutoff: Minimum non-zero mean values for returned genes
        :param log_nzm_cutoff: Minimum non-zero mean values (log scale) for
                               returned genes
        :param n: Number of genes to return (default: same number as HVGs)
        :param use_corrected_var: if True then uses corrected variance
                                  variance (default: True)
        :return: A list of gene names
        """
        if use_corrected_var is True and 'fixed_var' not in self.geneStats:
            raise ValueError('ERROR: "use_fixed_var" parameter is set to '
                             'True. Either  run "correct_var" method first '
                             'or set "use_fixed_var" to False')
        if use_corrected_var:
            use_var = 'fixed_var'
        else:
            use_var = 'variance'
        if nzm_cutoff is None and log_nzm_cutoff is None:
            raise ValueError('ERROR: Please provide a value for either of '
                             'the two parameters: `nzm_cutoff` '
                             'and `log_nzm_cutoff`')
        if nzm_cutoff is not None and log_nzm_cutoff is not None:
            raise ValueError('ERROR: Please provide a value for only ONE '
                             'the two parameters: `nzm_cutoff` '
                             'and `log_nzm_cutoff`')
        if nzm_cutoff is not None:
            cutoff = nzm_cutoff
        else:
            cutoff = np.e ** log_nzm_cutoff
        if n is None:
            n = len(self.hvgList)
        fv = self.geneStats[(self.geneStats['valid_gene']) &
                            (self.geneStats.nzm > cutoff)][use_var]
        if len(fv.index) < n:
            print('WARNING: Number of LVGs is lower than "n"/HVGs. Try '
                  'reducing "nzm_cutoff"')
        return list(fv.sort_values()[:n].index)

    def get_scaling_params(self, genes: List[str] = None,
                           only_valid: bool = True) -> pd.DataFrame:
        """
        Get genes' mean and standard deviation (uncorrected)

        :param genes: Name of genes whose parameters are required. Returns
                      every gene's parameter if value is None (default: None)
        :param only_valid: if True, include only valid genes (default: True)
        :return: A pandas DataFrame contains columns 'mu' and 'sigma'
        """
        if self.geneStats is None:
            self.set_gene_stats()
        params = pd.DataFrame({
            'mu': self.geneStats.m.values,
            'sigma': np.sqrt(self.geneStats.variance.values),
            'genes': self.geneStats.index
        }).set_index('genes')
        if only_valid:
            valid_genes = {x: None for x in self.geneStats[
                self.geneStats.valid_gene].index}
        else:
            valid_genes = {x: None for x in self.geneStats.index}
        if genes is None:
            goi_list = [x for x in valid_genes]
        else:
            goi_list = [x for x in genes if x in valid_genes]
        if len(goi_list) == 0:
            raise ValueError(
                'None of the input genes are valid! Genes should be '
                'valid as given in geneStats attribute')
        return params.reindex(goi_list)

    def get_scaled_values(self, scaling_params: pd.DataFrame,
                          tqdm_msg: str = '', disable_tqdm: bool = False,
                          fill_missing: bool = False) -> \
            Generator[Tuple[str, np.ndarray], None, bool]:
        """
        Generator that yields cell wise scaled expression values.
        The yielded vector will have genes in same order as provided in the
        input list.

        :param scaling_params: Scaling parameters i.e. mean and std. dev. for
                             each genes in form of a pandas DataFrame.
                             This can be obtained from `get_scaling_params`
                             method of Dataset.
        :param tqdm_msg: Message for progress bar (default: '')
        :param disable_tqdm: if True, progress wil not be displayed (
                             default: False)
        :param fill_missing: If True, then gene names in scaling_params that
                             are not present in the Dataset are assigned 0
                             value. (Default: False, raises error if gene name
                             not found). Using this parameter is not
                             recommended as of now because the implications
                             of doing this has not been thoroughly tested.
        :return: (cell name, scaled values)
        """
        mu, sigma = scaling_params['mu'].values, scaling_params['sigma'].values
        # goi makes sure that genes are in a right order
        goi = []
        missing_genes_pos = []
        for n, x in enumerate(scaling_params.index):
            if x not in self.geneIdx:
                if fill_missing is False:
                    raise KeyError(
                        "ERROR: Gene name %s not found! It may that there "
                        "are other gene names in 'scaling_params' that are "
                        "also "
                        "not present in this Dataset. This error mostly "
                        "occurs when you are using scaling_params obtained "
                        "from a different Dataset that was not processed "
                        "using same pipeline to generate gene-cell "
                        "counts/ expression values. To avoid this error, "
                        "one can try to intersect gene names. For example, "
                        "ref_data.hvgList = set("
                        "ref_data.hvgList).intersection(target_data.genes). "
                        "Then obtain the scaling params for this truncated "
                        "gene list. Another possibility is to set "
                        "'fill_missing' parameter to True, however it is "
                        "not recommended as we have not tested the effects "
                        "of using this parameter."
                        % x)
                else:
                    missing_genes_pos.append(n)
                    # use 0 index gene in self.genes as placeholder
                    goi.append(0)
            else:
                goi.append(self.geneIdx[x])
        if len(missing_genes_pos) > 0:
            print("WARNING: %d out %d genes are missing in this dataset" %
                  (len(missing_genes_pos), len(goi)))
        h5: h5py.File = h5py.File(self.h5Fn, mode='r', libver='latest')
        for i in tqdm(self.keepCellsIdx, bar_format=tqdm_bar, leave=False,
                      disable=disable_tqdm, desc=tqdm_msg):
            d = h5['cell_data'][self.cells[i]]
            a = self._get_empty_gene_array()
            a[d['idx']] = d['val']
            a = a[goi]
            a[missing_genes_pos] = 0
            # Normalization and standard scaling
            a = ((a * self.sf[i]) - mu) / sigma
            yield self.cells[i], a
        h5.close()
        return True

    def fit_ipca(self, genes: List[str], n_comps: int = 100,
                 batch_size: int = None, disable_tqdm: bool = False) -> None:
        """
        Fit PCA with genes of interest. The fitted PCA object is saved as
        instance attribute `ipca`. This function uses scikit-learn's
        incremental PCA.

        :param genes: List of genes to use to fit PCA
        :param n_comps: Number of components to use
        :param batch_size: Number of cells to use for fitting in a batch
        :param disable_tqdm: if True, progress will not be displayed (
                             default: False)
        :return: None
        """

        def make_eq_bins(n, bs):
            a = n // bs
            b = n // a
            c = n % a
            for i in range(a):
                if i < c:
                    yield b + 1
                else:
                    yield b

        n_comps = int(n_comps)
        if len(genes) < n_comps:
            n_comps = len(genes)
            print("WARNING: Number of components were reset to number of "
                  "features i.e. %d" % n_comps)
        if n_comps > len(self.keepCellsIdx):
            n_comps = len(self.keepCellsIdx) - 1
            print("WARNING: Number of components were reset to number of "
                  "cells - 1 i.e. %d" % n_comps)
        if batch_size is None or batch_size < n_comps:
            batch_size = n_comps * 2
        if batch_size > len(self.keepCellsIdx):
            batch_size = len(self.keepCellsIdx)
        scaling_params = self.get_scaling_params(genes)
        # Not using whitening parameter here. Testing it on a limited number
        # of datasets did not show any change in mapping, however it did
        # make the reference graph more noisy (read jittered).
        self.ipca = IncrementalPCA(n_components=n_comps)
        cache = []
        cache_sizer = make_eq_bins(len(self.keepCellsIdx), batch_size)
        cur_cache_size = next(cache_sizer)
        n = 0
        for _, a in self.get_scaled_values(
                scaling_params, disable_tqdm=disable_tqdm,
                tqdm_msg='Performing incremental PCA fit '):
            cache.append(list(a))
            n += 1
            if len(cache) == cur_cache_size:
                self.ipca.partial_fit(cache)
                cache = []
                try:
                    cur_cache_size = next(cache_sizer)
                except StopIteration:
                    pass
        if len(cache) > 0:
            print('WARNING: Not all cells were processed! This is a bug. '
                  'Please report it to the authors.')
            print('Debug info: %d %d' % (len(self.keepCellsIdx),
                                         len(cache)))
        self.ipca.genes = list(scaling_params.index)
        return None

    def transform_pca(self, out_file: str, pca_group_name: str,
                      transformer, scaling_params: pd.DataFrame,
                      disable_tqdm: bool = False,
                      fill_missing: bool = False) -> None:
        """
        Transforms values into PCA space and saves them in HDF5 format

        :param out_file: Name of HDF5 file for output
        :param pca_group_name: Name of HDF5 group wherein PCA transformed
                               values will be written. If the group exists
                               then it will be deleted
        :param transformer: sklearn's incremental PCA instance on which
                            fit function has already been called
        :param scaling_params: a DataFrame as return by `get_scaling_params`
                               method of reference sample
        :param disable_tqdm: if True, progress will not be displayed(
                             default: False)
        :param fill_missing: If True, then gene names in scaling_params that
                             are not present in the Dataset are assigned 0
                             value. (Default: False, raises error if gene name
                             not found). Using this parameter is not
                             recommended as of now because the implications
                             of doing this have not been thoroughly tested.
        :return: None
        """
        if transformer is None:
            raise ValueError('ERROR: None value found for transformer. Please '
                             'make sure that the PCA was fitted')
        if scaling_params is None:
            raise ValueError('ERROR: scaling_params need to be a DataFrame')
        try:
            h5 = h5py.File(out_file, mode='a', libver='latest')
        except:
            raise IOError('ERROR: Could not open file %s' % out_file)
        if pca_group_name in h5:
            del h5[pca_group_name]
        grp = h5.create_group(pca_group_name)

        try:
            for i, a in self.get_scaled_values(
                    scaling_params, disable_tqdm=disable_tqdm,
                    fill_missing=fill_missing,
                    tqdm_msg='Transforming to PCA space      '):
                grp.create_dataset(i, data=transformer.transform([a])[0])
        except KeyError as ke:
            h5.close()
            raise KeyError(ke)
        h5.flush(), h5.close()
        return None
