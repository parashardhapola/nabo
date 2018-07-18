import h5py
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
from typing import List
import random

__all__ = ['mtx_to_h5', 'csv_to_h5']

tqdm_bar = '{l_bar} {remaining}'


def fix_dup_names(names):
    """

    :param names:
    :return:
    """
    names_upper = [x.upper() for x in names]
    dup_names = {item: 0 for item, count in
                 Counter(names_upper).items() if count > 1}
    renamed_names = []
    for name in names_upper:
        if name in dup_names:
            dup_names[name] += 1
            name = name + '_' + str(dup_names[name])
        renamed_names.append(str(name))
    return renamed_names


def mtx_to_h5(in_dir: str, h5_fn: str, batch_size: int =10000,
              value_dtype=np.int64) -> None:
    """
    This is a wrapper function for MtxToH5 class which converts CellRanger's
    output data into Nabo compliant HDF5 format.


    :param in_dir: Name of directory containing barcodes.tsv, genes.tsv and
                   matrix.mtx files
    :param h5_fn: Name of output HDF5 file.
    :param batch_size: Number of cells to process in one chunk. Larger
                       values will lead to increased memory consumption and
                       slight increase in speed.
    :param value_dtype: Data type (Default 64 bit integer)
    :return: None
    """
    temp = MtxToH5(in_dir=in_dir, h5_fn=h5_fn,
                   batch_size=batch_size, value_dtype=value_dtype)
    del temp
    return None


class MtxToH5:
    """

    """

    def __init__(self, in_dir: str, h5_fn: str, batch_size: int,
                 value_dtype):
        """
        Converts 10x output files into Nabo's HDF5 format

        :param in_dir: Name of directory containing barcodes.tsv, genes.tsv and
                   matrix.mtx files
        :param h5_fn: Name of output HDF5 file.
        :param batch_size: Number of cells to process in one chunk. Larger
                           values will lead to increased memory consumption and
                           slight increase in speed.
        :param value_dtype: Data type (Default 64 bit integer)
        """
        self.inDir = in_dir
        self.h5FileName = h5_fn
        self.batchSize = batch_size
        self.dType = [('idx', np.uint), ('val', value_dtype)]

        self.h = open("%s/matrix.mtx" % self.inDir)
        self.h5 = self._make_fn()
        self.h5.create_group('names')

        self.nGenes, self.nCells, self.nVals, self.FreqGenes = self._get_info()

        self.cells = [x.rstrip('\n').upper() for x in open(
                      '%s/barcodes.tsv' % self.inDir).readlines()]
        if len(self.cells) != self.nCells:
            raise ValueError(
                'Number of cells in barcodes.tsv not same as in the mtx file')
        self.h5['names'].create_dataset(
            'cells', chunks=None, data=[x.encode("ascii") for x in self.cells])

        self.genes = fix_dup_names(np.genfromtxt(
            '%s/genes.tsv' % self.inDir, dtype=str)[:, 1])
        if len(self.genes) != self.nGenes:
            raise ValueError(
                'Number of gene in genes.tsv not same as in the mtx file')
        self.h5['names'].create_dataset(
            'genes', chunks=None, data=[x.encode("ascii") for x in self.genes])

        self._make_cell_index()
        self._make_gene_index()
        self.h5.close(), self.h.close()

    def _make_fn(self):
        if os.path.isfile(self.h5FileName):
            print('Overwriting %s' % self.h5FileName, flush=True)
            os.remove(self.h5FileName)
        h5 = h5py.File(self.h5FileName, mode="a", libver='latest')
        return h5

    def _get_info(self):
        for l in self.h:
            if l[0] != '%':
                i = [int(x) for x in l.rstrip('\n').split(' ')]
                return i[0], i[1], i[2], np.zeros(i[0])

    def _make_cell_index(self):

        def _write_data(cd, data):
            data = np.array(data, dtype=self.dType)
            data['idx'] -= 1  # Gene idx set to start from 0
            grp.create_dataset(self.cells[int(cd) - 1], data=data, chunks=None)
            self.FreqGenes[data['idx']] += 1

        grp = self.h5.create_group("cell_data")
        prev_cell, vec = '1', []
        i = None
        for l in tqdm(self.h, total=self.nVals, bar_format=tqdm_bar,
                      desc='Saving cell-wise data          '):
            i = l.rstrip('\n').split(' ')
            if i[1] != prev_cell:
                _write_data(prev_cell, vec)
                prev_cell, vec = i[1], []
            vec.append((i[0], i[2]))
        if len(vec) > 0:
            _write_data(i[1], vec)
        return None

    def _make_gene_index(self):

        grp = self.h5.create_group("gene_data")
        idx_tracker = {x: 0 for x in self.genes}
        for i in range(self.nGenes):
            grp.create_dataset(self.genes[i], shape=(self.FreqGenes[i],),
                               dtype=self.dType)
        gene_cache = {}
        for i in tqdm(range(self.nCells+1), bar_format=tqdm_bar,
                      desc='Saving gene-wise data          '):
            if i < self.nCells:
                d = self.h5['cell_data'][self.cells[i]][:]
                for j in d:
                    gene = self.genes[j[0]]
                    if gene not in gene_cache:
                        gene_cache[gene] = []
                    gene_cache[gene].append((i, j[1]))
            if ((i + 1) % self.batchSize == 0) or i == self.nCells:
                for g in gene_cache:
                    idx = idx_tracker[g]
                    new_idx = idx + len(gene_cache[g])
                    grp[g][idx:new_idx] = gene_cache[g]
                    idx_tracker[g] = new_idx
                gene_cache = {}
        return None


def csv_to_h5(csv_fn: str, h5_fn: str, sep: str = ',',
              genes_in_columns: bool = False, inv_log: int = None,
              value_dtype: type = float, null_thresh: float = None,
              batch_size: int = 500, rownames: List[str] = None,
              colnames: List[str] = None):
    """
    This is a wrapper function for CsvToH5 which allows conversion of

    :param csv_fn: Name of the input CSV file with full path
    :param h5_fn: Output HDF5 filename
    :param sep: Separator in CSV file (Default: ',')
    :param genes_in_columns: Set to True if genes are present in columns. (
                             Default: False; each row is treated as a gene
                             and each column as a cell)
    :param inv_log: base value for calculating antilog. Provide 'e' fir
                    antilog of natural logarithm. Use this if your data is
                    already log normalized. (Default: None; antilog is not
                    calculated)
    :param value_dtype: Data type. Change this to 'int' if you have count
                        values. This will improve the reading operations on the
                        HDF5 file. (Default: float)
    :param null_thresh: Values lower than this value are converted to 0. (
                        Default: None)
    :param batch_size: Number of cells to process in one chunk. Larger
                       values will lead to increased memory consumption and
                       slight increase in speed.
    :param rownames: If the CSV file does not have have rownames then
                     provide them as a list. Please make sure that length of
                     this list is same as the number of rows. (Default: [])
    :param colnames: If the CSV file does not have have rownames then
                     provide them as a list. Please make sure that length of
                     this list is same as the number of rows. (Default: [])
    :return: None
    """
    temp = CsvToH5(csv_fn=csv_fn, h5_fn=h5_fn, sep=sep, inv_log=inv_log,
                   value_dtype=value_dtype, null_thresh=null_thresh,
                   batch_size=batch_size, genes_in_columns=genes_in_columns,
                   rownames=rownames, colnames=colnames)
    del temp
    return None


class CsvToH5:
    """
    Converts csv to HDF5 format

    :param csv_fn: Name of the input CSV file with full path
    :param h5_fn: Output HDF5 filename
    :param sep: Separator in CSV file (Default: ',')
    :param genes_in_columns: Set to True if genes are present in columns. (
                             Default: False; each row is treated as a gene
                             and each column as a cell)
    :param inv_log: base value for calculating antilog. Provide 'e' fir
                    antilog of natural logarithm. Use this if your data is
                    already log normalized. (Default: None; antilog is not
                    calculated)
    :param value_dtype: Data type. Change this to 'int' if you have count
                        values. This will improve the reading operations on the
                        HDF5 file. (Default: float)
    :param null_thresh: Values lower than this value are converted to 0. (
                        Default: None)
    :param batch_size: Number of cells to process in one chunk. Larger
                       values will lead to increased memory consumption and
                       slight increase in speed.
    :param rownames: If the CSV file does not have have rownames then
                     provide them as a list. Please make sure that length of
                     this list is same as the number of rows. (Default: [])
    :param colnames: If the CSV file does not have have rownames then
                     provide them as a list. Please make sure that length of
                     this list is same as the number of rows. (Default: [])
    """
    def __init__(self, csv_fn: str, h5_fn: str, sep: str = ',',
                 inv_log: int = None, value_dtype: type = float,
                 null_thresh: float = None, batch_size: int = 500,
                 genes_in_columns: bool = False, rownames: List[str] = None,
                 colnames: List[str] = None):
        self.csvFn = csv_fn
        self.h5Fn = h5_fn
        self.h5 = self._make_fn()
        self.sep = sep
        self.batchSize = batch_size
        self.genesInCols = genes_in_columns
        self.valueDtype = value_dtype
        self.invLog = inv_log
        self.nullThresh = null_thresh

        self._header_present = False
        if colnames is None:
            colnames = open(self.csvFn).readline().rstrip(
                '\n').split(self.sep)
            colnames = [x.strip('"').strip("'").upper() for x in colnames]
            self._header_present = True
        if rownames is None:
            rownames = []
            colnames = colnames[1:]  # First element of header
            # will be ignored because first column will be treated as rowname
        else:
            rownames = fix_dup_names(rownames)
        self.rowNames = rownames

        self.colNames = fix_dup_names(colnames)

        self.nCols = len(self.colNames)
        self.colFreq = np.zeros(self.nCols)
        self.nRows = len(self.rowNames)

        self._write_row_data()
        self._write_col_data()
        if self.genesInCols:
            genes, cells = self.colNames, self.rowNames
        else:
            genes, cells = self.rowNames, self.colNames
        genes = [x.encode("ascii") for x in genes]
        cells = [x.encode("ascii") for x in cells]
        grp = self.h5.create_group('names')
        grp.create_dataset('genes', chunks=None, data=genes)
        grp.create_dataset('cells', chunks=None, data=cells)
        self.h5.flush()

    def _make_fn(self):
        if os.path.isfile(self.h5Fn):
            print('Overwriting %s' % self.h5Fn, flush=True)
            os.remove(self.h5Fn)
        h5 = h5py.File(self.h5Fn, mode="a", libver='latest')
        return h5

    def _write_row_data(self):
        if self.genesInCols:
            grp = self.h5.create_group("cell_data")
            tqdm_msg = 'Saving cell-wise data          '
        else:
            grp = self.h5.create_group("gene_data")
            tqdm_msg = 'Saving gene-wise data          '
        handle = open(self.csvFn)
        if self._header_present:
            next(handle)
        read_row_name = False
        temp_row_names = None
        if len(self.rowNames) == 0:
            read_row_name = True
            temp_row_names = {}
        # TODO: write an assert to see if len of colnames is correct
        line_num = 0
        for l in tqdm(handle, bar_format=tqdm_bar, desc=tqdm_msg):
            vec = l.rstrip('\n').split(self.sep)
            if read_row_name:
                row_name = vec[0].strip('"').strip("'").upper()
                vec = vec[1:]
                if row_name not in temp_row_names:
                    temp_row_names[row_name] = 1
                else:
                    temp_row_names[row_name] += 1
                    row_name = row_name + ('_%d' % temp_row_names[row_name])
                self.rowNames.append(row_name)
            row_name = self.rowNames[line_num]
            vec = np.array(vec, dtype=self.valueDtype)
            if self.nullThresh is not None:
                vec[vec < self.nullThresh] = 0
            idx = np.nonzero(vec)[0]
            if self.invLog is not None:
                vec[idx] = self.invLog**vec[idx]
            self.colFreq[idx] += 1
            data = np.rec.fromarrays((idx, vec[idx]), names=('idx', 'val'))
            grp.create_dataset(row_name, data=data, chunks=None)
            line_num += 1
        if read_row_name:
            self.nRows = len(self.rowNames)
        self.h5.flush()
        return None

    def _write_col_data(self):
        if self.genesInCols:
            grp1 = self.h5.create_group("gene_data")
            grp2 = self.h5["cell_data"]
            tqdm_msg = 'Saving gene-wise data          '
        else:
            grp1 = self.h5.create_group("cell_data")
            grp2 = self.h5["gene_data"]
            tqdm_msg = 'Saving cell-wise data          '
        idx_tracker = {x: 0 for x in self.colNames}
        # TODO: Maybe a better solution to assign dtype exists
        dtype = grp2[self.rowNames[0]].dtype.descr
        for i in range(self.nCols):
            grp1.create_dataset(self.colNames[i],
                                shape=(self.colFreq[i],), dtype=dtype)
        col_cache = {}
        for i in tqdm(range(self.nRows+1), bar_format=tqdm_bar, desc=tqdm_msg):
            if i < self.nRows:
                d = grp2[self.rowNames[i]][:]
                for j in d:
                    col = self.colNames[j[0]]
                    if col not in col_cache:
                        col_cache[col] = []
                    col_cache[col].append((i, j[1]))
            if ((i + 1) % self.batchSize == 0) or i == self.nRows:
                for g in col_cache:
                    idx = idx_tracker[g]
                    new_idx = idx + len(col_cache[g])
                    grp1[g][idx:new_idx] = col_cache[g]
                    idx_tracker[g] = new_idx
                col_cache = {}
        self.h5.flush()
        return None


def random_sample_h5(n: int, in_fn: str, out_fn: str, group: str='data'):
    """
    This is designed to downscale (subsample cells) from the HDF5 file
    containing PCA values.

    :param n: Number of cells to select
    :param in_fn: Input file
    :param out_fn: Output file
    :param group: Name of HDF5 group in input file that contains the cell
                  wise datasets. The same group name will be used in the
                  output file as well (Default: 'group')
    :return:
    """
    in_h5 = h5py.File(in_fn, mode='r')
    out_h5 = h5py.File(out_fn, mode='w')
    grp = out_h5.create_group(group)
    for i in random.sample(list(in_h5[group]), n):
        grp.create_dataset(i, data=in_h5['data'][i][:])
    in_h5.close()
    out_h5.close()


# def split_h5(in_fn: str, train_fn: str,
#              test_fn: str, batch_size: int = 1000) -> None:
#     """
#     Broken code. Need to to be fixed. Remember to put in __all__ after fixing
#
#     :param in_fn:
#     :param train_fn:
#     :param test_fn:
#     :param batch_size:
#     :return:
#     """
#     def make_gene_index(h, g, c, fqg, bs):
#         grp = h.create_group("gene_data")
#         idx_tracker = {x: 0 for x in g}
#         n_genes = len(g)
#         n_cells = len(c)
#         for x in range(n_genes):
#             grp.create_dataset(g[x], shape=(fqg[x], 2), dtype=np.uint32)
#         gene_cache = {}
#         for x in tqdm(range(n_cells + 1), bar_format=tqdm_bar,
#                       desc='Saving gene-wise data          '):
#             if x < n_cells:
#                 d = h['cell_data'][c[x]][:]
#                 for j in d:
#                     gene = g[j[0]]
#                     if gene not in gene_cache:
#                         gene_cache[gene] = []
#                     gene_cache[gene].append((x, j[1]))
#             if (x != 0 and x % bs == 0) or x == n_cells:
#                 for gene in gene_cache:
#                     dataset = grp[gene]
#                     idx = idx_tracker[gene]
#                     new_idx = idx + len(gene_cache[gene])
#                     dataset[idx:new_idx] = gene_cache[gene]
#                     idx_tracker[gene] = new_idx
#                 gene_cache = {}
#         return None
#
#     h5 = h5py.File(in_fn)
#     if os.path.isfile(train_fn):
#         os.remove(train_fn)
#     train_h5 = h5py.File(train_fn)
#     if os.path.isfile(test_fn):
#         os.remove(test_fn)
#     test_h5 = h5py.File(test_fn)
#
#     cells = np.array([x.decode('UTF8') for x in h5['names']['cells']])
#     genes = np.array([x.decode('UTF8') for x in h5['names']['genes']])
#     train_cells = np.random.choice(cells, 6000, replace=False)
#     test_cells = list(set(cells).difference(train_cells))
#     print("Train cells %d, Test cells %d" % (len(train_cells),
#                                              len(test_cells)), flush=True)
#
#     train_h5.create_dataset('cells', chunks=None,
#                             data=[x.encode("ascii") for x in train_cells])
#     test_h5.create_dataset('cells', chunks=None,
#                            data=[x.encode("ascii") for x in test_cells])
#     train_h5.create_dataset('genes', chunks=None,
#                             data=[x.encode("ascii") for x in genes])
#     test_h5.create_dataset('genes', chunks=None,
#                            data=[x.encode("ascii") for x in genes])
#     train_freq_genes = np.zeros(len(genes))
#     test_freq_genes = np.zeros(len(genes))
#     train_h5.create_group('cell_data')
#     test_h5.create_group('cell_data')
#     for i in tqdm(train_cells, bar_format=tqdm_bar,
#                   desc='Saving train data              '):
#         data = h5['cell_data'][i][:]
#         train_h5['cell_data'].create_dataset(i, data=data, chunks=None,
#                                              dtype=np.uint32)
#         train_freq_genes[data[:, 0]] += 1
#     for i in tqdm(test_cells, bar_format=tqdm_bar,
#                   desc='Saving test data               '):
#         data = h5['cell_data'][i][:]
#         test_h5['cell_data'].create_dataset(i, data=data, chunks=None,
#                                             dtype=np.uint32)
#         test_freq_genes[data[:, 0]] += 1
#     make_gene_index(train_h5, genes, train_cells,
#                     train_freq_genes, batch_size)
#     make_gene_index(test_h5, genes, test_cells, test_freq_genes, batch_size)
