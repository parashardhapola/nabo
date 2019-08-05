import h5py
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
from typing import Generator, IO, Tuple, List, Dict, Optional
import random
import gzip
from ._dataset import Dataset
from abc import ABC, abstractmethod
import pandas as pd
import zarr


__all__ = ['mtx_to_h5', 'csv_to_h5', 'merge_h5', 'extract_cells_from_h5',
           'FileReader', 'MtxReader', 'CrReader', 'CrH5Reader', 'CrDirReader',
           'CrToZarr', 'SubSetZarr']

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


def cellranger_h5_to_mtx(h5, out_dir, genome='GRCh38'):
    a = h5py.File(h5)[genome]

    with open('%s/barcodes.tsv' % out_dir.rstrip('/'), 'w') as OUT:
        OUT.write('\n'.join([x.decode('UTF-8') for x in a['barcodes'][:]]))
    with open('%s/genes.tsv' % out_dir.rstrip('/'), 'w') as OUT:
        gi = [x.decode('UTF-8') for x in a['genes'][:]]
        gn = [x.decode('UTF-8') for x in a['gene_names'][:]]
        OUT.write('\n'.join([x + '\t' + y for x, y in zip(gi, gn)]))

    OUT = open('%s/matrix.mtx' % out_dir.rstrip('/'), 'w')
    OUT.write('%%MatrixMarket matrix coordinate integer general\n%\n')
    OUT.write(' '.join(map(str, [a['genes'].shape[0],
                                 a['barcodes'].shape[0],
                                 a['data'].shape[0]])) + '\n')
    idx = 0
    for n, i in tqdm(enumerate(a['indptr'][1:], 1),
                     total=a['barcodes'].shape[0]):
        cell_chunk = []
        for g, v in zip(a['indices'][idx: i], a['data'][idx: i]):
            cell_chunk.append(' '.join(map(str, [g + 1, n, v])))
        idx = i
        OUT.write('\n'.join(cell_chunk) + '\n')
    OUT.close()


def mtx_to_h5(in_dir: str, h5_fn: str, batch_size: int = 10000,
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
    temp.make_cell_index()
    temp.make_gene_index()
    temp.h5.close(), temp.h.close()
    del temp
    return None


class MtxToH5:
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
        self.isZip = False

        self.h = self._open_mtx_file()
        self.nGenes, self.nCells, self.nVals, self.FreqGenes = self._get_info()
        self.genes, self.cells = self._read_barcodes_genes()

        self.h5 = self._make_h5()
        self.h5.create_group('names')
        self.h5['names'].create_dataset(
            'genes', chunks=None, data=[x.encode("ascii") for x in self.genes])
        self.h5['names'].create_dataset(
            'cells', chunks=None, data=[x.encode("ascii") for x in self.cells])

    def _open_mtx_file(self):
        try:
            handle = open("%s/matrix.mtx" % self.inDir)
        except (OSError, IOError, FileNotFoundError):
            try:
                handle = gzip.open("%s/matrix.mtx.gz" % self.inDir)
                self.isZip = True
            except (OSError, IOError, FileNotFoundError):
                raise FileNotFoundError(
                    'Could not find either either matrix.mtx or '
                    'matrix.mtx.gz in %s' % self.inDir)
        if self.isZip:
            for l in handle:
                yield l.decode('UTF-8').rstrip('\n')
        else:
            for l in handle:
                yield l.rstrip('\n')
        handle.close()

    def _get_info(self):
        while True:
            line = next(self.h)
            if line[0] != '%':
                i = [int(x) for x in line.rstrip('\n').split(' ')]
                return i[0], i[1], i[2], np.zeros(i[0])

    def _make_h5(self):
        if os.path.isfile(self.h5FileName):
            print('Overwriting %s' % self.h5FileName, flush=True)
            os.remove(self.h5FileName)
        h5 = h5py.File(self.h5FileName, mode="a", libver='latest')
        return h5

    def _read_barcodes_genes(self):
        def read_genes(fn):
            try:
                g = np.genfromtxt(fn, dtype=str)[:, 1]
            except (OSError, IOError, FileNotFoundError):
                return False
            return g

        for i in ['genes.tsv', 'genes.tsv.gz',
                  'features.tsv', 'features.tsv.gz']:
            genes = read_genes(self.inDir + '/' + i)
            if genes is not False:
                break
        if genes is not False:
            genes = fix_dup_names(genes)
        else:
            raise FileNotFoundError('ERROR: Could not file gene/features file')
        if len(genes) != self.nGenes:
            raise ValueError(
                'Number of gene in %s not same as in the mtx file' % i)
        try:
            cells = [x.rstrip('\n').upper() for x in open(
                '%s/barcodes.tsv' % self.inDir).readlines()]
        except (OSError, IOError, FileNotFoundError):
            try:
                cells = [x.decode('UTF-8').rstrip('\n').upper() for x in
                         gzip.open(
                             '%s/barcodes.tsv.gz' % self.inDir).readlines()]
            except (OSError, IOError, FileNotFoundError):
                raise FileNotFoundError(
                    'ERROR: Could not barcodes file')
        if len(cells) != self.nCells:
            raise ValueError(
                'Number of cells in barcodes.tsv not same as in the mtx file')
        return genes, cells

    def make_cell_index(self):

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
            i = l.split(' ')
            if i[1] != prev_cell:
                _write_data(prev_cell, vec)
                prev_cell, vec = i[1], []
            vec.append((i[0], i[2]))
        if len(vec) > 0:
            _write_data(i[1], vec)
        return None

    def make_gene_index(self):

        grp = self.h5.create_group("gene_data")
        idx_tracker = {x: 0 for x in self.genes}
        for i in range(self.nGenes):
            grp.create_dataset(self.genes[i], shape=(self.FreqGenes[i],),
                               dtype=self.dType)
        gene_cache = {}
        cell_keys = {x:None for x in self.h5['cell_data']}
        for i in tqdm(range(self.nCells + 1), bar_format=tqdm_bar,
                      desc='Saving gene-wise data          '):
            if i < self.nCells:
                if self.cells[i] not in cell_keys:
                    continue
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
              colnames: List[str] = None, colname_converter: dict = None,
              rowname_converter: dict = None, skip_lines: int = 0):
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
    :param colname_converter: Convert existing row names in file using this
                              dictionary
    :param rowname_converter: Convert existing column names in file using this
                              dictionary
    :param skip_lines: Number of lines to skip from top of the file
                       (Default: 0)
    :return: None
    """
    temp = CsvToH5(csv_fn=csv_fn, h5_fn=h5_fn, sep=sep, inv_log=inv_log,
                   value_dtype=value_dtype, null_thresh=null_thresh,
                   batch_size=batch_size, genes_in_columns=genes_in_columns,
                   rownames=rownames, colnames=colnames,
                   colname_converter=colname_converter,
                   rowname_converter=rowname_converter,
                   skip_lines=skip_lines)
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
    :param colname_converter: Convert existing row names in file using this
                              dictionary
    :param rowname_converter: Convert existing column names in file using this
                              dictionary
    :param skip_lines: Number of lines to skip from top of the file
    """

    def __init__(self, csv_fn: str, h5_fn: str, sep: str = ',',
                 inv_log: int = None, value_dtype: type = float,
                 null_thresh: float = None, batch_size: int = 500,
                 genes_in_columns: bool = False, rownames: List[str] = None,
                 colnames: List[str] = None, colname_converter: dict = None,
                 rowname_converter: dict = None, skip_lines: int = 0):
        self.csvFn = csv_fn
        self.h5Fn = h5_fn
        self.h5 = self._make_fn()
        self.sep = sep
        self.batchSize = batch_size
        self.genesInCols = genes_in_columns
        self.colnameConverter = colname_converter
        self.rownameConverter = rowname_converter
        self.valueDtype = value_dtype
        self.invLog = inv_log
        self.nullThresh = null_thresh
        self.skipLines = skip_lines
        self._header_present = False

        if colnames is None:
            handle = open(self.csvFn)
            for i in range(skip_lines):
                next(handle)
            colnames = next(handle).rstrip('\n').split(self.sep)
            colnames = [x.strip('"').strip("'").upper() for x in colnames]
            self._header_present = True
        if rownames is None:
            rownames = []
            colnames = colnames[1:]  # First element of header
            # will be ignored because first column will be treated as rowname
        else:
            rownames = fix_dup_names(rownames)
        self.rowNames = rownames

        if self.colnameConverter is not None:
            self.colNames = []
            for i in colnames:
                if i in self.colnameConverter:
                    self.colNames.append(self.colnameConverter[i].upper())
                else:
                    self.colNames.append(i)
            self.colNames = fix_dup_names(self.colNames)
        else:
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
        for i in range(self.skipLines):
            next(handle)
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
                if self.rownameConverter is not None:
                    if row_name in self.rownameConverter:
                        row_name = self.rownameConverter[row_name].upper()
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
                vec[idx] = self.invLog ** vec[idx]
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
        for i in tqdm(range(self.nRows + 1), bar_format=tqdm_bar,
                      desc=tqdm_msg):
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


def random_sample_h5(n: int, in_fn: str, out_fn: str, group: str = 'data'):
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


def merge_h5(h5_fns: List[str], merged_fn: str, cell_suffixes=None) -> None:
    """
    Merge multiple Nabo's dataset h5 files

    :param h5_fns: List containing names of input files to be merged
    :param merged_fn: Output file name
    :param cell_suffixes: A list containing strings (same number as number
                          of H5 files) to be appended to each cell's name
    :return: None
    """
    handles = [h5py.File(x, 'r') for x in h5_fns]
    merged = h5py.File(merged_fn, 'w')

    required_keys = ['gene_data', 'cell_data', 'names/cells', 'names/genes']
    for n, h in enumerate(handles):
        for i in required_keys:
            if i not in h:
                [x.close() for x in handles]
                merged.close()
                raise KeyError(
                    'ERROR: Group %s missing in file %s' % (i, h5_fns[n]))

    if cell_suffixes is None:
        cell_suffixes = [str(x + 1) for x in range(len(h5_fns))]
    elif len(cell_suffixes) != len(h5_fns):
        [x.close() for x in handles]
        merged.close()
        raise ValueError(
            "ERROR: Number of cell_suffixes not same as number of input H5 "
            "files")

    print('Creating gene indices', flush=True)
    union_genes = set(handles[0]['names/genes'][:])
    for h in handles[1:]:
        union_genes.union(h['names/genes'][:])
    union_genes = sorted([x.decode('UTF-8') for x in union_genes])
    ugk = {x: n for n, x in enumerate(union_genes)}
    ugk_maps = []
    for h in handles:
        temp_ugk_map = {}
        for n, g in enumerate(h['names/genes']):
            temp_ugk_map[n] = ugk[g.decode('UTF-8')]
        ugk_maps.append(temp_ugk_map)

    print('Creating cell indices', flush=True)
    union_cells = []
    for n, h in enumerate(handles):
        union_cells.extend([x.decode('UTF-8') + '-' + cell_suffixes[n] for x in
                            h['names/cells'][:]])
    union_cells = random.sample(union_cells, len(union_cells))
    uck = {x: n for n, x in enumerate(union_cells)}
    uck_maps = []
    for n, h in enumerate(handles):
        temp_uck_map = {}
        for n2, c in enumerate(h['names/cells']):
            temp_uck_map[n2] = uck[c.decode('UTF-8') + '-' + cell_suffixes[n]]
        uck_maps.append(temp_uck_map)

    merged.create_group('names')
    merged['names'].create_dataset(
        'genes', chunks=None,
        data=[str(x).encode("ascii") for x in union_genes])
    merged['names'].create_dataset(
        'cells', chunks=None,
        data=[str(x).encode("ascii") for x in union_cells])

    print('Saving gene data', flush=True)
    grp = merged.create_group("gene_data")
    for g in tqdm(union_genes):
        data = []
        for n, h in enumerate(handles):
            try:
                d = h['gene_data'][g][:]
            except KeyError:
                continue
            else:
                d['idx'] = [uck_maps[n][x] for x in d['idx']]
                data.append(d)
        grp.create_dataset(g, data=np.concatenate(data), chunks=None)

    print('Saving cell data', flush=True)
    grp = merged.create_group("cell_data")
    for n, h in enumerate(handles):
        for c in tqdm(h['cell_data'], desc=cell_suffixes[n]):
            data = h['cell_data'][c][:]
            data['idx'] = [ugk_maps[n][x] for x in data['idx']]
            grp.create_dataset(c + '-' + cell_suffixes[n], data=data,
                               chunks=None)

    [x.close() for x in handles]
    merged.close()

    return None


def extract_cells_from_h5(in_fn: str, out_fn: str,
                         coi: List[str]) -> None:
    """
    Extract data for given cells from a H5 file and create a new H5 file
    with the cell subset

    :param in_fn: Input H5 filename
    :param out_fn: Output H5 file name
    :param coi: cellnames to be extracted
    :return:
    """
    coi = {x: None for x in coi}
    ih = h5py.File(in_fn, 'r')
    oh = h5py.File(out_fn, 'w')
    in_cells = {x.decode('UTF-8'): n for n, x in enumerate(ih['names/cells'])}
    out_cells = [x for x in in_cells if x in coi]

    if len(out_cells) == 0:
        raise ValueError(
            "ERROR: None of the input cells were found in the H5 file")
    if len(out_cells) != len(coi):
        print("WARNING: only %d/%d cells found in the H5 file" % (
            len(out_cells), len(coi)))
    cell_idx_map = {in_cells[x]: n for n, x in enumerate(out_cells)}

    oh.create_group('names')
    oh['names'].create_dataset(
        'genes', chunks=None, data=list(ih['names/genes']))
    oh['names'].create_dataset(
        'cells', chunks=None, data=[x.encode("ascii") for x in out_cells])

    grp = oh.create_group("cell_data")
    for i in tqdm(out_cells):
        grp.create_dataset(i, data=ih['cell_data'][i], chunks=None)

    grp = oh.create_group("gene_data")
    for g in tqdm(ih['gene_data']):
        data = []
        d = ih['gene_data'][g][:]
        for i in d:
            if i[0] in cell_idx_map:
                data.append((cell_idx_map[i[0]], i[1]))
        grp.create_dataset(g, data=np.array(data, dtype=d.dtype), chunks=None)

    ih.close(), oh.close()
    return None


def h5_to_mtx(fn, outdir, temp_suffix):
    data = Dataset(fn, force_recalc=True)
    h5 = h5py.File(fn, mode='r')
    total = 0
    o1 = temp_suffix + '1'
    o2 = temp_suffix + '2'
    with open(o1, 'w') as h:
        for n, i in tqdm(enumerate(data.cells)):
            a = h5['cell_data'][i][:]
            a.sort(order=['idx'], axis=0)
            b = np.vstack([a['idx']+1,
                           [n+1 for _ in range(len(a))],
                           a['val']]).astype(int).T
            total += b.shape[0]
            h.write('\n'.join(['\t'.join(map(str, x)) for x in b])+'\n')
    h5.close()

    with open(o2, 'w') as h:
        h.write('%%MatrixMarket matrix coordinate integer general\n%metadata_json: {"format_version": 2, "software_version": "3.0.2"}\n')
        h.write("%d %d %d\n" % (data.rawNGenes, data.rawNCells, total))

    os.system("cat %s %s > ./nabo_10x/%s/matrix.mtx" % (o2, o1, temp_suffix))
    os.system("rm %s %s" % (o1,  o2))

    with open('%s/genes.tsv' % outdir, 'w') as h:
        h.write('\n'.join([x+'\t'+x for x in data.genes]))
    with open('%s/barcodes.tsv' % temp_suffix, 'w') as h:
        h.write('\n'.join(data.cells))


class FileReader:
    def __init__(self, fn: str, chunk_size: int = 10000,
                 by_line: bool = False):
        self.fn = fn
        self.chunkSize = chunk_size
        self._isCompressed = self._check_compressed()
        self.byLine = by_line

    def _check_compressed(self) -> bool:
        if self.fn.rsplit('.', 1)[-1] == 'gz':
            return True
        return False

    def _open(self) -> iter:
        try:
            if self._isCompressed:
                return gzip.open(self.fn, mode='rt')
            else:
                return open(self.fn, 'r')
        except (OSError, IOError, FileNotFoundError):
            raise FileNotFoundError("ERROR: FILE NOT FOUND: %s" % self.fn)

    def _read_chunk(self, fh: IO) -> Generator[str, None, None]:
        while True:
            chunk = fh.readlines(self.chunkSize)
            if not chunk:
                break
            yield chunk

    def consume(self) -> Generator[str, None, None]:
        fh = self._open()
        for i in self._read_chunk(fh):
            if self.byLine:
                for j in map(str.rstrip, i):
                    yield j
            else:
                yield list(map(str.rstrip, i))
        fh.close()

    def __repr__(self):
        return (
            f"FileReader:\n File: {self.fn}\n Compressed: {self._isCompressed}")


class MtxReader(FileReader):
    def __init__(self, fn: str, n_features: int, chunk_size=int(1e4)):
        super().__init__(fn, chunk_size, by_line=False)
        self.nFeatures = n_features

    def _to_dense(self, i: np.ndarray) -> np.ndarray:
        a = np.zeros(self.nFeatures, dtype=int)
        a[i[:, 0] - 1] = i[:, 2]
        return a

    def consume(self) -> Generator[np.array, None, None]:
        fh = self._open()
        [next(fh) for _ in range(3)]
        carry_chunk: np.ndarray = np.ndarray((0, 0))
        for chunk in self._read_chunk(fh):
            chunk = map(lambda x: np.fromstring(x, sep=' ', dtype=int), chunk)
            chunk = np.array(list(carry_chunk) + list(chunk))
            start: int = 0
            pos: np.ndarray = np.where(np.diff(chunk[:, 1]) == 1)[0] + 1
            for end in pos:
                yield self._to_dense(chunk[start:end])
                start = end
            carry_chunk = chunk[start:]
        yield self._to_dense(carry_chunk)
        fh.close()


class CrReader(ABC):
    def __init__(self, grp_names):
        self.grpNames: Dict = grp_names
        self.nFeatures: int = len(self.featureNames())
        self.nCells: int = len(self.cellNames())
        self.assayFeats = self._make_feat_table()

    @abstractmethod
    def _handle_version(self):
        pass

    @abstractmethod
    def _read_dataset(self, key: Optional[str] = None) -> List:
        pass

    @abstractmethod
    def consume(self) -> Generator[List[np.ndarray], None, None]:
        pass

    def _subset_by_assay(self, v, assay) -> List:
        if assay is None:
            return v
        elif assay not in self.assayFeats:
            raise ValueError("ERROR: Assay ID %s is not valid" % assay)
        l = self.assayFeats[assay]
        return v[l.start: l.end]

    def _make_feat_table(self) -> pd.DataFrame:
        s = self.featureTypes()
        span: List[Tuple] = []
        last = s[0]
        last_n: int = 0
        for n, i in enumerate(s[1:], 1):
            if i != last:
                span.append((last, last_n, n))
                last_n = n
            elif n == len(s) - 1:
                span.append((last, last_n, n + 1))
            last = i
        df = pd.DataFrame(span, columns=['type', 'start', 'end'])
        df.index = ["assay%s" % str(x + 1) for x in df.index]
        df['nFeatures'] = df.end - df.start
        return df.T

    def rename_assays(self, name_map: Dict[str, str]) -> None:
        self.assayFeats.rename(columns=name_map, inplace=True)

    def featureIds(self, assay: str = None) -> List[str]:
        return self._subset_by_assay(self._read_dataset('featureIds'), assay)

    def featureNames(self, assay: str = None) -> List[str]:
        return self._subset_by_assay(self._read_dataset('featureNames'), assay)

    def featureTypes(self) -> List[str]:
        if self.grpNames['featureTypes'] is not None:
            return self._read_dataset('featureTypes')
        else:
            return ['Gene Expression' for x in range(self.nFeatures)]

    def cellNames(self, dummy_arg=None) -> List[str]:
        return self._read_dataset('cellNames')


class CrH5Reader(CrReader):
    def __init__(self, h5_fn):
        self.h5obj = h5py.File(h5_fn, mode='r')
        self.grp = None
        super().__init__(self._handle_version())

    def _handle_version(self):
        root_key = list(self.h5obj.keys())[0]
        self.grp = self.h5obj[root_key]
        if root_key == 'matrix':
            grps = {'featureIds': 'features/id',
                    'featureNames': 'features/name',
                    'featureTypes': 'features/feature_type',
                    'cellNames': 'barcodes'}
        else:
            grps = {'featureIds': 'genes', 'featureNames': 'gene_names',
                    'featureTypes': None, 'cellNames': 'barcodes'}
        return grps

    def _read_dataset(self, key: Optional[str] = None):
        return [x.decode('UTF-8') for x in self.grp[self.grpNames[key]][:]]

    def consume(self) -> Generator[List[np.ndarray], None, None]:
        idx = 0
        split_idx = self.assayFeats.T.end.values
        for n, i in enumerate(self.grp['indptr'][1:], 1):
            g, v = self.grp['indices'][idx: i], self.grp['data'][idx: i]
            idx = i
            a = np.zeros(self.nFeatures, dtype=int)
            a[g] = v
            yield np.split(a, split_idx)[:-1]

    def close(self) -> None:
        self.h5obj.close()


class CrDirReader(CrReader):
    def __init__(self, loc):
        self.loc: str = loc.rstrip('/') + '/'
        self.matFn = None
        super().__init__(self._handle_version())

    def _handle_version(self):
        if os.path.isfile(self.loc + 'features.tsv.gz'):
            self.matFn = self.loc + 'matrix.mtx.gz'
            grps = {'featureIds': ('features.tsv.gz', 0),
                    'featureNames': ('features.tsv.gz', 1),
                    'featureTypes': ('features.tsv.gz', 2),
                    'cellNames': ('barcodes.tsv.gz', 0)}
        elif os.path.isfile(self.loc + 'genes.tsv'):
            self.matFn = self.loc + 'matrix.mtx'
            grps = {'featureIds': ('genes.tsv', 0),
                    'featureNames': ('genes.tsv', 1),
                    'featureTypes': None, 'cellNames': ('barcodes.tsv', 0)}
        else:
            raise IOError("ERROR: Couldn't find files")
        return grps

    def _read_dataset(self, key: Optional[str] = None):
        return [x.split('\t')[self.grpNames[key][1]] for x in
                FileReader(self.loc + self.grpNames[key][0],
                           by_line=True).consume()]

    def consume(self, mtx_chunk_size: int = int(1e5)) -> \
            Generator[List[np.ndarray], None, None]:
        split_idx = self.assayFeats.T.end.values
        mtx = MtxReader(self.matFn, self.nFeatures, mtx_chunk_size)
        for a in mtx.consume():
            yield np.split(a, split_idx)[:-1]


class CrToZarr:
    def __init__(self, cr: CrReader, zarr_fn: str):
        self.cr = cr
        self.z = self._initialize_store(zarr_fn, (1000, 1000))

    def _initialize_store(self, fn: str,
                          chunk_shape: Tuple[int, int]) -> zarr.Group:
        z = zarr.open(fn, mode='w')
        for assay in self.cr.assayFeats.columns:
            g = z.create_group(assay)
            g.create_dataset('counts', chunks=chunk_shape, dtype='i4',
                             shape=(self.cr.nCells,
                                    self.cr.assayFeats[assay]['nFeatures']))
            for i in ['featureIds', 'featureNames', 'cellNames']:
                data = self.cr.__getattribute__(i)(assay)
                dtype = 'U' + str(max([len(x) for x in data]))
                g.create_dataset(i, data=data, shape=len(data), dtype=dtype)
        return z

    def _read_stream(self, show_progress: bool) -> \
            Generator[np.ndarray, None, None]:
        if show_progress:
            for i in tqdm(self.cr.consume(), total=self.cr.nCells):
                yield i
        else:
            for i in self.cr.consume():
                yield i

    def dump(self, chunk_size: int = 100, show_progress: bool = True) -> None:
        stores = [self.z["%s/counts" % x] for x in self.cr.assayFeats.columns]
        n_stores = len(stores)
        chunk, p = [], 0
        for n, i in enumerate(self._read_stream(show_progress), start=1):
            chunk.append(i)
            if n % chunk_size == 0 or n == self.cr.nCells:
                for j in range(n_stores):
                    stores[j][p:n] = [x[j] for x in chunk]
                chunk, p = [], n


class SubSetZarr:
    def __init__(self, in_fn, out_fn, cells):
        self.iz = zarr.open(in_fn, mode='r')
        self.oz = zarr.open(out_fn, mode='w')
        self.cells, self.cellsIdx = self._get_cell_idx(cells)
        self.nCells = len(self.cells)
        self.nFeatures = {x: self.iz[x]['featureIds'].shape[0] for x in
                          self.iz.group_keys()}
        self._initialize_store((1000, 1000))

    def _get_cell_idx(self, cells):
        g = list(self.iz.group_keys())[0]
        idx_map = {x: n for n, x in enumerate(self.iz['%s/cellNames' % g][:])}
        valid_cells, idx = [], []
        for i in cells:
            if i in idx_map:
                valid_cells.append(i)
                idx.append(idx_map[i])
        return valid_cells, idx

    def _initialize_store(self, chunk_shape: Tuple[int, int]) -> None:
        for assay in self.iz.group_keys():
            g = self.oz.create_group(assay)
            g.create_dataset('counts', chunks=chunk_shape, dtype='i4',
                             shape=(self.nCells, self.nFeatures[assay]))
            cell_dtype = 'U' + str(max([len(x) for x in self.cells]))
            g.create_dataset('cellNames', data=self.cells, shape=self.nCells,
                             dtype=cell_dtype)
            for i in ['featureIds', 'featureNames']:
                data = self.iz[assay + '/' + i][:]
                g.create_dataset(i, data=data, shape=len(data),
                                 dtype=data.dtype)

    def _read_stream(self, chunk_size: int, show_progress: bool):
        chunk_size = int(self.nCells / chunk_size) + 1
        chunks = np.array_split(self.cellsIdx, chunk_size)
        if show_progress:
            for chunk in tqdm(chunks):
                yield chunk
        else:
            for chunk in chunks:
                yield chunk

    def dump(self, chunk_size: int = 100, show_progress: bool = True) -> None:
        for assay in self.iz.group_keys():
            slot = '%s/counts' % assay
            gos = self.iz[slot].get_orthogonal_selection
            start, end = 0, 0
            for chunk in self._read_stream(chunk_size, show_progress):
                start = end
                end = start + len(chunk)
                self.oz[slot][start:end] = gos((chunk, slice(None)))
