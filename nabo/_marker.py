from typing import List, Dict
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm import tqdm
from ._dataset import Dataset
from ._graph import Graph
import numpy as np
from collections import Counter

__all__ = ['run_de_test', 'find_cluster_markers', 'show_full_table']

tqdm_bar = '{l_bar} {remaining}'


def run_de_test(dataset1: Dataset, dataset2,
                test_cells: List[str], control_cells: List[List[str]],
                test_label: str = None,  control_group_labels: list = None,
                exp_frac_thresh: float = 0.25, log2_fc_thresh: float = 1,
                qval_thresh: float = 0.05, tqdm_msg: str = '') -> pd.DataFrame:
    """
    Identifies differentially expressed genes using Mann Whitney U test.

    :param dataset1: nabo.Dataset instance
    :param dataset2: nabo.Dataset instance or None
    :param test_cells: list of cells for which markers has to be found.
                       These could be cells from a cluster,cells with high
                       mapping score, etc
    :param control_cells: List of cell groups against which markers need to
                          be found. This could just one groups of cells or
                          multiple groups of cells.
    :param test_label: Label for test cells.
    :param control_group_labels: Labels of control cell groups
    :param exp_frac_thresh: Fraction of cells that should have a non zero
                            value for a gene.
    :param log2_fc_thresh: Threshold for log2 fold change
    :param qval_thresh: Threshold for adjusted p value
    :param tqdm_msg: Message to print while displaying progress
    :return: pd.Dataframe
    """

    test_cells_idx = [dataset1.cellIdx[x] for x in test_cells]
    control_cells_idx_group = []
    for i in control_cells:
        if dataset2 is None:
            control_cells_idx_group.append([dataset1.cellIdx[x] for x in i])
        else:
            control_cells_idx_group.append([dataset2.cellIdx[x] for x in i])
    if test_label is None:
        test_label = 'Test group'
    if control_group_labels is None:
        control_group_labels = ['Ctrl group %d' % x for x in range(len(
                                 control_cells_idx_group))]
    num_test_cells = len(test_cells_idx)
    num_groups = len(control_cells_idx_group)
    min_n = [min(num_test_cells, len(x)) for x in control_cells_idx_group]
    n1n2 = [num_test_cells * x for x in min_n]

    if dataset2 is None:
        valid_genes = {dataset1.genes[x]: None for x in dataset1.keepGenesIdx}
    else:
        valid_genes = {}
        control_gene_list = {x: None for x in dataset2.genes}
        for i in dataset1.keepGenesIdx:
            gene = dataset1.genes[i]
            if gene in control_gene_list:
                valid_genes[gene] = None
        del control_gene_list

    de = []
    for gene in tqdm(valid_genes, bar_format=tqdm_bar, desc=tqdm_msg):
        rbc, mw_p, log_fc = 0, 1, 0

        all_vals = dataset1.get_norm_exp(gene)
        test_vals = all_vals[test_cells_idx]
        ef = np.nonzero(test_vals)[0].shape[0] / num_test_cells
        if ef < exp_frac_thresh:
            continue

        if dataset2 is None:
            all_control_vals = all_vals
        else:
            all_control_vals = dataset2.get_norm_exp(gene)

        log_mean_test_vals = np.log2(test_vals.mean())
        for i in range(num_groups):
            control_vals = all_control_vals[control_cells_idx_group[i]]
            control_vals.sort()
            control_vals = control_vals[-min_n[i]:]

            mean_control_vals = control_vals.mean()
            if mean_control_vals == 0:
                log_fc = np.inf
            else:
                log_fc = log_mean_test_vals - np.log2(mean_control_vals)
            if log_fc < log2_fc_thresh:
                continue
            try:
                u, mw_p = mannwhitneyu(test_vals, control_vals)
            except ValueError:
                pass
            else:
                rbc = 1 - ((2 * u) / n1n2[i])
            de.append((gene, ef, control_group_labels[i], rbc, log_fc, mw_p))

    de = pd.DataFrame(de, columns=['gene', 'exp_frac', 'versus_group',
                                   'rbc', 'log2_fc', 'pval'])
    if de.shape[0] > 1:
        de['qval'] = multipletests(de['pval'].values, method='fdr_bh')[1]
    else:
        de['qval'] = [np.nan for _ in range(de.shape[0])]
    de['test_group'] = [test_label for _ in range(de.shape[0])]
    out_order = ['gene', 'exp_frac', 'test_group', 'versus_group',
                 'rbc', 'log2_fc', 'pval', 'qval']
    de = de[out_order].sort_values(by='qval')
    return de[(de.qval < qval_thresh)].reset_index().drop(columns=['index'])


def find_cluster_markers(clusters: dict, dataset: Dataset,
                         de_frequency: int, exp_frac_thresh: float = 0.25,
                         log2_fc_thresh: float = 0.5,
                         qval_thresh: float = 0.05) -> (
                            pd.DataFrame, Dict[int, List[str]]):
    """
    Identifies marker genes for each cluster in a Graph. This function works a
    wrapper for `run_de_test`.

    :param clusters: dict
    :param dataset: nabo.Dataset
    :param de_frequency: Minimum number of clusters against a gene should be
                         significantly differentially expressed for it to
                         qualify as a marker
    :param exp_frac_thresh: Fraction of cells that should have a non zero
                            value for a gene.
    :param log2_fc_thresh: Threshold for log2 fold change
    :param qval_thresh: Threshold for adjusted p value
    :return: A tuple where first element is a pandas DataFrame and second
             element is a dictionary where keys are cluster numbers and values
             are lists of marker genes for the corresponding clusters
    """
    cluster_groups = {}
    for k, v in clusters.items():
        if v not in cluster_groups:
            cluster_groups[v] = []
        cluster_groups[v].append(k.rsplit('_', 1)[0])
    if de_frequency >= len(cluster_groups):
        de_frequency = len(cluster_groups) - 1
        print("WARNING: Value of 'de_frequency' reset to %d as number of "
              "clusters are %d" % (de_frequency, len(cluster_groups)))

    de_tables = []
    de_genes = {}
    cluster_set = sorted(set(cluster_groups.keys()))
    for i in cluster_set:
        j = sorted(set(cluster_set).difference([i]))
        test_cells = cluster_groups[i]
        control_cells = [cluster_groups[x] for x in j]

        de = run_de_test(dataset, None, test_cells, control_cells,
                         'Cluster %s' % str(i),
                         ['Cluster %s' % str(x) for x in j],
                         exp_frac_thresh=exp_frac_thresh,
                         log2_fc_thresh=log2_fc_thresh,
                         qval_thresh=qval_thresh,
                         tqdm_msg='Finding markers for cluster %s' % str(i))
        de_tables.append(de.copy())
        de_genes[i] = [k for k, v in Counter(de['gene']).items()
                       if v >= de_frequency]
    return pd.concat(de_tables).reset_index().drop(columns=['index']), de_genes


def show_full_table(df):
    """
    Display full DataFrame

    :param df: A pandas DataFrame
    :return:
    """
    with pd.option_context('display.max_rows', None):
        print(df)
