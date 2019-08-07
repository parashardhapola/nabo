import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

__all__ = ['correct_var']


def correct_var(stats, n_bins: int = 100,
                lowess_frac: float = 0.4) -> pd.Series:
    stats = stats[stats.use][['mu', 'sigmas']].apply(np.log)
    bin_edges = np.histogram(stats.mu, bins=n_bins)[1]
    bin_edges[-1] += 0.1  # For including last gene
    bin_genes = []
    for i in range(n_bins):
        idx: pd.Series = (stats.mu >= bin_edges[i]) & \
                         (stats.mu < bin_edges[i + 1])
        if sum(idx) > 0:
            bin_genes.append(list(idx[idx].index))
    bin_vals = []
    for genes in bin_genes:
        temp_stat = stats.reindex(genes)
        temp_gene = temp_stat.idxmin().sigmas
        bin_vals.append(
            [temp_stat.sigmas[temp_gene], temp_stat.mu[temp_gene]])
    bin_vals = np.array(bin_vals).T
    bin_cor_fac = lowess(bin_vals[0], bin_vals[1], return_sorted=False,
                         frac=lowess_frac, it=100).T
    fixed_var = {}
    for bcf, genes in zip(bin_cor_fac, bin_genes):
        for gene in genes:
            fixed_var[gene] = np.e ** (stats.sigmas[gene] - bcf)
    return pd.Series(fixed_var)
