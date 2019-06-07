import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # Needed for 3D UMAPS
import seaborn as sns
import numpy as np
from typing import Dict, List
plt.style.use('fivethirtyeight')
from natsort import natsorted, ns
import pandas as pd

__all__ = ['plot_summary_data', 'plot_mean_var', 'plot_scree',
           'plot_cluster_scores', 'plot_target_class_counts',
           'plot_box_exp', 'plot_3dumap']


def clean_axis(ax, ts=11, ga=0.4):
    ax.xaxis.set_tick_params(labelsize=ts)
    ax.yaxis.set_tick_params(labelsize=ts)
    for i in ['top', 'bottom', 'left', 'right']:
        ax.spines[i].set_visible(False)
    ax.grid(which='major', linestyle='--', alpha=ga)
    ax.figure.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return True


def plot_summary_data(data, plot_names, color: str,
                      display_stats: bool, savename: str,
                      showfig: bool = True) -> None:
    fig, axis = plt.subplots(1, len(plot_names), figsize=(13, 3))
    for i in range(len(plot_names)):
        val = np.array(data[i])
        ax = axis[i]
        sns.violinplot(val, ax=ax, linewidth=1, orient='v',
                       inner=None, cut=0, color=color)
        sns.stripplot(val, jitter=0.4, ax=ax, orient='v',
                      s=1.1, color='k', alpha=0.4)
        ax.set_ylabel(plot_names[i], fontsize=13)
        if display_stats:
            if i < 2:
                ax.set_title('Min: %d, Max: %d, Median: %d' % (
                    val.min(), val.max(), int(np.median(val))), fontsize=8)
            else:
                ax.set_title('Min: %.1f, Max: %.1f, Median: %.1f' % (
                    val.min(), val.max(), int(np.median(val))), fontsize=8)
        clean_axis(ax)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=300)
    if showfig:
        plt.show()
    else:
        plt.close()
    return None


def plot_mean_var(data, hvg_bool, min_dt, min_et, max_dt, max_et,
                  baseline):
    ax_label_fs = 12
    fig, axis = plt.subplots(1, 3, figsize=(11, 3))
    ax = axis[0]
    ax.scatter(data.nzm[~hvg_bool].values,
               data.fixed_var[~hvg_bool].values, s=10, alpha=0.6, c='grey')
    ax.scatter(data.nzm[hvg_bool].values,
               data.fixed_var[hvg_bool].values,
               s=10, alpha=0.6, c='dodgerblue')
    ax.axhline(min_dt, lw=1, ls='--', c='r')
    ax.axvline(min_et, lw=1, ls='--', c='r')
    ax.axhline(max_dt, lw=1, ls='--', c='r')
    ax.axvline(max_et, lw=1, ls='--', c='r')

    ax.set_xlabel('Log mean non-zero expression', fontsize=ax_label_fs)
    ax.set_ylabel('Log corrected variance', fontsize=ax_label_fs)
    clean_axis(ax)

    ax = axis[1]
    ax.scatter(data.m[~hvg_bool].values,
               data.fixed_var[~hvg_bool].values,
               s=10, alpha=0.6, c='grey')
    ax.scatter(data.m[hvg_bool].values,
               data.fixed_var[hvg_bool].values,
               s=10, alpha=0.6, c='dodgerblue')
    ax.set_xlabel('Log mean expression', fontsize=ax_label_fs)
    ax.set_ylabel('Log corrected variance', fontsize=ax_label_fs)
    clean_axis(ax)

    ax = axis[2]
    ax.scatter(data.m[~hvg_bool].values,
               data.variance[~hvg_bool].values,
               s=10, alpha=0.6, c='grey')
    ax.scatter(data.m[hvg_bool].values,
               data.variance[hvg_bool].values,
               s=10, alpha=0.6, c='dodgerblue')
    ax.scatter(baseline[0], baseline[1], s=5, c='crimson')
    ax.set_xlabel('Log mean expression', fontsize=ax_label_fs)
    ax.set_ylabel('Log variance', fontsize=ax_label_fs)
    clean_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_scree(pca, var_target, plot_comps):
    var_ratios = pca.explained_variance_ratio_[:plot_comps]
    y_line = int(np.argmin(abs(np.cumsum(var_ratios) - var_target)))
    # noinspection PyTypeChecker
    fig, axis = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax = axis[0]
    ax.plot(var_ratios, c='r')
    ax.set_ylabel('Fraction\nvariance explained', fontsize=13)
    ax.axvline(y_line, lw=2, ls='-.', c='green')
    clean_axis(ax)

    ax = axis[1]
    ax.plot(np.cumsum(var_ratios))
    plot_comps = min(plot_comps, len(var_ratios))
    ax.set_xticks(range(0, plot_comps, 2))
    ax.set_xticklabels(["%d" % (x + 1) for x in range(0, plot_comps, 2)])
    ax.set_xlabel('Principal component', fontsize=13)
    ax.set_ylabel('Cumulative fraction\nvariance explained', fontsize=13)
    ax.axhline(var_target, lw=1, ls='--', c='k')
    ax.axvline(y_line, lw=2, ls='-.', c='green')

    clean_axis(ax)
    plt.tight_layout()
    print("Variance target closest at PC%d" % (y_line+1))
    plt.show()


def plot_cluster_scores(values: Dict, clusters: Dict, sort: bool = False,
                        order: List = None, show_outliers: bool = False,
                        vertical: bool = False, fig_size: tuple = None,
                        x_ticks: List = None, x_lim: tuple = None,
                        tick_fs: int = 10, label_fs: int = 12,
                        tlabel_rotation: int = 0, default_color: str = None,
                        colors: Dict = None, cmap: str = 'hls',
                        save_name: str = None, display: bool = True) -> None:
    """

    :param values:
    :param clusters:
    :param sort:
    :param order:
    :param show_outliers:
    :param vertical:
    :param fig_size:
    :param x_ticks:
    :param x_lim:
    :param tick_fs:
    :param label_fs:
    :param tlabel_rotation:
    :param cmap:
    :param colors:
    :param save_name:
    :param display:
    :return:
    """
    scores = pd.DataFrame([pd.Series(values, name='values'),
                           pd.Series(clusters, name='clusters')]).T
    if order is None:
        order = np.array(natsorted(scores.clusters.unique()))
    scores = scores.groupby('clusters')
    values = np.array([scores.get_group(x)['values'].values for x in order])
    idx = None
    if sort is True:
        idx = np.argsort([np.mean(x) for x in values])
        order = order[idx]
        values = values[idx]
    if vertical == sort:
        order = order[::-1]
        values = values[::-1]
        if idx is not None:
            idx = idx[::-1]
        elif vertical is False:
            idx = list(range(len(values)))[::-1]

    if fig_size is None:
        if vertical is False:
            fig_size = (4, len(values) / 4)
        else:
            fig_size = (len(values) / 4, 4)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    if show_outliers:
        sym = '+'
    else:
        sym = ''
    bp = ax.boxplot(values, sym=sym, patch_artist=True, vert=vertical)
    if colors is None:
        colors = np.array(sns.color_palette(cmap, len(values)))
        if idx is not None:
            colors = colors[idx]
    else:
        if default_color is None:
            default_color = 'grey'
        temp = []
        for i in order:
            if i not in colors:
                print("WARNING: cluster %s is not present in "
                      "color dict" % str(i))
                temp.append(default_color)
            else:
                temp.append(colors[i])
        colors = [x for x in temp]
    for i, j in zip(bp['boxes'], colors):
        i.set_facecolor(j)
    plt.setp(bp['medians'], color='k')
    if vertical is True:
        ax.set_ylabel('Mapping score', fontsize=label_fs)
        ax.set_xticklabels(order, rotation=tlabel_rotation, ha='center',
                           va='center')
    else:
        ax.set_xlabel('Mapping score', fontsize=label_fs)
        ax.set_yticklabels(order, rotation=tlabel_rotation, ha='center',
                           va='center')
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    clean_axis(ax, ts=tick_fs)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, dpi=300)
    if display:
        plt.show()
    else:
        plt.close()


def plot_target_class_counts(values: Dict, ref_values: Dict,
                             vertical: bool = False, dropna: bool = False,
                             sort: bool = True, percent: bool = True,
                             enrichment: bool = False, order: bool = None,
                             line_kw: Dict = None, na_label: str = 'NA',
                             cmap: str ='hls', fig_size: tuple = None,
                             tick_fs: int = 10, label_fs: int = 12,
                             tlabel_rotation: int = 0, save_name: str = None,
                             display: bool = True) -> None:
    """

    :param values:
    :param ref_values:
    :param vertical:
    :param dropna:
    :param sort:
    :param percent:
    :param enrichment:
    :param order:
    :param line_kw:
    :param na_label:
    :param cmap:
    :param fig_size:
    :param tick_fs:
    :param label_fs:
    :param tlabel_rotation:
    :param save_name:
    :param display:
    :return:
    """
    values = pd.Series(values).value_counts()
    ref_values = pd.Series(ref_values).value_counts()

    for i in set(ref_values.index).difference(values.index):
        values[i] = 0
    if dropna is True:
        values = values.drop(na_label, errors='ignore')
    label = '#cells in cluster'
    if enrichment is True:
        values = (values / values.sum()) / (ref_values / ref_values.sum())
        values.fillna(0, inplace=True)
        label = 'Relative enrichment'
    elif percent is True:
        values = 100*values / values.sum()
        label = '%cells in cluster'

    if order is None:
        order = np.array(natsorted(values.index))
    values = np.array([values[x] for x in order])
    colors = np.array(sns.color_palette(cmap, len(values)))

    if sort is True:
        idx = np.argsort(values)[::-1]
    else:
        idx = list(range(len(values)))
    if vertical is False:
        idx = idx[::-1]
    values = values[idx]
    colors = colors[idx]
    order = order[idx]

    if fig_size is None:
        if vertical:
            fig_size = (5, 3)
        else:
            fig_size = (3, 5)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    if vertical is True:
        ax.bar(range(len(order)), values, color=colors)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=tlabel_rotation, ha='center',
                           va='top')
        ax.set_ylabel(label, fontsize=label_fs)
    else:
        ax.barh(range(len(order)), values, color=colors)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, rotation=tlabel_rotation, ha='right',
                           va='center')
        ax.set_xlabel(label, fontsize=label_fs)

    if enrichment is True:
        if line_kw is None:
            line_kw = {'lw': 1, 'ls': '--', 'color': 'k'}
        if vertical:
            ax.axhline(1, **line_kw)
        else:
            ax.axvline(1, **line_kw)
    clean_axis(ax, ts=tick_fs)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, dpi=300)
    if display:
        plt.show()
    else:
        plt.close()


def plot_box_exp(dataset, gene, groups, group_names,
                 save_name=None, dpi=200, show_rest: bool = True) -> None:
    if len(groups) != len(group_names):
        raise ValueError('Number of groups names not same as number of groups')
    exp = dataset.get_norm_exp(gene, as_dict=True)
    rest_keys = {x: None for x in exp}
    vals = []
    for g in groups:
        temp = []
        for i in g:
            if i in exp:
                temp.append(exp[i])
                del rest_keys[i]
        vals.append(temp)
    if show_rest:
        vals.append([exp[x] for x in rest_keys])
        group_names = group_names + ['Rest']

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    bp = ax.boxplot(vals, widths=0.8, sym='', patch_artist=True,
                    notch=True, medianprops=dict(linewidth=1, color='k'))
    for box in bp['boxes']:
        box.set(color='grey', linewidth=2)
    ax.set_title(gene, fontsize=15)
    ax.set_ylabel('Normalized expression', fontsize=14)
    clean_axis(ax)
    ax.set_xticklabels(group_names, fontsize=14)
    plt.tight_layout()
    # lgd = ax.legend(bp['boxes'], group_names,
    #                 loc=(1.1, 0.5), frameon=False, fontsize=12)
    if save_name is not None:
        # fig.savefig(save_name, bbox_extra_artists=(lgd,),
        #             bbox_inches='tight', dpi=dpi, transparent=True)
        fig.savefig(save_name, bbox_inches='tight', dpi=dpi, transparent=True)
    plt.show()
    return None


def plot_3dumap(dims, figsize=(6, 6), vc='k', vs=15,
                cmap='magma', ec=None, lw=0.1,
                title=None, title_fs=10,
                text_coords=None, text_fs=9,
                axis_labels=None, axis_labels_fs=8,
                alpha=0.7, grid_alpha=0.4,
                cbar=False, orient=None,
                savename=None, dpi=300):
    if cbar:
        fig = plt.figure(figsize=(figsize[0]+1, figsize))
    else:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    p = ax.scatter3D(dims.Dim1, dims.Dim2, dims.Dim3, s=vs, c=vc, cmap=cmap,
                     linewidths=lw, edgecolor=ec, alpha=alpha)
    ax.figure.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    for i in [ax.xaxis, ax.yaxis, ax.zaxis]:
        i.set_pane_color((1.0, 1.0, 1.0, 0.0))
        i._axinfo["grid"]['linestyle'] = '--'
        i._axinfo["grid"]['color'] = (0, 0, 0, 0.1)
        i.set_tick_params(labelsize=6)
    ax.grid(which='major', linestyle='--', alpha=grid_alpha)
    if axis_labels is None:
        axis_labels = ['UMAP1', 'UMAP2', 'UMAP3']
    ax.set_xlabel(axis_labels[0], fontsize=axis_labels_fs)
    ax.set_ylabel(axis_labels[1], fontsize=axis_labels_fs)
    ax.set_zlabel(axis_labels[2], fontsize=axis_labels_fs)
    if orient is not None:
        ax.view_init(orient[0], orient[1])
    if title is not None:
        ax.set_title(title, fontsize=title_fs)
    if text_coords is not None:
        for i in text_coords:
            v = text_coords[i].values
            ax.text(v[0], v[1], v[2], i, fontsize=text_fs)
    if cbar:
        fig.colorbar(p)
    if savename is None:
        plt.show()
    else:
        fig.savefig(savename, dpi=dpi)
        plt.close()
