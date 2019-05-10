import h5py
from typing import List, Dict
import networkx as nx
import numpy as np
from fa2 import ForceAtlas2
from itertools import combinations
import operator
import os
from collections import Counter
import pandas as pd
import hac
import json
from tqdm import tqdm

__all__ = ['Graph']


class Graph(nx.Graph):
    """
    Class for storing Nabo's SNN graph. Inherits from networkx's `Graph` class

    """

    def __init__(self):
        super().__init__()
        self.refName = None
        self.refNodes: List[str] = []
        self.refG = None
        self.targetNames: List[str] = []
        self.targetNodes: Dict[str, List[str]] = {}
        self.deTestCells: List[str] = None
        self.deCtrlCells: List[str] = None
        self._agglomDendrogram = None

    def load_from_h5(self, fn: str, name: str, kind: str) -> None:
        """
        Loads a graph saved by `Mapping` class in HDF5 format

        :param fn: Path to HDF5 file
        :param name: Label/name of sample used in Mapping object. This
                     function assumes that the group in HDF5 containing
                     graph data is named: `name` + '_graph'
        :param kind: Can have a value of either 'reference' or 'target'.
                     Only be one sample can have kind='reference' for an
                     instance of this class
        :return: None
        """
        if os.path.exists(fn) is False:
            raise IOError('ERROR: File %s does not exist' % fn)
        if kind == 'reference':
            if self.refName is not None:
                raise ValueError('ERROR: A reference kind is already loaded')
        elif kind == 'target':
            if name in self.targetNames:
                raise ValueError('ERROR: %s target group already present in '
                                 'graph' % name)
            if self.refName is None:
                raise ValueError('ERROR: Please load reference kind first')
        else:
            raise ValueError('ERROR: Kind can be either "reference" or '
                             '"target"')
        try:
            h5 = h5py.File(fn, mode='r')
        except (IOError, OSError):
            raise IOError('ERROR: Unable to open file %s' % fn)
        if kind == 'reference':
            try:
                saved_name = h5['name_stash/ref_name'][0].decode('UTF-8')
                uid = h5['name_stash/ref_name'][1].decode('UTF-8')
            except KeyError:
                raise KeyError("ERROR: Could not find stashed names in the "
                               "mapping file. Make sure reference graph has "
                               "been created in the mapping file")
            if name != saved_name:
                raise KeyError("ERROR: The reference is named %s in the "
                               "mapping file and not %s. Please verify that "
                               "you are trying to load right reference." % (
                                saved_name, name))
        else:
            try:
                target_names = h5['name_stash/target_names'][:]
            except KeyError:
                raise KeyError("ERROR: Could not find stashed names in the "
                               "mapping file. Make sure reference graph has "
                               "been created in the mapping file")
            uid = None
            for i in target_names:
                if i[0].decode('UTF-8') == name:
                    uid = i[1].decode('UTF-8')
            if uid is None:
                raise KeyError("ERROR: The target name not could not be found "
                               "in the mapping file")
        grp = uid + '_graph'
        if grp not in h5:
            h5.close()
            raise KeyError('ERROR: Group %s not found in HDF5 file %s'
                           % (grp, fn))
        attrs = {'kind': kind, 'name': name}
        existing_nodes = {x: None for x in self.nodes()}
        new_nodes = []
        for node in h5[grp]:
            if node in existing_nodes:
                print('WARNING: node %s already present in the graph. Will '
                      'not add.' % node)
            else:
                new_nodes.append(node)
                self.add_node(node, **attrs)
                for j in h5[grp][node]:
                    node2 = j[0].decode('UTF-8')
                    weight = float(j[1].decode('UTF-8'))
                    self.add_edge(node, node2, weight=weight)
        h5.close()
        if kind == 'reference':
            self.refName = name
            self.refNodes = new_nodes
            self.refG = self.subgraph(self.refNodes)
        else:
            self.targetNames.append(name)
            self.targetNodes[name] = new_nodes
        return None

    def load_from_gml(self, fn: str) -> None:
        """
        Load data from GML format file. It is critical that this graph was
        generated using Nabo's `Mapping` class.

        :param fn: Full path of GML file
        :return: None
        """
        if len(self.nodes) > 0:
            raise ValueError('ERROR: The graph already contains nodes. '
                             'Cannot load GML file on this object.')
        if os.path.exists(fn) is False:
            raise IOError('ERROR: File %s does not exist' % fn)
        try:
            g = nx.read_gml(fn)
        except (IOError, OSError):
            raise IOError('ERROR: Could open the file %s. Make sure the file '
                          'is in GML format' % fn)
        for i in g.nodes(data=True):
            attrs = i[1]
            if 'kind' not in attrs or 'name' not in attrs:
                self.clear()
                raise ValueError('ERROR: Attributes "kind" and/or "name" '
                                 'not found for one or more cells. Make '
                                 'sure that the GML was saved using Nabo')
            if attrs['kind'] == 'reference':
                if self.refName is None:
                    self.refName = attrs['name']
                elif self.refName != attrs['name']:
                    self.clear()
                    raise ValueError('ERROR: Multiple reference samples '
                                     'found. Please make sure you saved '
                                     'the GML with Nabo.')
                self.refNodes.append(i[0])
            elif attrs['kind'] == 'target':
                if attrs['name'] not in self.targetNames:
                    self.targetNames.append(attrs['name'])
                    self.targetNodes[attrs['name']] = []
                self.targetNodes[attrs['name']].append(i[0])
            else:
                self.clear()
                raise ValueError('ERROR: Kind can only be either "reference" '
                                 'or "target"')
            if 'pos' in i[1] and i[1]['pos'] == 'None':
                i[1]['pos'] = None
            self.add_node(i[0], **i[1])
        for i in g.edges(data=True):
            self.add_edge(i[0], i[1], weight=i[2]['weight'])
        self.refG = self.subgraph(self.refNodes)
        return None

    def save_graph(self, save_name: str) -> None:
        """
        Save graph in GML format

        :param save_name: Output filename with path
        :return: None
        """
        nx.write_gml(self, save_name, stringizer=lambda x: str(x))
        return None

    def set_ref_layout(self, niter: int = 500, verbose: bool = True,
                       init_pos: dict = None, disable_rescaling: bool = False,
                       outbound_attraction_distribution: bool = True,
                       edge_weight_influence: float = 1.0,
                       jitter_tolerance: float = 1.0,
                       barnes_hut_optimize: bool = True,
                       barnes_hut_theta: float = 1.2,
                       scaling_ratio: float = 1.0,
                       strong_gravity_mode: bool = False,
                       gravity: float = 1.0) -> None:
        """
        Calculates a 2D graph layout using ForceAtlas2 algorithm.
        The ForceAtlas2 implementation being used here will not prevent
        nodes in the graph from overlapping with each other. We aim to
        improve this in the future.

        :param niter: Number of iterations (default: 500)
        :param verbose: Print the progress (default: True)
        :param init_pos: Initial positions of nodes
        :param disable_rescaling: If True then layout coordinates are not
                                  rescaled to only have non negative
                                  positions (Default: False)
        :param outbound_attraction_distribution:
        :param edge_weight_influence:
        :param jitter_tolerance:
        :param barnes_hut_optimize:
        :param barnes_hut_theta:
        :param scaling_ratio:
        :param strong_gravity_mode:
        :param gravity:
        :return: None
        """
        force_atlas = ForceAtlas2(
            outboundAttractionDistribution=outbound_attraction_distribution,
            edgeWeightInfluence=edge_weight_influence,
            jitterTolerance=jitter_tolerance,
            barnesHutOptimize=barnes_hut_optimize,
            barnesHutTheta=barnes_hut_theta, scalingRatio=scaling_ratio,
            strongGravityMode=strong_gravity_mode, gravity=gravity,
            verbose=verbose)
        pos = force_atlas.forceatlas2_networkx_layout(
            self.refG, pos=init_pos, iterations=niter)
        if disable_rescaling is False:
            pos_array = np.array(list(pos.values())).T
            min_x, min_y = pos_array[0].min(), pos_array[1].min()
            # max_x, max_y = pos_array[0].max(), pos_array[1].max()
            pos = {k: ((v[0] - min_x), (v[1] - min_y)) for k, v in pos.items()}
            # pos = {k: ((v[0] - min_x) / (max_x - min_x),
            #            (v[1] - min_y) / (max_y - min_y))
            #        for k, v in pos.items()}
        for node in pos:
            self.nodes[node]['pos'] = (float(pos[node][0]),
                                       float(pos[node][1]))
        for node in self:
            if node not in pos:
                self.nodes[node]['pos'] = None
        return None

    @property
    def clusters(self) -> Dict[str, str]:
        ret_val = {}
        for i in self.refG.nodes(data=True):
            if 'cluster' in i[1]:
                ret_val[i[0]] = i[1]['cluster']
        return ret_val

    def make_clusters(self, n_clusters: int) -> None:
        """
        Performs graph agglomerative clustering using algorithm in Newman 2004

        :param n_clusters: Number of clusters
        :return: None
        """
        if self._agglomDendrogram is None:
            clusterer = hac.GreedyAgglomerativeClusterer()
            self._agglomDendrogram = clusterer.cluster(self.refG)
        cluster_list = self._agglomDendrogram.clusters(n_clusters)
        for n, node_group in enumerate(cluster_list):
            for node in node_group:
                clust_num = n + 1
                self.nodes[node]['cluster'] = str(clust_num)
        return None

    def get_cluster_identity_weights(self) -> pd.Series:
        """

        :return: Cluster identity weights for each cell
        """
        ciw = {}
        clusters = self.clusters
        if clusters == {}:
            raise ValueError("ERROR: Please make sure that clusters have "
                             "been assigned to cells. Run 'make_clusters' or "
                             "import clusters")
        skipped_cells = 0
        max_nodes = []
        for i in tqdm(self.refG, total=len(self.refG)):
            cw = []
            for j in self.refG.edges(i, data=True):
                try:
                    cw.append((clusters[j[1]], j[2]['weight']))
                except KeyError:
                    skipped_cells += 1
                    continue
            if len(cw) > 0:
                cw = pd.DataFrame(cw)
                cw = cw.groupby(0).size() * \
                     cw.groupby(0).sum()[1].sort_values().values
                if len(cw) > 1:
                    ciw[i] = cw[-1] / cw[:-1].sum()
                else:
                    ciw[i] = cw[-1]
                    max_nodes.append(i)
            else:
                skipped_cells += 1
        if skipped_cells > 0:
            print("WARNING: %d cells were skipped" % skipped_cells)
        ciw = pd.Series(ciw)
        if len(max_nodes) > 0:
            for i in max_nodes:
                ciw[i] = ciw.max()
        return ciw

    def import_clusters(self, cluster_dict: Dict[str, str] = None,
                        missing_val: str = 'NA') -> None:
        """
        Import cluster information for reference cells.

        :param cluster_dict: Dictionary with cell names as keys and cluster
                             number as values. Cluster numbers should start
                             from 1
        :param missing_val: This value will be filled in when fill_missing
                            is True (Default: NA)
        :return: None
        """
        skipped_nodes = len(cluster_dict)
        for node in self.refNodes:
            if node in cluster_dict:
                self.nodes[node]['cluster'] = str(cluster_dict[node])
                skipped_nodes -= 1
            else:
                self.nodes[node]['cluster'] = missing_val
        if skipped_nodes > 0:
            print('WARNING: %d cells do not exist in the reference graph and '
                  'their cluster info was not imported.' % skipped_nodes)

    def import_clusters_from_json(self, fn):
        """
        Import clusters om JSON file

        :param fn: Input file in JSON format.
        :return: None
        """
        return self.import_clusters(json.load(open(fn)))

    def import_clusters_from_csv(self, csv: str, csv_sep: str = ',',
                                 cluster_col: int = 0, header = None,
                                 append_ref_name: bool = False):
        """
        :param csv: Filename containing cluster information. Make
                    sure that the first column contains cell names and
                    second contains the cluster labels.
        :param csv_sep: Separator for CSV file (default: ',')
        :param cluster_col: Column number (0 based count) where cluster
                            info is present (Default: 0)
        :param append_ref_name: Append the reference name to the cell name (
                                Default: True)
        :return: None
        """
        df = pd.read_csv(csv, index_col=0, sep=csv_sep, header=header)
        cluster_dict = df[df.columns[cluster_col]].to_dict()
        if append_ref_name:
            cluster_dict = {k + '_' + self.refName: v for
                            k, v in cluster_dict.items()}
        return self.import_clusters(cluster_dict)

    def save_clusters_as_json(self, outfn):
        """

        :param outfn: Output JSON file
        :return:
        """
        with open(outfn, 'w') as OUT:
            json.dump(self.clusters, OUT, indent=2)

    def save_clusters_as_csv(self, outfn):
        """

        :param outfn: Output CSV file
        :return:
        """
        pd.Series(self.clusters).to_csv(outfn)

    def _validate_clusters(self):
        nclusts = len(set(self.clusters.values()))
        if nclusts == 0:
            raise ValueError('ERROR: Calculate clusters first using '
                             '"make_clusters" or import clusters using '
                             '"import_clusters"')
        elif nclusts == 1:
            raise ValueError('ERROR: Cannot classify targets when only '
                             'one cluster is present in the graph')
        return True

    def calc_modularity(self) -> float:
        """
        Calculates modularity of the reference graph. The clusters should have
        already been defined.

        :return: Value between 0 and 1
        """
        partition = {}
        for k, v in self.clusters.items():
            if v not in partition:
                partition[v] = {}
            partition[v][k] = None
        partition = list(partition.values())
        if sum([len(x) for x in partition]) != len(self.refG):
            raise AssertionError('ERROR: Not all reference nodes have been '
                                 'assigned to a cluster. Cannot calculate '
                                 'modularity!')
        # noinspection PyCallingNonCallable
        w_degree = dict(self.refG.degree(weight='weight'))
        norm = 1 / (2 * self.refG.size(weight='weight'))
        q = 0
        for p in partition:
            for i in p:
                t = -w_degree[i] * norm
                q += sum([t * w_degree[x] for x in p])
                q += sum([self.refG[i][x]['weight']
                          for x in self.refG[i] if x in p])
        return q * norm

    def import_layout(self, pos_dict) -> None:
        """
         Alternatively one can provide a
        dictionary with keys as node name and values as coordinate (x,
        y) tuple.

        :param pos_dict: Dictionary with keys as node names and values as
                         2D coordinates of nodes on the graph.
        :return: None
        """
        skipped_nodes = len(pos_dict)
        error_nodes = 0
        for node in self.nodes:
            if node in pos_dict:
                try:
                    self.nodes[node]['pos'] = (
                        float(pos_dict[node][0]),
                        float(pos_dict[node][1])
                    )
                    skipped_nodes -= 1
                except (IndexError, TypeError):
                    error_nodes += 1
            else:
                self.nodes[node]['pos'] = None
        if skipped_nodes > 0:
            print('WARNING: %d cells do not exist in the reference graph and '
                  'their position info was not imported.' % skipped_nodes)
            if error_nodes > 0:
                print('WARNING: %d cells had position info in incorrect '
                      'format' % error_nodes)
        return None

    def import_layout_from_json(self, fn):
        """

        :param fn: Input json file
        :return:
        """
        return self.import_layout(json.load(open(fn)))

    def import_layout_from_csv(self, csv: str, csv_sep: str = ',',
                               dim_cols: tuple = (0, 1), header = None,
                               append_ref_name: bool = False):
        """
        Import graph layout coordinates from a CSV file

        :param csv: Filename containing layout coordinates. Make
                    sure that the first column contains cell names and
                    second and thrid contain the x and y coordinates
        :param csv_sep: Separator for CSV file (default: ',')
        :param append_ref_name: Append the reference name to the cell name (
                                Default: True)
        :return: None
        """
        layout = pd.read_csv(csv, index_col=0, sep=csv_sep, header=header)
        d1 = layout.columns[dim_cols[0]]
        d2 = layout.columns[dim_cols[1]]
        if append_ref_name:
            layout = {x + '_' + self.refName: (layout[d1][x], layout[d2][x])
                      for x in layout.index}
        else:
            layout = {x: (layout[d1][x], layout[d2][x]) for x in layout.index}
        return self.import_layout(layout)

    @property
    def layout(self):
        """
        Copies 'pos' attribute values (x/y coordinate tuple) from  graph nodes
        and returns a dictionary
        :return:
        """
        pos_dict = {}
        for i in self.nodes(data=True):
            try:
                pos_dict[i[0]] = i[1]['pos']
            except (KeyError, IndexError):
                pos_dict[i[0]] = None
        return pos_dict

    def save_layout_as_json(self, out_fn):
        """

        :param out_fn: Output json file
        :return:
        """
        with open(out_fn, 'w') as OUT:
            json.dump(self.layout, OUT, indent=2)
        return None

    def save_layout_as_csv(self, out_fn):
        """
        Saves the layout in CSV format

        :param out_fn: Output CSV file
        :return:
        """
        pd.DataFrame(self.layout).T.to_csv(out_fn, header=None)
        return None

    @staticmethod
    def get_score_percentile(score: Dict[str, int], p: int) -> float:
        """
        Get value for at a given percentile

        :param score: Mapping score or any other dictionary
                      where values are numbers
        :param p: Percentile
        :return: Percentile value
        """
        return np.percentile(list(score.values()), p)

    def get_mapping_score(self, target: str, min_weight: float = 0,
                          min_score: float = 0, weighted: bool = True,
                          by_cluster: bool = False,
                          sorted_names_only: bool = False,
                          top_n_only: int = None,
                          all_nodes: bool = True, score_multiplier: int = 1000,
                          ignore_nodes: List[str] = None,
                          include_nodes: List[str] = None,
                          remove_suffix: bool = False, verbose: bool = False):
        """
        Calculate a weighted/unweighted degree of incident target nodes on
        reference nodes.

        :param target: Target sample name
        :param min_weight: Ignore a edge if edge weight is smaller then this
                           value in the SNN graph. Only applicable if
                           calculating a weighted mapping score
        :param min_score: If score is smaller then reset score to zero
        :param weighted: Use edge weights if True
        :param by_cluster: If True, then combine scores from nodes of same
                           cluster into a list. The keys are cluster number
                           in the output dictionary
        :param sorted_names_only: If True, then return only sorted list of
                                  base cells from highest to lowest mapping
                                  score. Cells with mapping score less than
                                  `min_score` are not reported (default: False)
        :param top_n_only: If sorted_names_only is True and an integer value is
                           provided then this method will return top n
                           number of nodes sorted based on score. min_score
                           value will be ignored.
        :param all_nodes: if False, then returns only nodes with non-zero
                          score (after resetting using min_score)
        :param score_multiplier: Score is multiplied by this number after
                                 normalizing for total number of target cells.
        :param ignore_nodes: List of nodes from 'target' sample to be
                             ignored while calculating the score (default:
                             None).
        :param include_nodes: List of target nodes from 'target' sample.
                              Mapping score will be calculated ONLY for those
                              reference cells that are connected to this
                              subset of target cells in the graph. By
                              default mapping score will be calculated
                              against each target node.
        :param remove_suffix: Remove suffix from cell names (default: False)
        :param verbose: Prints graph stats
        :return: Mapping score
        """
        if by_cluster:
            if set(self.clusters.values()) == 'NA':
                raise ValueError('ERROR: Calculate clusters first using '
                                 '"make_clusters" or import clusters using '
                                 '"import_clusters"')
        if target not in self.targetNames:
            raise ValueError('ERROR: %s not present in graph' % target)
        if ignore_nodes is not None and include_nodes is not None:
            raise ValueError("ERROR: PLease provide only one of "
                             "either 'ignore_nodes' or 'include_nodes' at a "
                             "time")
        target_nodes = {x: None for x in self.targetNodes[target]}
        if ignore_nodes is None:
            ignore_nodes = []
        else:
            temp = []
            for node in ignore_nodes:
                if node in target_nodes:
                    temp.append(node)
            ignore_nodes = list(temp)
        if include_nodes is None:
            include_nodes = list(self.targetNodes[target])
        else:
            temp = []
            for node in include_nodes:
                if node in target_nodes:
                    temp.append(node)
            include_nodes = list(temp)

        include_nodes = list(set(include_nodes).difference(ignore_nodes))
        g = nx.Graph(self.subgraph(self.refNodes + include_nodes))
        g.remove_edges_from(self.refG.edges)
        if verbose:
            isolates = set(list(nx.isolates(g)))
            print("INFO: The bipartite graph has %d edges" % g.size())
            print("INFO: Mapping calculated against %d %s nodes" % (
                len(include_nodes), target))
            print("INFO: %d reference nodes do not connect with any target"
                  " node" % len(isolates.intersection(self.refNodes)))
            print("INFO: %d target nodes do not connect with any reference"
                  " node" % len(isolates.intersection(include_nodes)))
        score = {}
        for i in self.refNodes:
            score[i] = 0
            for j in g.edges(i, data=True):
                if weighted:
                    if j[2]['weight'] > min_weight:
                        score[i] += j[2]['weight']
                else:
                    score[i] += 1
        score = {k: score_multiplier * v / len(self.targetNodes[target])
                 for k, v in score.items()}

        if by_cluster:
            cluster_dict = self.clusters
            cluster_values = {x: [] for x in set(cluster_dict.values())}
            na_cluster_score = []
            for node in score:
                try:
                    cluster_values[cluster_dict[node]].append(score[node])
                except KeyError:
                    na_cluster_score.append(score[node])
            if len(na_cluster_score) > 0:
                if 'NA' not in cluster_values:
                    cluster_values['NA'] = []
                else:
                    print("WARNING: 'NA' cluster already exists. Appending "
                          "value to it")
                cluster_values['NA'].extend(na_cluster_score)
            return cluster_values

        if sorted_names_only:
            if top_n_only is not None:
                if top_n_only > len(score):
                    raise ValueError('ERROR: Value of top_n_only should be '
                                     'less than total number of nodes in '
                                     'reference graph')
                retval = [x[0] for x in sorted(score.items(),
                                               key=lambda x: x[1])][::-1][
                         :top_n_only]
            else:
                ms = {k: v for k, v in score.items() if v >= min_score}
                retval = [x[0] for x in sorted(ms.items(),
                                               key=lambda x: x[1])][::-1]
            if remove_suffix:
                return [x.rsplit('_', 1)[0] for x in retval]
            else:
                return retval
        if not all_nodes:
            retval = {k: v for k, v in score.items() if v >= min_score}
        else:
            retval = {k: v if v >= min_score else 0 for k, v in score.items()}
        if remove_suffix:
            return [x.rsplit('_', 1)[0] for x in retval]
        else:
            return retval

    def get_cells_from_clusters(self, clusters: List[str],
                                remove_suffix: bool = True) -> List[str]:
        """
        Get cell names for input cluster numbers

        :param clusters: list of cluster identifiers
        :param remove_suffix: Remove suffix from cell names
        :return: List of cell names
        """
        if set(self.clusters.values()) == 'NA':
            raise ValueError('ERROR: Calculate clusters first using '
                             '"make_clusters" or import clusters using '
                             '"import_clusters"')
        cells = []
        clusters = {str(x) for x in clusters}
        for k, v in self.clusters.items():
            if v in clusters:
                if remove_suffix:
                    cells.append(k.rsplit('_', 1)[0])
                else:
                    cells.append(k)
        return cells

    def classify_target(self, target: str, weight_frac: float = 0.5,
                        min_degree: int = 2, min_weight: float = 0.1,
                        cluster_dict: Dict[str, int] = None, na_label: str
                        = 'NA', ret_counts: bool = False) -> dict:
        """
        This classifier identifies the total weight of all the connections made
        by each target cell to each cluster (of reference cells). If a target
        cell has more than 50% (default value) of it's total connection weight
        in one of the clusters then the target cell is labeled to be from that
        cluster. One useful aspect of this classifier is that it will not
        classify the target cell to be from any cluster if it fails to reach
        the threshold (default, 50%) for any cluster (such target cell be
        labeled as '0' by default).

        :param target: Name of target sample
        :param weight_frac: Required minimum fraction of weight in a cluster
                            to be classified into that cluster
        :param min_degree: Minimum degree of the target node
        :param min_weight: Minimum edge weight. Edges with less weight
                           than min_weight will be ignored but will still
                           contribute to total weight.
        :param cluster_dict: Cluster labels for each reference cell. If not
                             provided then the stored cluster information is
                             used.
        :param na_label: Label for cells that failed to get classified
                           into any cluster
        :param ret_counts: It True, then returns number of target cells
                           classified to each cluster, else returns predicted
                           cluster for each target cell
        :return: Dictionary. Keys are target cell names and value their
                             predicted custer if re_Count is False. Otherwise,
                             keys are cluster labels and values are the number
                             of target cells classified to that cluster
        """
        if cluster_dict is None:
            self._validate_clusters()
            cluster_dict = self.clusters
        clusts = set(cluster_dict.values())
        classified_clusters = []
        degrees = dict(self.degree)
        for i in self.targetNodes[target]:
            if i not in degrees:
                continue
            if degrees[i] < min_degree:
                classified_clusters.append(na_label)
                continue
            clust_weights = {x: 0 for x in clusts}
            tot_weight = 0
            for j in self.edges(i, data=True):
                if j[2]['weight'] > min_weight and j[1] in cluster_dict:
                    clust_weights[cluster_dict[j[1]]] += j[2]['weight']
                tot_weight += j[2]['weight']  # even low weight is added to
                # total weight to allow poor mappings to be penalized.
            max_clust = max(clust_weights.items(),
                            key=operator.itemgetter(1))[0]
            if clust_weights[max_clust] > (weight_frac * tot_weight):
                classified_clusters.append(max_clust)
            else:
                classified_clusters.append(na_label)
        if ret_counts:
            counts = Counter(classified_clusters)
            if na_label not in counts:
                counts[na_label] = 0
            for i in set(cluster_dict.values()):
                if i not in counts:
                    counts[i] = 0
            return counts
        else:
            return dict(zip(self.targetNodes[target], classified_clusters))

    def get_mapping_specificity(self, target_name: str,
                                fill_na: bool = True) -> Dict[str, float]:
        """
        Calculates the mapping specificity of target nodes. Mapping
        specificity of a target node is calculated as the mean of shortest
        path lengths between all pairs of mapped reference nodes.

        :param target_name: Name of target sample
        :param fill_na: if True, then nan values will be replaced with
                        largest value (default: True)
        :return: Dictionary with target node names as keys and mapping
                 specificity as values
        """

        path_lengths = {}
        # TODO: FIX FOR MISSING TARGET NODES IF NODES ARE REMOVED MANUALLY
        for node in tqdm(self.targetNodes[target_name]):
            spls = []
            targets = [x[1] for x in self.edges(node)]
            nt = len(targets)
            for i in range(nt):
                for j in range(nt):
                    if i < j:
                        spls.append(nx.algorithms.shortest_path_length(
                            self.refG, source=targets[i], target=targets[j]))
            path_lengths[node] = float(np.mean(spls))
        if fill_na:
            max_val = max(path_lengths.values())
            return pd.Series(path_lengths).fillna(max_val).to_dict()
        else:
            return path_lengths

    def get_ref_specificity(self, target: str, target_values: Dict[str, float],
                            incl_unmapped: bool = False) -> Dict[str, float]:
        """
        Calculates the average mapping specificity of all target nodes that
        mapped to a given a reference node. Requires that the mapping
        specificity of target nodes is already calculated.

        :param target: Name of target sample
        :param target_values: Mapping specificity values of target nodes
        :param incl_unmapped: If True, then includes unmapped reference
                              nodes in the dictionary with value set at 0
                              Default: False)
        :return: Dictionary with reference node names as keys and values as
        mean mapping specificity of their mapped target nodes
        """

        back_prom = {}
        for i in self.refNodes:
            back_prom[i] = []
            for j in self.edges(i, data=True):
                if j[1][-len(target):] == target:
                    back_prom[i].append(target_values[j[1]])
        new_back_prom = {}
        for i in back_prom:
            if len(back_prom[i]) > 1:
                new_back_prom[i] = np.mean(back_prom[i])
            elif len(back_prom[i]) == 1:
                new_back_prom[i] = back_prom[i][0]
            else:
                if incl_unmapped:
                    new_back_prom[i] = 0
        return new_back_prom

    def get_mapped_cells(self, target: str, ref_cells: str,
                         remove_suffix: bool = True) -> List[str]:
        """
        Get target cells that map to a given list of reference cells.

        :param target: Name of target sample
        :param ref_cells: List of reference cell names
        :param remove_suffix: If True then removes target name suffix from
                              end of node name
        :return: List of target cell names
        """
        if target not in self.targetNames:
            raise ValueError('ERROR: %s not present in graph!' % target)
        target_cells = {x: None for x in self.targetNodes[target]}
        mapped_cells = []
        for i in ref_cells:
            if remove_suffix:
                i = i + '_' + self.refName
            for j in self.edges(i):
                if j[1] in target_cells:
                    mapped_cells.append(j[1])
        mapped_cells = list(set(mapped_cells))
        if remove_suffix:
            return [x.rsplit('_', 1)[0] for x in mapped_cells]
        else:
            return mapped_cells

    def get_random_nodes(self, n: int) -> List[str]:
        """
        Get random list of nodes from reference graph.

        :param n: Number of nodes to return
        :return: A list of reference nodes
        """
        all_nodes = list(self.refNodes)
        if n >= len(all_nodes):
            raise ValueError('ERROR: n should be lower than total nodes in '
                             'reference graph')
        random_nodes = []
        for i in range(n):
            x = np.random.choice(all_nodes)
            random_nodes.append(x)
            all_nodes.remove(x)
        return sorted(random_nodes)

    def calc_contiguous_spl(self, nodes: List[str]) -> float:
        """
        Calculates mean of shortest path lengths between subsequent nodes
        provided in the input list in reference graph.

        :param nodes: List of nodes from reference sample
        :return: Mean shortest path length
        """
        spl = []
        for i in range(len(nodes) - 1):
            spl.append(nx.shortest_path_length(self.refG, nodes[i],
                                               nodes[i + 1]))
        return float(np.mean(spl))

    def calc_diff_potential(self,
                            r: Dict[str, float]=None) -> Dict[str, float]:
        """
        Calculate differentiation potential of cells.

        This function is a reimplementation of population balance analysis
        (PBA) approach published in Weinreb et al. 2017, PNAS.
        This function computes the random walk normalized Laplacian matrix
        of the reference graph, L_rw = I-A/D and then calculates a
        Moore-Penrose pseudoinverse of L_rw. The method takes an optional
        but recommended parameter 'r' which represents the relative rates of
        proliferation and loss in different gene expression states (R). If
        not provided then a vector with ones is used. The differentiation
        potential is the dot product of inverse L_rw and R

        :param r: Same as parameter R in the above said reference. Should be a
                  dictionary with each reference cell name as a key and its
                  corresponding R values.
        :return: V (Vector potential) as dictionary. Smaller values
                 represent less differentiated cells.
        """
        adj = nx.to_pandas_adjacency(self.refG, weight=False)
        degree = adj.sum(axis=1)
        lap = np.identity(adj.shape[0]) - adj / np.meshgrid(degree, degree)[1]
        ilap = np.linalg.pinv(lap)
        if r is None:
            rvec = np.ones(ilap.shape[0])
        else:
            rvec = []
            for i in list(self.refG.nodes):
                if i in r:
                    rvec.append(r[i])
                else:
                    print('ERROR: %s node is missing in r')
                    return {}
            rvec = np.array(rvec)
        return dict(zip(adj.columns, np.dot(ilap, rvec)))

    def get_k_path_neighbours(self, nodes: List[str], k_dist: int,
                              full_trail: bool = False,
                              trail_start: int = 0) -> List[str]:
        """
        Get set of nodes at a given distance

        :param nodes: Input nodes
        :param k_dist: Path distance from input node
        :param full_trail: If True then returns only nodes at k_dist path
                           distance else return all nodes upto k_dist (
                           default: False)
        :param trail_start: If full_trail is True, then the trail starts at
                            this path distance (default: 0).
        :return: List of nodes
        """

        def get_neighbours(g, n):
            g.remove_edges_from(list(combinations(n, 2)))
            n = list(set(n).intersection(list(g.nodes())))
            return list(set(sum([list(g.neighbors(x)) for x in n], [])))

        neighbours = [list(nodes)]
        # copying because will modify the graph
        ref_g = nx.Graph(self.refG)
        for i in range(k_dist):
            neighbours.append(get_neighbours(ref_g, neighbours[i]))
            ref_g.remove_nodes_from(neighbours[i])
        if full_trail:
            return sum(neighbours[1 + trail_start:], [])
        else:
            return neighbours[-1]

    def set_de_groups(self, target: str, min_score: float,
                      node_dist: int, from_clusters: List[str] = None,
                      full_trail: bool = False, trail_start: int = 1,
                      stringent_control: bool = False) -> None:
        """
        Categorises reference nodes into either 'Test', 'Control' or 'Other'
        group. Nodes with mapping score higher than `min_score` are
        categorized as 'Test', cells at `node_dist` path distance are
        categorized as 'Control' and rest of the nodes are categorized as
        'Other'.

        :param target: Name of target sample whose corresponding mapping
                       scores to be considered
        :param min_score: Minimum mapping score
        :param node_dist: Path distance
        :param from_clusters: List of cluster number. 'Test' cells will only be
                              limited to these clusters.
        :param full_trail: If True then returns only nodes at `node_dist` path
                           distance else return all nodes upto `node_dist` (
                           default: False)
        :param trail_start: If full_trail is True, then the trail starts at
                            this path distance (default: 0).
        :param stringent_control: If True then control group will not
                                  contain cells that have mapping score higher
                                  than min_score
        :return: None
        """
        if from_clusters is not None:
            self._validate_clusters()
            if type(from_clusters) != list:
                raise TypeError("ERROR: from_cluster parameter value "
                                "should be a list")
            from_clusters = {str(x): None for x in from_clusters}
            valid_nodes = []
            cluster_dict = self.clusters
            for i in list(self.refNodes):
                if i in cluster_dict and cluster_dict[i] in from_clusters:
                    valid_nodes.append(i)
        else:
            valid_nodes = list(self.refNodes)
        valid_scores = self.get_mapping_score(target, min_score=min_score,
                                              all_nodes=False)
        test_nodes = {x: None for x in valid_nodes if x in valid_scores}
        if len(test_nodes) < 5:
            print('WARNING: Less than 5 test nodes found! Will not '
                  'set "de_group"')
            return None
        control_nodes = [x for x in self.get_k_path_neighbours(
            list(test_nodes.keys()), node_dist,
            full_trail=full_trail, trail_start=trail_start)]
        if stringent_control:
            control_nodes = {x: None for x in control_nodes
                             if x not in valid_scores}
        else:
            control_nodes = {x: None for x in control_nodes}
        for node in self.refNodes:
            if node in test_nodes:
                self.nodes[node]['de_group'] = 'Test'
            elif node in control_nodes:
                self.nodes[node]['de_group'] = 'Control'
            else:
                self.nodes[node]['de_group'] = 'Other'
        self.deTestCells = [x.rsplit('_', 1)[0] for x in test_nodes]
        self.deCtrlCells = [x.rsplit('_', 1)[0] for x in control_nodes]
        print("Test nodes: %d, Control nodes: %d" % (
            len(self.deTestCells), len(self.deCtrlCells)), flush=True)
        return None
