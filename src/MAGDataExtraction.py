import os
import sys

sys.path.append((os.path.abspath(os.path.dirname(__file__)).split('src')[0] + 'src'))
from utils.ogb_utils import DglNodePropPredDataset
from utils.settings import *
from utils.dgl_utils import *
from utils.util_funcs import *
import dgl
import torch as th
import numpy as np
from tqdm import trange


def mag_extraction(g, label, year_list, author_degree_threshold=0):
    '''
    Extract authors that publish at least one paper in each year
    '''
    # ! Find candidate authors by year
    # init to all authors, intersect with each year's author to get authors that publish in each and every year.
    all_authors = g.nodes('author')
    author_subset = set(all_authors.tolist())
    # subset authors with threshold
    if author_degree_threshold > 0:
        authors_above_thresholds = all_authors[g.out_degrees(all_authors, etype='writes') > author_degree_threshold]
        author_subset.intersection_update(set(authors_above_thresholds.tolist()))

    for year in year_list:
        paper_this_year = g.filter_nodes(lambda nodes: (nodes.data['year'] == year).squeeze(1), ntype='paper')
        author_this_year = set(neighbors(g, paper_this_year, 'writes', 'dst'))
        author_subset.intersection_update(author_this_year)

    # ! Subsetting nodes related to authors
    author_subset = list(author_subset)
    node_dict = {'author': list(author_subset)}
    node_dict['paper'] = neighbors(g, author_subset, 'writes')
    node_dict['institution'] = neighbors(g, author_subset, 'affiliated_with')
    node_dict['field_of_study'] = neighbors(g, node_dict['paper'], 'has_topic')

    g = dgl.node_subgraph(g, node_dict, store_ids=True)

    # ! Process features
    # Features generated from average paper emb
    p_feat = g.ndata['feat']['paper'].numpy()
    feat_dict = {t: np.zeros((len(g.nodes(t)), p_feat.shape[1])) for t in g.ntypes}
    feat_dict['paper'] = p_feat

    for i in trange(len(g.nodes('author')), desc='Author feature processing'):
        feat_dict['author'][i, :] = p_feat[neighbors(g, i, 'writes'), :].mean(axis=0)

    for i in trange(len(g.nodes('institution')), desc='Institution feature processing'):
        auth_neighbors = neighbors(g, i, 'affiliated_with', 'dst')  # I -> A, A -> P
        feat_dict['institution'][i, :] = p_feat[neighbors(g, auth_neighbors, 'writes'), :].mean(axis=0)

    for i in trange(len(g.nodes('field_of_study')), desc='Field feature processing'):
        feat_dict['field_of_study'][i, :] = p_feat[neighbors(g, i, 'has_topic', 'dst'), :].mean(axis=0)

    g.ndata['feat'] = {t: th.tensor(feat_dict[t], dtype=th.float32) for t in g.ntypes}
    print(f'Subset graph finished\n{dgl_graph_to_str(g)}')

    # ! Subset labels
    labels = np.ones(len(g.nodes('paper'))) * -1
    origin_node_id_list = g.ndata['_ID']['paper']
    for i in g.nodes('paper').tolist():
        labels[i] = label['paper'][origin_node_id_list[i]]
    assert sum(labels == -1) == 0
    save_pickle(labels.astype(int), LABEL_PATH)
    dgl.save_graphs(FULL_MAG_PATH, [g])
    print('MAG extraction finished')
    return g


def mag_subg_by_year(g, year_list):
    # ! Create subgraph list by year
    glist = []
    for year in year_list:
        paper_this_year = g.filter_nodes(lambda nodes: (nodes.data['year'] == year).squeeze(1), ntype='paper')
        field_this_year = neighbors(g, paper_this_year, 'has_topic')
        author_this_year = neighbors(g, paper_this_year, 'writes', 'dst')
        institution_this_year = neighbors(g, author_this_year, 'affiliated_with')
        assert author_this_year == list(g.nodes('author'))
        glist.append(dgl.node_subgraph(g, {'author': author_this_year, 'field_of_study': field_this_year,
                                           'institution': institution_this_year, 'paper': paper_this_year}))
    dgl.save_graphs(SUB_MAG_PATH, glist)
    print('Subgraphs by each year saved, the number of total nodes in each subgraph are as follows:')
    print([len(g_) for g_ in glist])


if __name__ == "__main__":
    dataset = DglNodePropPredDataset(name='ogbn-mag', meta_dict=MAG_META_DICT)

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    g, label = dataset[0]  # graph: dgl graph object, label: th tensor of shape (num_nodes, num_tasks)
    print(f'Load original MAG graph finished\n{dgl_graph_to_str(g)}')

    author_degree_threshold = 200

    year_list = list(range(2010, 2020))
    g = mag_extraction(g, label, year_list, author_degree_threshold)
    mag_subg_by_year(g, year_list)
    print(f'Subset MAG subgraph with threshold {author_degree_threshold} finished')
