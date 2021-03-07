import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from dgl.data.utils import load_graphs, save_graphs, Subset
import dgl
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_dgl import read_graph_dgl, read_heterograph_dgl
from ogb.io.read_graph_raw import read_node_label_hetero, read_nodesplitidx_split_hetero


class DglNodePropPredDataset(object):
    def __init__(self, name,meta_dict=None):

        self.name = name  ## original name, e.g., ogbn-proteins

        # ! MODIFIED
        self.original_root = self.data_root_path = meta_dict['data_root_path']
        self.raw_data_path = meta_dict['raw_data_path']
        self.dir_name = '_'.join(name.split('-'))

        # check if previously-downloaded folder exists. If so, use that one.
        if osp.exists(osp.join(meta_dict['data_root_path'], self.dir_name + '_dgl')):
            self.dir_name = self.dir_name + '_dgl'

        self.original_root = meta_dict['data_root_path']
        self.root = osp.join(meta_dict['data_root_path'], self.dir_name)
        self.meta_info = meta_dict
        # ! MODIFIED

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']  ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.num_classes = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(DglNodePropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if osp.exists(pre_processed_file_path):
            self.graph, label_dict = load_graphs(pre_processed_file_path)
            if self.is_hetero:
                self.labels = label_dict
            else:
                self.labels = label_dict['labels']
        else:
            ### check if the downloaded file exists
            if self.binary:
                # npz format
                has_necessary_file_simple = osp.exists(osp.join(self.root, 'raw', 'data.npz')) and (not self.is_hetero)
                has_necessary_file_hetero = osp.exists(osp.join(self.root, 'raw', 'edge_index_dict.npz')) and self.is_hetero
            else:
                # csv file
                has_necessary_file_simple = osp.exists(osp.join(self.root, 'raw', 'edge.csv.gz')) and (not self.is_hetero)
                has_necessary_file_hetero = osp.exists(osp.join(self.root, 'raw', 'triplet-type-list.csv.gz')) and self.is_hetero

            has_necessary_file = has_necessary_file_simple or has_necessary_file_hetero

            if not has_necessary_file:
                url = self.meta_info['url']
                path = self.raw_data_path
                extract_zip(path, self.data_root_path)
                os.unlink(path)
                # delete folder if there exists
                try:
                    shutil.rmtree(self.root)
                except:
                    pass
                shutil.move(osp.join(self.original_root, self.download_name), self.root)

            raw_dir = osp.join(self.root, 'raw')

            ### pre-process and save
            add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

            if self.meta_info['additional node files'] == 'None':
                additional_node_files = []
            else:
                additional_node_files = self.meta_info['additional node files'].split(',')

            if self.meta_info['additional edge files'] == 'None':
                additional_edge_files = []
            else:
                additional_edge_files = self.meta_info['additional edge files'].split(',')

            if self.is_hetero:
                graph = read_heterograph_dgl(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]

                if self.binary:
                    tmp = np.load(osp.join(raw_dir, 'node-label.npz'))
                    label_dict = {}
                    for key in list(tmp.keys()):
                        label_dict[key] = tmp[key]
                    del tmp
                else:
                    label_dict = read_node_label_hetero(raw_dir)

                # convert into torch tensor
                if 'classification' in self.task_type:
                    for nodetype in label_dict.keys():
                        # detect if there is any nan
                        node_label = label_dict[nodetype]
                        if np.isnan(node_label).any():
                            label_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)
                        else:
                            label_dict[nodetype] = torch.from_numpy(node_label).to(torch.long)
                else:
                    for nodetype in label_dict.keys():
                        node_label = label_dict[nodetype]
                        label_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)

            else:
                graph = read_graph_dgl(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)[0]

                ### adding prediction target
                if self.binary:
                    node_label = np.load(osp.join(raw_dir, 'node-label.npz'))['node_label']
                else:
                    node_label = pd.read_csv(osp.join(raw_dir, 'node-label.csv.gz'), compression='gzip', header=None).values

                if 'classification' in self.task_type:
                    # detect if there is any nan
                    if np.isnan(node_label).any():
                        node_label = torch.from_numpy(node_label).to(torch.float32)
                    else:
                        node_label = torch.from_numpy(node_label).to(torch.long)
                else:
                    node_label = torch.from_numpy(node_label).to(torch.float32)

                label_dict = {'labels': node_label}

            print('Saving...')
            save_graphs(pre_processed_file_path, graph, label_dict)

            self.graph, label_dict = load_graphs(pre_processed_file_path)

            if self.is_hetero:
                self.labels = label_dict
            else:
                self.labels = label_dict['labels']

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = torch.from_numpy(train_idx_dict[nodetype]).to(torch.long)
                valid_idx_dict[nodetype] = torch.from_numpy(valid_idx_dict[nodetype]).to(torch.long)
                test_idx_dict[nodetype] = torch.from_numpy(test_idx_dict[nodetype]).to(torch.long)

                return {'train': train_idx_dict, 'valid': valid_idx_dict, 'test': test_idx_dict}

        else:
            train_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            valid_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]).to(torch.long)

            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph[idx], self.labels

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


if __name__ == '__main__':
    dgl_dataset = DglNodePropPredDataset(name='ogbn-proteins')
    print(dgl_dataset.num_classes)
    split_index = dgl_dataset.get_idx_split()
    print(dgl_dataset[0])
    print(split_index)
