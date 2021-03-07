import os
import sys

sys.path.append((os.path.abspath(os.path.dirname(__file__)).split('src')[0] + 'src'))
from utils.ogb_utils import DglNodePropPredDataset
from utils.proj_settings import MAG_META_DICT

if __name__ == "__main__":
    dataset = DglNodePropPredDataset(name='ogbn-mag', meta_dict=MAG_META_DICT)

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    graph
