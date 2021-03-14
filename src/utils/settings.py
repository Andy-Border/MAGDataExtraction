MAG_META_DICT = {
    # OGB original MAG MAG_META_DICT
    'add_inverse_edge': 'False',
    'additional edge files': 'edge_reltype',
    'additional node files': 'node_year',
    'binary': 'False',
    'download_name': 'mag',
    'eval metric': 'acc',
    'has_edge_attr': 'False',
    'has_node_attr': 'True',
    'is hetero': 'True',
    'num classes': '349',
    'num tasks': '1',
    'split': 'time',
    'task type': 'multiclass classification',
    'url': 'https://snap.stanford.edu/ogb/data/nodeproppred/mag.zip',
    'version': '2',

    # Modified MAG path info
    'data_root_path': 'data/',
    'raw_data_path': 'data/ogb_raw/mag.zip',
}

FULL_MAG_PATH = 'data/ogb_processed/full_graph.bin'
SUB_MAG_PATH = 'data/ogb_processed/sub_graphs.bin'
LABEL_PATH = 'data/ogb_processed/labels.pickle'
