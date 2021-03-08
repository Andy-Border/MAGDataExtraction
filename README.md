# MAGDataExtraction

Create and processing of MAG dataset

# Requirements
- python >= 3.8
- dgl == 0.6.0
- ogb == 1.2.6

# Usage
- Download dataset https://snap.stanford.edu/ogb/data/nodeproppred/mag.zip
- Copy mag.zip to data/ogb_raw folder
- python src/MAGDataExtraction.py

Settings are stored in src/utils/settings.py.

Default output location is data/ogb_processed