def dgl_graph_to_str(g):
    node_attr_info = {t: g.node_attr_schemes(t) for t in g.ntypes}
    print(f'Graph summary: {g}\n'
          f'Node attributes:{node_attr_info}')


def neighbors(g, nodes, etype, derive_from='src', unique=True):
    if derive_from == 'src':
        neighbor_tensor = g.out_edges(nodes, etype=etype)[1]
    elif derive_from == 'dst':
        neighbor_tensor = g.in_edges(nodes, etype=etype)[0]
    else:
        ValueError('The derive_from attr must be "src" or "dest"')
    if unique:
        return neighbor_tensor.unique().tolist()
    else:
        return neighbor_tensor.tolist()
