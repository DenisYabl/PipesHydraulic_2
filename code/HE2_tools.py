from HE2_Pipe import HE2_WaterPipe
import HE2_Vertices as vrtxs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_random_net_v0(N=15, E=20, SRC=3, SNK=3, Q=20, P=200, D=0.5, H=50, L=1000, RGH=1e-4, SEGS=10,
                           randseed=None):
    '''
    :param N: total nodes count
    :param E: total edges count, cannot be less than N-1
    :param SRC: sources count
    :param SNK: sinks count
    :param Q: maximum boundary flow on source (not sink!)  node
    :param P: maximum boundary pressure on one of source/sink node
    :param D: maximum pipe segment inner diameter on graph edge
    :param H: maximum pipe segment slope
    :param L: maximum pipe segment length
    :param SEGS: maximum pipe segments count
    :return: DiGraph (not Multi)
    This method produce water pipelines network with one pressure constrained node, with constant temperature
    and with some sources/sinks. Network graph can contains cycles
    '''
    np.random.seed(randseed)
    E = max(E, N - 1)
    J = N - SRC - SNK
    juncs = {f'junc_{i}': vrtxs.HE2_ABC_GraphVertex() for i in range(J)}
    coin = np.random.randint(0, 2)
    pressure = np.random.randint(-P, P)
    if coin == 1:
        p_nodes = {'p_node_0': vrtxs.HE2_Source_Vertex('P', pressure, 'water', 20)}
    else:
        p_nodes = {'p_node_0': vrtxs.HE2_Boundary_Vertex('P', pressure)}

    src_q = np.random.uniform(0, Q, SRC)
    total_q = sum(src_q)
    sink_q = np.random.uniform(0, 1, SNK)
    sink_q = total_q * sink_q / sum(sink_q)
    np.testing.assert_almost_equal(total_q, sum(sink_q))

    SRC = SRC - coin
    SNK = SNK - (1 - coin)
    sources = {f'src_{i}': vrtxs.HE2_Source_Vertex('Q', src_q[i], 'water', 20) for i in range(SRC)}
    sinks = {f'sink_{i}': vrtxs.HE2_Boundary_Vertex('Q', -sink_q[i]) for i in range(SNK)}

    nodes = {**p_nodes, **sources, **sinks, **juncs}
    mapping = dict(zip(range(len(nodes)), nodes.keys()))
    UDRG = nx.generators.random_graphs.gnm_random_graph(N, E, directed=False, seed=randseed)
    while nx.algorithms.components.number_connected_components(nx.Graph(UDRG)) != 1:
        UDRG = nx.generators.random_graphs.gnm_random_graph(N, E, directed=False)
    edgelist = []
    for u, v in UDRG.edges:
        if np.random.randint(0, 2):
            edgelist += [(u, v)]
        else:
            edgelist += [(v, u)]
    DRG = nx.DiGraph(edgelist)
    G = nx.relabel.relabel_nodes(DRG, mapping)
    nx.set_node_attributes(G, name='obj', values=nodes)

    # for n in G.nodes():
    #     print(n)

    pipes = dict()
    for u, v in G.edges():
        segs = np.random.randint(SEGS) + 1
        Ls = np.random.uniform(1e-5, L, segs)
        Hs = np.random.uniform(-H, H, segs)
        Ds = np.random.uniform(1e-5, D, segs)
        Rs = np.random.uniform(0, RGH, segs)
        # print(u, v, Ls, Hs, Ds, Rs)
        pipe = HE2_WaterPipe(Ls, Hs, Ds, Rs)
        pipes[(u, v)] = pipe
    nx.set_edge_attributes(G, name='obj', values=pipes)

    return G, dict(p_nodes=p_nodes, juncs=juncs, sources=sources, sinks=sinks)

def generate_random_net_v1(N=15, E=20, SRC=3, SNK=3, P_CNT=1, Q=20, P=200, D=0.5, H=50, L=1000, RGH=1e-4, SEGS=1,
                           randseed=None):
    '''
    :param N: total nodes count
    :param E: total edges count, cannot be less than N-1
    :param SRC: sources count
    :param SNK: sinks count
    :param P_CNT: count of nodes with fixed pressure
    :param Q: maximum boundary flow on source (not sink!)  node
    :param P: maximum boundary pressure on one of source/sink node
    :param D: maximum pipe segment inner diameter on graph edge
    :param H: maximum pipe segment slope
    :param L: maximum pipe segment length
    :param SEGS: maximum pipe segments count
    :return: MultiDiGraph
    This method produce water pipelines network with some pressure constrained node, with constant temperature
    and with some sources/sinks. Result will be a non-tree, linked multigraph with objects attached to nodes and edges
    '''
    np.random.seed(randseed)
    E = max(E, N - 1)
    B = SRC + SNK
    J = N - B
    juncs = {f'junc_{i}': vrtxs.HE2_ABC_GraphVertex() for i in range(J)}

    kinds = ['P']*P_CNT + ['Q']*(B-P_CNT)
    srcs = [True] * SRC + [False] * SNK
    np.random.shuffle(kinds)
    total_q = np.random.uniform(Q * B)
    src_q = np.random.uniform(0, 1, SRC)
    src_q = total_q * src_q / sum(src_q)
    snk_q = np.random.uniform(0, 1, SNK)
    snk_q = total_q * snk_q / sum(snk_q)
    qs = list(src_q) + list(snk_q)

    p_nodes, sources, sinks = dict(), dict(), dict()
    for kind, src, q in zip(kinds, srcs, qs):
        if src and kind == 'P':
            node = vrtxs.HE2_Source_Vertex(kind, np.random.randint(-P, P), 'water', 20)
            name = f'p_node_{len(p_nodes)}'
            p_nodes[name] = node
        elif src and kind == 'Q':
            node = vrtxs.HE2_Source_Vertex(kind, q, 'water', 20)
            name = f'src_{len(sources)}'
            sources[name] = node
        elif not src and kind == 'P':
            node = vrtxs.HE2_Boundary_Vertex(kind, np.random.randint(-P, P))
            name = f'p_node_{len(p_nodes)}'
            p_nodes[name] = node
        elif not src and kind == 'Q':
            node = vrtxs.HE2_Boundary_Vertex(kind, q)
            name = f'snk_{len(sinks)}'
            sinks[name] = node

    nodes = {**p_nodes, **sources, **sinks, **juncs}
    assert len(nodes) == N
    mapping = dict(zip(range(N), nodes.keys()))
    RT = nx.generators.trees.random_tree(N, seed=randseed)
    edgelist = [tuple(np.random.choice([u, v], 2, replace=False)) for u, v in RT.edges]
    edgelist += [tuple(np.random.choice(range(N), 2, replace=False)) for i in range(E-(N-1))]

    MDRG = nx.MultiDiGraph(edgelist)
    G = nx.relabel.relabel_nodes(MDRG, mapping)
    nx.set_node_attributes(G, name='obj', values=nodes)

    pipes = dict()
    for u, v, k in G.edges:
        segs = np.random.randint(SEGS) + 1
        Ls = np.random.uniform(1e-5, L, segs)
        Hs = np.random.uniform(-H, H, segs)
        Ds = np.random.uniform(1e-5, D, segs)
        Rs = np.random.uniform(0, RGH, segs)
        # print(u, v, Ls, Hs, Ds, Rs)
        pipe = HE2_WaterPipe(Ls, Hs, Ds, Rs)
        pipes[(u, v, k)] = pipe
    nx.set_edge_attributes(G, name='obj', values=pipes)

    assert nx.algorithms.components.number_connected_components(nx.Graph(G)) == 1

    return G, dict(p_nodes=p_nodes, juncs=juncs, sources=sources, sinks=sinks)

def HE2_draw_node_labels(G, g_nodes, nodelist, keys, **kwargs):
    lbls = dict()
    for n in list(set(nodelist) & g_nodes):
        sss = [n]
        obj = G.nodes[n]['obj']
        for k in keys:
            try:
                if 'result' in obj.__dict__ and k in obj.result:
                    sss += [f'{obj.result[k]:.2f}']
                elif k in obj.__dict__:
                    sss += [f'{obj.__dict__[k]:.2f}']
            except:
                pass
        lbls.update({n: '\n'.join(sss)})
    nx.draw_networkx_labels(G, labels=lbls, **kwargs, font_size=7)

def draw_solution(G, shifts, p_nodes, sources, sinks, juncs):
    #TODO Не однообразно рисую узлы и дуги, просит рефакторинга
    #TODO Не однообразно формирую лейблы для узлов и для дуг, тоже просит рефакторинга
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    # pos = nx.drawing.layout.planar_layout(G)
    pos = nx.drawing.layout.planar_layout(G)
    g_nodes = set(G.nodes)
    # params = zip([p_nodes, sources, sinks, juncs], [50, 50, 50, 10], ['red', 'blue','blue','black'], [[], ['Q'], ['Q'], []])
    params = zip([sources, sinks], [50, 10], ['blue','black'], [['Q'], ['Q']])
    label_pos = {k:(pos[k][0] + shifts[k][0], pos[k][1] + shifts[k][1]) for k in pos} if shifts is not None else pos
    for nodelist, node_size, node_color, ks in params:
        nx.draw_networkx_nodes(G, nodelist=list(set(nodelist) & g_nodes), node_size=node_size, node_color=node_color, ax=ax, pos=pos)
        HE2_draw_node_labels(G, g_nodes, list(set(nodelist) & g_nodes), keys=['P_bar']+ks, ax=ax, pos=label_pos)


    # edge_labels = {(u,v): str(G[u][v]['obj'])+f"\n{G[u][v]['obj'].result['x']:.2f}" for u, v in G.edges()}
    edge_labels = {(u,v): f"\n{G[u][v]['obj'].result['x']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=7)

    # if type(G) == nx.MultiDiGraph:
    #     edge_labels = {(u,v): str(G[u][v][k]['obj'])+f"\n{G[u][v][k]['obj'].result['x']:.2f}" for u, v, k in G.edges}
    # else:
    #     edge_labels = {(u, v): str(G[u][v]['obj']) + f"\n{G[u][v]['obj'].result['x']:.2f}" for u, v in G.edges}

    if type(G) == nx.MultiDiGraph:
        edge_labels = {(u,v): f"{G[u][v][k]['obj'].result['x']:.2f}" for u, v, k in G.edges}
    else:
        edge_labels = {(u, v): f"{G[u][v]['obj'].result['x']:.2f}" for u, v in G.edges}

    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=9)

    nx.draw_networkx_edges(G, pos=pos, width=2, ax=ax, edge_color='black')
    plt.show()


def evaluate_1stCL_residual(graph):
    G = nx.MultiDiGraph(graph)
    Q_dict = {}
    X_sum_dict = dict(zip(G.nodes, [0]*len(G.nodes)))
    p_nodes = []
    for n in G.nodes:
        obj = G.nodes[n]['obj']
        if isinstance(obj, vrtxs.HE2_Boundary_Vertex) and obj.kind == 'P':
            p_nodes += [n]
            continue

        Q_dict[n] = 0
        if isinstance(obj, vrtxs.HE2_Boundary_Vertex) and obj.kind == 'Q':
            Q_dict[n] = obj.value if obj.is_source else -obj.value

    for u, v, k in G.edges:
        x = G[u][v][k]['obj'].result['x']
        X_sum_dict[u] -= x
        X_sum_dict[v] += x

    residual = 0
    for n in Q_dict:
        residual += abs(Q_dict[n] + X_sum_dict[n])

    Q_net_balance = sum(Q_dict.values())
    p_x_sum = 0
    for n in p_nodes:
        p_x_sum += X_sum_dict[n]
    residual += abs(p_x_sum - Q_net_balance)

    return residual

def evaluate_2ndCL_residual(graph):
    G = nx.MultiDiGraph(graph)
    residual = 0
    for (u, v, k) in G.edges:
        u_obj = G.nodes[u]['obj']
        v_obj = G.nodes[v]['obj']
        edge_obj = G[u][v][k]['obj']
        x = edge_obj.result['x']
        p_u = u_obj.result['P_bar']
        t_u = u_obj.result['T_C']
        p_v = v_obj.result['P_bar']
        t_v = v_obj.result['T_C']
        p, t = edge_obj.perform_calc_forward(p_u, t_u, x)
        residual += abs(p - p_v)
    return residual

def check_solution(G):
    res1 = evaluate_1stCL_residual(G)
    res2 = evaluate_2ndCL_residual(G)
    return res1, res2

def check_fluid_mixation(G, x_dict, cocktails, sources):
    S = len(sources)
    for n in G.nodes:
        x_in, x_out = 0, 0
        for e in G.in_edges(n):
            x_in += x_dict.get(e, 0)
        for e in G.out_edges(n):
            x_out += x_dict.get(e, 0)

        out_cktls = []
        if n in cocktails:
            out_cktls += [cocktails[n]]
        for e in G.out_edges(n):
            if e in cocktails:
                out_cktls += [cocktails[e]]

        if len(out_cktls) == 0:
            assert x_in == 0
            assert x_out == 0
            continue

        for c1, c2 in zip(out_cktls, out_cktls[1:]):
            if np.linalg.norm(c1-c2) > 1e-5:
                return False

        out_cktl = out_cktls[0]
        in_cktl = np.zeros(S)
        if n in sources:
            assert x_out >= x_in
            idx = sources.index(n)
            in_cktl[idx] = x_out - x_in

        for e in G.in_edges(n):
            if e in cocktails:
                x = x_dict.get(e, 0)
                in_cktl += x * cocktails[e]

        in_cktl /= sum(in_cktl)
        if np.linalg.norm(in_cktl - out_cktl) > 1e-5:
            return False

    return True


def build_dual_schema_from_solved(schema, p_nodes, sources, sinks, juncs):
    G = nx.MultiDiGraph(schema, data=True)
    nodes_P, nodes_Q = {}, {}

    X_sum_dict = dict(zip(G.nodes, [0]*len(G.nodes)))
    for u, v, k in G.edges:
        x = G[u][v][k]['obj'].result['x']
        X_sum_dict[u] -= x
        X_sum_dict[v] += x

    for n in p_nodes:
        nodes_P[n] = G.nodes[n]['obj'].P
        nodes_Q[n] = -X_sum_dict[n]

    for n in juncs:
        nodes_Q[n] = 0
    for n in sources:
        nodes_Q[n] = G.nodes[n]['obj'].Q
    for n in sinks:
        nodes_Q[n] = -G.nodes[n]['obj'].Q

    for n in {**sources, **sinks, **juncs}:
        obj = G.nodes[n]['obj']
        nodes_P[n] = obj.result['P_bar']

    newschema = nx.MultiDiGraph()
    p_n = np.random.choice(G.nodes)
    for n in G.nodes:
        kind = np.random.choice(['P', 'Q'], p=[0.2, 0.8])
        if n == p_n:
            kind = 'P'
        P = nodes_P[n]
        Q = nodes_Q[n]
        value = abs(Q) if kind=='Q' else P
        obj = None

        if (Q<0) or (Q==0 and kind == 'P'):
            obj = vrtxs.HE2_Boundary_Vertex(kind, value)
        elif Q==0 and kind == 'Q':
            obj = vrtxs.HE2_ABC_GraphVertex()
        elif Q>0:
            obj = vrtxs.HE2_Source_Vertex(kind, value, 'water', 20)

        newschema.add_node(n, obj=obj)

    for u, v, k in G.edges:
        obj = G[u][v][k]['obj']
        dxs, dys, diams, rghs = [], [], [], []
        for seg in obj.segments:
            dxs += [seg.dx_m]
            dys += [seg.uphill_m]
            diams += [seg.inner_diam_m]
            rghs += [seg.roughness_m]

        newobj = HE2_WaterPipe(dxs, dys, diams, rghs)
        newschema.add_edge(u, v, obj=newobj)

    return newschema

def generate_superpositioned_colored_flows_graph(N=10, E=13, SRC=3, SNK=3, randseed=None):
    return
    # Bullshit. Main idea doesnt work
    np.random.seed(randseed)
    RT = nx.generators.trees.random_tree(N, seed=randseed)
    edge_list = [tuple(np.random.choice([u, v], 2, replace=False)) for u, v in RT.edges]
    edge_set = set(edge_list)
    while len(edge_set) < E:
        u, v = tuple(np.random.choice(range(N), 2, replace=False))
        if ((v, u) in edge_set) or ((v, u) in edge_set):
            continue
        edge_set |= {(u, v)}
    edge_list = list(edge_set)

    DRG = nx.DiGraph()
    DRG.add_nodes_from(RT.nodes)
    DRG.add_edges_from(edge_list)
    base = DRG
    can_be_source, can_be_sink = set(), set()
    for n in base.nodes:
        if len(base.in_edges(n)) > 0:
            can_be_sink |= {n}
        if len(base.out_edges(n)) > 0:
            can_be_source |= {n}
    A = can_be_source - can_be_sink
    B = can_be_source - A
    srcs = list(A) + list(B)
    sources = {srcs[i] for i in range(SRC)}

    can_be_sink -= sources
    A = can_be_sink - can_be_source
    B = can_be_sink - A
    snks = list(A) + list(B)
    sinks = {snks[i] for i in range(SNK)}

    scaffold = nx.DiGraph(base)
    scaffold_nodes = ['SUPERSOURCE', 'SUPERSINK']
    scaffold.add_nodes_from(scaffold_nodes)
    scaffold_edges = []

    for n in sources:
        scaffold_edges += [('SUPERSOURCE', n)]
    src_edges = scaffold_edges[:]
    sink_edges = []
    for n in sinks:
        sink_edges += [(n, 'SUPERSINK')]
    scaffold_edges += sink_edges
    scaffold.add_edges_from(scaffold_edges)
    zero_capacity = {e:0 for e in src_edges}
    all_flows, Q = {}, {}
    for n in sources:
        nx.set_edge_attributes(scaffold, name='capacity', values=zero_capacity)
        scaffold['SUPERSOURCE'][n]['capacity'] = 1000
        capacities = {e:np.random.randint(20, 100) for e in base.edges}
        # capacities.update({e:np.random.randint(0, 1000) for e in sink_edges})
        nx.set_edge_attributes(scaffold, name='capacity', values=capacities)
        flow_value, flow_dict = nx.algorithms.flow.maximum_flow(scaffold, 'SUPERSOURCE', 'SUPERSINK')
        all_flows[n] = flow_dict
        Q[n] = flow_value

    rez = {}
    for n, flow_dict in all_flows.items():
        for u, v_x_dict in flow_dict.items():
            for v, x in v_x_dict.items():
                xs = rez.get((u, v), [])
                xs += [x]
                rez[(u, v)] = xs

    result, x_dict = {}, {}
    for (u, v), xs in rez.items():
        if len({u, v} & {'SUPERSOURCE', 'SUPERSINK'}) > 0:
            continue
        assert len(xs) == len(sources)
        arr = np.array(xs)
        sss = sum(arr)
        x_dict[(u, v)] = sss
        if sss == 0:
            continue
        arr = arr / sss
        result[(u, v)] = arr
        print(u, v, arr)



    return result, sources, base, x_dict


def split_input_df_to_pipes_and_boundaries(df):
    bnd_cols1 = ['kind', 'Q', 'is_source', 'P']
    bnd_cols2 = ['node_id', 'node_name']
    start_cols = [col + '_start' for col in bnd_cols2] + ['start_' + col for col in bnd_cols1]
    end_cols = [col + '_end' for col in bnd_cols2] + ['end_' + col for col in bnd_cols1]
    df_bnd_start = df[start_cols]
    df_bnd_start.columns = bnd_cols2 + bnd_cols1
    df_bnd_end = df[end_cols]
    df_bnd_end.columns = bnd_cols2 + bnd_cols1
    df_bnds = pd.concat([df_bnd_start, df_bnd_end])
    df_bnds = df_bnds[~df_bnds.kind.isna()]
    to_drop = start_cols + end_cols
    df_pipes = df.drop(columns=to_drop)
    return df_pipes, df_bnds

def split_result_df_to_pipes_and_nodes(df):
    bnds_cols = ['kind', 'Q', 'is_source', 'P']
    node_cols = ['node_id', 'node_name', 'node_type', 'x', 'y']
    start_cols = [col + '_start' for col in node_cols] + ['result_start_P']
    end_cols = [col + '_end' for col in node_cols] + ['result_end_P']
    df_node_start = df[start_cols]
    df_node_start.columns = node_cols + ['result_P']
    df_node_end = df[end_cols]
    df_node_end.columns = node_cols + ['result_P']
    df_nodes = pd.concat([df_node_start, df_node_end]).drop_duplicates()
    node_id_set = set(df.node_id_start.values) | set(df.node_id_end.values)
    print(len(node_id_set))
    from collections import Counter
    cntr = Counter(df_nodes.node_id)
    print(cntr.most_common(10))



    to_drop = start_cols + end_cols
    df_pipes = df.drop(columns=to_drop)
    # assert len(df_nodes) == len(df_nodes.node_id.unique())
    return df_pipes, df_nodes
