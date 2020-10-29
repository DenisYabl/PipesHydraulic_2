import numpy as np
import networkx as nx
import scipy.optimize as scop
from HE2_SpecialEdges import HE2_MockEdge
import HE2_Vertices as vrtxs
import HE2_ABC as abc

Root = 'Root'


class HE2_Solver():
    def __init__(self, schema):
        # TODO have to implement take MultiDiGraph and convert it to equal DiGraph with some added mock edges
        self.schema = schema
        self.graph = None
        self.op_result = None
        self.C = None
        self.span_tree = None
        self.chordes = None
        self.edge_list = []
        self.node_list = []
        self.A_tree = None
        self.A_chordes = None
        self.A_inv = None
        self.Q_static = None
        self.edges_x = None
        self.pt_on_tree = None
        self.tree_travers = None
        self.mock_nodes = []
        self.mock_edges = []

    def solve(self):
        self.graph = self.add_root_to_graph()
        self.span_tree, self.chordes = self.split_graph(self.graph)
        self.edge_list = self.span_tree + self.chordes
        self.node_list = list(self.graph.nodes())
        assert self.node_list[-1] == Root
        self.C = len(self.chordes)
        # span_tree and chordes are DiGraphs, which edges match with self.graph edges
        self.tree_travers = self.build_tree_travers(self.span_tree, Root)
        self.A_tree, self.A_chordes = self.build_incidence_matrices()
        self.A_inv = np.linalg.inv(self.A_tree)
        self.Q_static = self.build_static_Q_vec(self.graph)

        def target(x_chordes):
            Q = self.Q_static
            if self.C:
                Q_dynamic = np.matmul(self.A_chordes, x_chordes)
                Q = Q - Q_dynamic
            x_tree = np.matmul(self.A_inv, Q)
            self.edges_x = dict(zip(self.span_tree, x_tree))
            if self.C:
                self.edges_x.update(dict(zip(self.chordes, x_chordes)))

            self.perform_self_test_for_1stCL()

            self.pt_on_tree = self.evalute_pressures_by_tree()
            pt_residual_vec = self.evalute_chordes_pressure_residual()
            rez = np.linalg.norm(pt_residual_vec)
            return rez

        if self.C:
            x0 = np.zeros(self.C)
            # Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact не хочут, Jacobian is required
            # SLSQP           7/50 6.34s  [15, 18, 23, 26, 34, 35, 43]
            # BFGS            7/50 11.8s  [5, 15, 18, 23, 34, 36, 46]
            # L-BFGS-B,       13/50
            # Powell          14/50
            # CG              15/50
            # trust-constr    15/50
            # Nelder-Mead     25/50
            # TNC             bullshit
            # COBYLA          bullshit

            self.op_result = scop.minimize(target, x0, method='SLSQP')
            # print(self.op_result)
            _x = self.op_result.x
            target(_x)
            # TODO Вот здесь надо забирать давления по ключу op_result.x из промежуточных результатов, когда они будут
        else:
            target(None)

        self.attach_results_to_schema()
        return

    def perform_self_test_for_1stCL(self):
        resd_1stCL = self.evaluate_1stCL_residual()
        x_sum = sum(map(abs, self.edges_x.values()))
        if abs(resd_1stCL) > 1e-7 * x_sum:
            assert False

    def build_tree_travers(self, di_tree, root):
        di_edges = set(di_tree)
        undirected_tree = nx.Graph(di_tree)
        tree_travers = []
        for u, v in nx.algorithms.traversal.edgebfs.edge_bfs(undirected_tree, root):
            if (u, v) in di_edges:
                tree_travers += [(u, v, 1)]
            else:
                assert (v, u) in di_edges
                tree_travers += [(v, u, -1)]
        return tree_travers

    def split_graph(self, graph):
        G = nx.Graph(graph)

        t_ = nx.minimum_spanning_tree(G)
        te_ = set(t_.edges())

        tl, cl = [], []
        for u, v in self.graph.edges():
            if (u, v) in te_ or (v, u) in te_:
                tl += [(u, v)]
            else:
                cl += [(u, v)]

        return tl, cl

    def add_root_to_graph(self):
        self.mock_nodes = [Root]
        self.mock_edges = []
        G = nx.DiGraph(self.schema)
        G.add_node(Root, obj=None)
        for n in G.nodes:
            obj = G.nodes[n]['obj']
            if isinstance(obj, vrtxs.HE2_Boundary_Vertex) and obj.kind == 'P':
                new_obj = vrtxs.HE2_ABC_GraphVertex()
                G.nodes[n]['obj'] = new_obj
                G.add_edge(Root, n, obj=HE2_MockEdge(obj.value))
                self.mock_edges += [(Root, n)]
        return G

    def build_static_Q_vec(self, G):
        q_vec = np.zeros(len(self.node_list)-1)
        for i, node in enumerate(G.nodes):
            obj = G.nodes[node]['obj']
            if isinstance(obj, vrtxs.HE2_Boundary_Vertex):
                assert obj.kind == 'Q'
                q_vec[i] = obj.value if obj.is_source else -obj.value
        return q_vec

    def build_incidence_matrices(self):
        nodelist = self.node_list
        assert nodelist[-1] == Root
        tree_edgelist = self.span_tree
        chordes_edgelist = self.chordes

        A_full = nx.incidence_matrix(self.span_tree, nodelist=nodelist, edgelist=tree_edgelist, oriented=True)
        A_full = -1 * A_full.toarray()
        A_truncated = A_full[:-1]

        A_chordes_full = nx.incidence_matrix(self.chordes, nodelist=nodelist, edgelist=chordes_edgelist, oriented=True)
        A_chordes_full = -1 * A_chordes_full.toarray()
        A_chordes_truncated = A_chordes_full[:-1]
        return A_truncated, A_chordes_truncated

    def evalute_pressures_by_tree(self):
        pt = dict()
        pt[Root] = (0, 20)  # TODO: get initial T from some source

        for u, v, direction in self.tree_travers:
            obj = self.graph[u][v]['obj']
            if not isinstance(obj, abc.HE2_ABC_GraphEdge):
                assert False
            known, unknown = u, v
            if v in pt:
                known, unknown = v, u

            assert not (unknown in pt)
            p_kn, t_kn = pt[known]
            x = self.edges_x[(u, v)]
            if u == known:
                p_unk, t_unk = obj.perform_calc_forward(p_kn, t_kn, x)
            else:
                p_unk, t_unk = obj.perform_calc_backward(p_kn, t_kn, x)
            pt[unknown] = (p_unk, t_unk)
        return pt

    def evalute_chordes_pressure_residual(self):
        if self.C == 0:
            return 0
        pt_v1, pt_v2 = [], []
        for (u, v) in self.chordes:
            x = self.edges_x[(u, v)]
            obj = self.graph[u][v]['obj']
            if not isinstance(obj, abc.HE2_ABC_GraphEdge):
                assert False
            p_u, t_u = self.pt_on_tree[u]
            p_v, t_v = obj.perform_calc_forward(p_u, t_u, x)
            pt_v1 += [(p_v, t_v)]
            pt_v2 += [self.pt_on_tree[v]]
        pt_v1_vec = np.array(pt_v1)
        pt_v2_vec = np.array(pt_v2)
        pt_residual_vec = pt_v1_vec - pt_v2_vec
        return pt_residual_vec

    def attach_results_to_schema(self):
        for u, pt in self.pt_on_tree.items():
            if u in self.mock_nodes:
                continue
            obj = self.schema.nodes[u]['obj']
            obj.result = dict(P_bar=pt[0], T_C=pt[1])
        for u, v in self.schema.edges:
            obj = self.schema[u][v]['obj']
            x = self.edges_x[(u, v)]
            obj.result = dict(x=x)

    def evaluate_1stCL_residual(self):
        residual = 0
        G = self.graph
        nodes = set(G.nodes())
        nodes -= {Root}
        Q_net_balance = 0
        for n in list(nodes) + [Root]:
            if n != Root:
                Q = 0
                obj = G.nodes[n]['obj']
                if isinstance(obj, vrtxs.HE2_Boundary_Vertex) and obj.kind == 'P':
                    continue

                if isinstance(obj, vrtxs.HE2_Boundary_Vertex) and obj.kind == 'Q':
                    Q = obj.value if obj.is_source else -obj.value

                Q_net_balance += Q
            else:
                Q = -Q_net_balance

            X_sum = 0
            for u, v in G.in_edges(n):
                X_sum -= self.edges_x[(u, v)]
            for u, v in G.out_edges(n):
                X_sum += self.edges_x[(u, v)]
            residual += abs(Q - X_sum)

        return residual

    def evaluate_2ndCL_residual(self):
        residual = 0
        G = self.graph
        for (u, v) in G.edges():
            edge_obj = G[u][v]['obj']
            x = self.edges_x[(u, v)]
            p_u, t_u = self.pt_on_tree[u]
            p_v, t_v = self.pt_on_tree[v]
            p, t = edge_obj.perform_calc_forward(p_u, t_u, x)
            residual += abs(p - p_v)
        return residual
