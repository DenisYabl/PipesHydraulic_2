from DFOperations import HE2_DateframeWrapper as model
import scipy.optimize as scop
import numpy as np
import pandas as pd
import os
from Solver.HE2_Solver import HE2_Solver
from Tools.HE2_schema_maker import make_oilpipe_schema_from_OT_dataset, make_calc_df
from Tools.HE2_tools import check_solution
import logging
import GraphNodes.HE2_Vertices as vrtx
import matplotlib.pyplot as plt
import random
from Tests.Optimization_test import cut_single_well_subgraph
import itertools

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


class HE2_OilGatheringNetwork_Model():
    def __init__(self, folder):
        # tree = os.walk(folder)
        # for fld, subfolders, files in tree:
        #     print(fld, subfolders, files)
        self.folder = folder
        self.solvers = dict()
        self.calc_dfs = dict()
        self.original_dfs = dict()
        self.graphs = dict()
        self.pad_well_list = None
        self.pad_wells_dict = None
        self.fact = dict()
        self.outlayers = dict()
        self.result = dict()
        self.last_well_result = dict()
        self.well_result_before = dict()
        self.well_result_after = dict()
        self.initial_x = dict()
        self.total_target_cnt, self.not_solved = 0, 0
        self.bad_wells = []
        self.N = 32

    def gimme_original_df(self, i):
        if i in self.original_dfs:
            return self.original_dfs[i]
        folder = self.folder
        filename = f'{folder}data/oilgathering_fit/DNS2_with_wells_{i}.csv'
        df = pd.read_csv(filename)
        self.original_dfs[i] = df
        return df

    def gimme_calc_df(self, i):
        if i in self.calc_dfs:
            return self.calc_dfs[i]
        folder = self.folder
        calc_df_filename = f'{folder}data/oilgathering_fit/calc_df_{i}.csv'
        try:
            calc_df = pd.read_csv(calc_df_filename)
            self.calc_dfs[i] = calc_df
            return calc_df
        except:
            pass
        original_df = self.gimme_original_df(i)
        calc_df = make_calc_df(original_df, self.folder + 'CommonData/')
        calc_df.to_csv(calc_df_filename)
        self.calc_dfs[i] = calc_df
        return calc_df

    def gimme_graph(self, i):
        if i in self.graphs:
            return self.graphs[i]
        df = self.gimme_original_df(i)
        calc_df = self.gimme_calc_df(i)
        # G, _ = make_oilpipe_schema_from_OT_dataset(df, self.folder + 'CommonData/', calc_df, ignore_Watercut=True)
        G, _ = make_oilpipe_schema_from_OT_dataset(df, self.folder + 'CommonData/', calc_df, ignore_Watercut=False)
        self.graphs[i] = G
        return self.graphs[i]

    def gimme_solver(self, i):
        if i in self.solvers:
            return self.solvers[i]

        G = self.gimme_graph(i)
        solver = HE2_Solver(G)
        self.solvers[i] = solver
        return solver

    def grab_results_to_one_dataframe(self):
        p_rez = dict()
        q_rez = dict()
        for i in range(self.N):
            G = self.gimme_graph(i)
            for n in G.nodes:
                obj = G.nodes[n]['obj']
                grab_q = isinstance(obj, vrtx.HE2_Boundary_Vertex)
                grab_q |= isinstance(obj, vrtx.HE2_Source_Vertex)
                have_to_grab = grab_q
                have_to_grab |= 'pump' in n
                have_to_grab |= 'wellhead' in n
                if not have_to_grab:
                    continue
                res = obj.result
                if grab_q:
                    q_lst = q_rez.get(n, np.zeros(self.N))
                    q_lst[i] = res['Q']
                    q_rez[n] = q_lst

                p_lst = p_rez.get(n, np.zeros(self.N))
                p_lst[i] = res['P_bar']
                p_rez[n] = p_lst

        nodes = set(q_rez.keys()) | set(p_rez.keys())

        df_P_res = pd.DataFrame()
        df_Q_res = pd.DataFrame()
        for n in nodes:
            if n in q_rez:
                df_Q_res[n] = q_rez[n]
            df_P_res[n] = p_rez[n]
        # print(df_P_res.head())
        # print(df_Q_res.head())

        # print()
        # print(n)
        # if n in q_rez:
        #     print(np.round(np.array(q_rez[n]), 3))
        # print(np.round(np.array(p_rez[n]), 3))


    # def grab_fact_to_one_dataframe(self):
    #     p_fact = dict()
    #     q_fact = dict()
    #     for i in range(self.N):
    #         G = self.gimme_graph(i)
    #         calc_df = self.gimme_calc_df(i)
    #         for n in G.nodes:
    #             obj = G.nodes[n]['obj']
    #             grab_q = isinstance(obj, vrtx.HE2_Boundary_Vertex)
    #             grab_q |= isinstance(obj, vrtx.HE2_Source_Vertex)
    #             if grab_q:
    #                 q_lst = q_fact.get(n, np.zeros(self.N))
    #                 q_lst[i] = calc_df.loc[calc_df.node_id_start==n,  'debit']
    #                 q_fact[n] = q_lst
    #             if 'pump' in n and 'intake' in n:
    #                 p_lst = p_fact.get(n, np.zeros(self.N))
    #                 p_lst[i] = calc_df.loc[calc_df.node_id_start==n,  'debit']
    #                 p_fact[n] = p_lst
    #             if 'pump' in n and 'outlet' in n:
    #                 p_lst = p_fact.get(n, np.zeros(self.N))
    #                 p_lst[i] = calc_df.loc[calc_df.node_id_start==n,  'debit']
    #                 p_fact[n] = p_lst
    #             if 'wellhead' in n:
    #                 p_lst = p_fact.get(n, np.zeros(self.N))
    #                 p_lst[i] = calc_df.loc[calc_df.node_id_start==n,  'debit']
    #                 p_fact[n] = p_lst
    #
    #     nodes = set(q_fact.keys()) | set(p_fact.keys())
    #
    #     df_P_fact = pd.DataFrame()
    #     df_Q_fact = pd.DataFrame()
    #     for n in nodes:
    #         if n in q_fact:
    #             df_Q_fact[n] = q_fact[n]
    #         df_P_fact[n] = p_fact[n]

    def gimme_wells(self):
        if self.pad_wells_dict:
            return self.pad_wells_dict
        dfs = []
        for i in range(self.N):
            df = self.gimme_original_df(i)
            df = df[['juncType', 'padNum', 'wellNum']]
            df = df[df.juncType == 'oilwell']
            dfs += [df]
        df = pd.concat(dfs).drop_duplicates()[['padNum', 'wellNum']]
        raw_pw_list = list(df.to_records(index=False))
        self.pad_well_list = []
        for pad, well in raw_pw_list:
            self.pad_well_list += [(int(pad), int(well))]
        pad_wells_dict = dict()
        for (pad, well) in self.pad_well_list:
            wlist = pad_wells_dict.get(pad, [])
            wlist += [int(well)]
            pad_wells_dict[int(pad)] = wlist
        self.pad_wells_dict = pad_wells_dict
        return self.pad_wells_dict

    def grab_fact(self):
        pad_wells_dict = self.gimme_wells()
        p_zab = dict()
        p_intake = dict()
        p_head = dict()
        q_well = dict()
        freq = dict()
        # for key in pad_well_list:
        #     p_zab[key] = np.zeros(self.N)
        #     p_intake[key] = np.zeros(self.N)
        #     p_head[key] = np.zeros(self.N)
        #     q_well[key] = np.zeros(self.N)

        cols = ['juncType', 'padNum', 'wellNum', 'zaboy_pressure','input_pressure', 'buffer_pressure', 'debit', 'frequency']
        dfs = []
        for i in range(self.N):
            df = self.gimme_original_df(i)
            df = df[cols]
            df['N'] = i
            dfs += [df]
        fact_df = pd.concat(dfs)
        for pad in pad_wells_dict:
            pad_df = fact_df[fact_df.padNum == pad]
            wells = pad_wells_dict[pad]
            for well in wells:
                df = pad_df[pad_df.wellNum == well]
                df = df.sort_values(by=['N'])
                p_zab[(pad, well)] = df.zaboy_pressure.values
                p_intake[(pad, well)] = df.input_pressure.values
                p_head[(pad, well)] = df.buffer_pressure.values
                q_well[(pad, well)] = df.debit.values
                freq[(pad, well)] = df.frequency.values

        return dict(p_zab=p_zab, p_intake=p_intake, p_head=p_head, q_well=q_well, freq=freq)

    def grab_results(self):
        pad_wells_dict = self.gimme_wells()
        p_zab = dict()
        p_intake = dict()
        p_head = dict()
        q_well = dict()
        for pad in pad_wells_dict:
            wells = pad_wells_dict[pad]
            for well in wells:
                key = (pad, well)
                zab_name = f"PAD_{pad}_WELL_{well}_zaboi"
                pump_name = f"PAD_{pad}_WELL_{well}_pump_intake"
                head_name = f"PAD_{pad}_WELL_{well}_wellhead"
                p_zab[key] = np.zeros(self.N)
                p_intake[key] = np.zeros(self.N)
                p_head[key] = np.zeros(self.N)
                q_well[key] = np.zeros(self.N)

                for i in range(self.N):
                    solver = self.gimme_solver(i)
                    G = solver.graph
                    p_zab[key][i] = G.nodes[zab_name]['obj'].result['P_bar']
                    p_intake[key][i] = G.nodes[pump_name]['obj'].result['P_bar']
                    p_head[key][i] = G.nodes[head_name]['obj'].result['P_bar']
                    obj = G[zab_name][pump_name]['obj']
                    q_well[key][i] = 86400 * obj.result['x'] / obj.result['liquid_density']
        return dict(p_zab=p_zab, p_intake=p_intake, p_head=p_head, q_well=q_well)

    def solve_em_all(self):
        for i in range(self.N):
            solver = self.gimme_solver(i)
            solver.solve(threshold=0.25)
            print(i, solver.op_result.success, solver.op_result.fun)

    def plot_fact_and_results(self, keys_to_plot=('head', 'intake', 'bottom', 'debit'), wells=(), pads=()):
        fig = plt.figure(constrained_layout=True, figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(keys_to_plot)
        ax.set_xlabel('fact')
        ax.set_ylabel('result')
        colors = plt.get_cmap('tab20c').colors
        colors = list(colors) * 3
        random.shuffle(colors)
        for i, (pad, well) in enumerate(self.pad_well_list):
            alpha = 0.2
            if well in wells or pad in pads:
                alpha = 0.8
            if 'head' in keys_to_plot:
                ax.plot(self.fact['p_head'][(pad, well)], self.result['p_head'][(pad, well)], color=colors[i], linewidth=1, alpha=alpha)
            if 'intake' in keys_to_plot:
                ax.plot(self.fact['p_intake'][(pad, well)], self.result['p_intake'][(pad, well)], color=colors[i], linewidth=1, alpha=alpha)
            if 'bottom' in keys_to_plot:
                ax.plot(self.fact['p_zab'][(pad, well)], self.result['p_zab'][(pad, well)], color=colors[i], linewidth=1, alpha=alpha)
            if 'debit' in keys_to_plot:
                ax.plot(self.fact['q_well'][(pad, well)], self.result['q_well'][(pad, well)], color=colors[i], linewidth=1, alpha=alpha)

        plt.show()

    def calc_well_score(self, pad, well):
        debit_scale = 150
        head_scale = 20
        intake_scale = 50
        bottom_scale = 80

        score = 0
        score += abs(self.fact['p_head'][(pad, well)] - self.result['p_head'][(pad, well)]) / head_scale
        score += abs(self.fact['p_intake'][(pad, well)] - self.result['p_intake'][(pad, well)]) / intake_scale
        score += abs(self.fact['p_zab'][(pad, well)] - self.result['p_zab'][(pad, well)]) / bottom_scale
        score += abs(self.fact['q_well'][(pad, well)] - self.result['q_well'][(pad, well)])  / debit_scale
        return score


    def prefit_all(self):
        rez = []
        self.gimme_wells()
        pad_well_list = self.pad_well_list
        for it, (pad, well) in enumerate(pad_well_list):
            if not (well, pad) in self.bad_wells:
                continue
            # if not (well, pad) == (567, 39):
            #     continue
            G0 = self.gimme_graph(0)
            nodes = [f'PAD_{pad}_WELL_{well}']
            nodes += [f'PAD_{pad}_WELL_{well}_zaboi']
            nodes += [f'PAD_{pad}_WELL_{well}_pump_intake']
            nodes += [f'PAD_{pad}_WELL_{well}_pump_outlet']
            nodes += [f'PAD_{pad}_WELL_{well}_wellhead']
            well_G, _ = cut_single_well_subgraph(G0, pad, well, nodes)
            solver = HE2_Solver(well_G)
            pump_obj = well_G[nodes[2]][nodes[3]]['obj']
            plast_obj = well_G[nodes[0]][nodes[1]]['obj']
            bounds = ((0.1, 3 * plast_obj.Productivity), (0, 1.2))
            path = []
            def target_fit(x):
                nonlocal pump_obj, plast_obj, solver, well_G, pad, well, nodes, bounds, path
                path += [x]
                productivity = x[0]
                plast_obj.Productivity = productivity
                pump_keff = x[1]
                pump_obj.change_stages_ratio(pump_keff)
                score = self.score_well(solver, well_G, pad, well, nodes)
                return score

            x0 = np.array([plast_obj.Productivity, 1])
            target_fit(x0)
            well_res_before = self.last_well_result.copy()

            xs = np.linspace(start=bounds[0][0], stop=bounds[0][1], num=15)
            ys = np.linspace(start=bounds[1][0], stop=bounds[1][1], num=15)
            zs = np.zeros((len(xs), len(ys)))

            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    zs[i, j] = target_fit(np.array([x, y]))

            i, j = np.unravel_index(np.argmin(zs, axis=None), zs.shape)
            x0 = np.array([xs[i], ys[j]])
            op_result = scop.minimize(target_fit, x0)
            best_x = op_result.x
            target_fit(best_x)
            well_res_after = self.last_well_result.copy()

            zs[zs==100500] = -1
            self.plot_single_well_chart(pad, well, well_res_before, well_res_after, well_G, nodes, op_result, xs, ys, zs, path)
            rez += [(well, pad, best_x, op_result.fun)]
            break
        return rez



    def score_well(self, solver, well_G, pad, well, nodes):
        # debit_scale, head_scale, intake_scale, bottom_scale = 150, 20, 50, 80
        debit_scale, head_scale, intake_scale, bottom_scale = 5, 20, 30, 100

        rez_p_i = np.ones(self.N) * 100500
        rez_p_b = np.ones(self.N) * 100500
        rez_q = np.ones(self.N) * 100500

        q0 = self.fact['q_well'][(pad, well)][0]
        known_Q = {nodes[0]: 900 * q0 / 86400}
        solver.set_known_Q(known_Q)
        fact_phs = self.fact['p_head'][(pad, well)]
        pump_obj = well_G[nodes[2]][nodes[3]]['obj']
        freqs = self.fact['freq'][(pad, well)]
        ol_freqs = self.outlayers['freq'][(pad, well)]
        freq = 50
        for i in range(self.N):
            if not ol_freqs[i]:
                freq = freqs[i]
            pump_obj.changeFrequency(freq) # Set last non-outlayer frequency

            well_G.nodes[nodes[-1]]['obj'].value = fact_phs[i]
            if (pad, well, i) in self.initial_x:
                solver.initial_edges_x = self.initial_x[(pad, well, i)]

            solver.solve(mix_fluids=False, threshold=0.2, it_limit=30)
            self.total_target_cnt += solver.op_result.nfev
            if not solver.op_result.success:
                self.not_solved += 1
                return 100500
                # continue
            self.initial_x[(pad, well, i)] = solver.edges_x.copy()

            rez_p_i[i] = well_G.nodes[nodes[2]]['obj'].result['P_bar']
            rez_p_b[i] = well_G.nodes[nodes[1]]['obj'].result['P_bar']
            rez = well_G[nodes[1]][nodes[2]]['obj'].result
            rez_q[i] = 86400 * rez['x'] / rez['liquid_density']

        self.last_well_result['p_zab'] = rez_p_b
        self.last_well_result['p_intake'] = rez_p_i
        self.last_well_result['q_well'] = rez_q

        ol_intake = self.outlayers['p_intake'][(pad, well)]
        ip_error = self.fact['p_intake'][(pad, well)] - rez_p_i
        ip_error = np.linalg.norm(ip_error[~ol_intake])
        score = ip_error / intake_scale

        ol_debit = self.outlayers['q_well'][(pad, well)]
        q_error = self.fact['q_well'][(pad, well)] - rez_q
        q_error = np.linalg.norm(q_error[~ol_debit])
        score += q_error / debit_scale

        score += np.linalg.norm(self.fact['p_zab'][(pad, well)] - rez_p_b) / bottom_scale
        return score


    def plot_single_well_chart(self, pad, well, well_res_before, well_res_after, well_G, nodes, op_result, xs, ys, zs, path):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f'Pad {pad} well {well} y {op_result.fun:.3f}, prod {op_result.x[0]:.3f}  pump_keff {op_result.x[1]:.3f}')
        # Xs, Ys = np.meshgrid(xs, ys)
        # Zs = zs.reshape(len(ys), len(xs))
        # im = plt.pcolormesh(Xs, Ys, Zs, cmap='plasma')
        # fig.colorbar(im, ax=ax, fraction=0.05, shrink=0.5)

        xs = np.round(xs, 3)
        ys = np.round(ys, 3)
        im, cbar = heatmap(zs, ys, xs, ax=ax, cmap="plasma")

        # i, j = np.unravel_index(np.argmin(zs, axis=None), zs.shape)
        # ax.plot(xs[i], ys[j], color='w', marker='X', markersize=15)

        ax.set_xlabel('prod')
        ax.set_ylabel('pump keff')
        plt.show()

        fig = plt.figure(constrained_layout=True, figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        WC = well_G.nodes[nodes[0]]['obj'].fluid.oil_params.volumewater_percent
        ax.set_title(f'Pad {pad} well {well} WC {WC}% y {op_result.fun:.3f}, prod {op_result.x[0]:.3f}  pump_keff {op_result.x[1]:.3f}')
        # ax.set_xlabel('time')
        ax.set_ylabel('P, Q')
        colors = plt.get_cmap('Set2').colors

        xs = np.array(range(self.N))
        kwargs = dict(linewidth=1, alpha=0.9)
        mask = ~self.outlayers['p_intake'][(pad, well)]
        hndl1 = ax.plot(xs[mask], self.fact['p_intake'][(pad, well)][mask], color=colors[0], label='Fact: intake', **kwargs)
        hndl2 = ax.plot(xs, self.fact['p_zab'][(pad, well)], color=colors[1], label='Fact: zaboi', **kwargs)

        mask = ~self.outlayers['q_well'][(pad, well)]
        hndl3 = ax.plot(xs[mask], self.fact['q_well'][(pad, well)][mask], color=colors[2], label='Fact: debit', **kwargs)

        kwargs = dict(linewidth=1, alpha=0.6)
        mask = ~self.outlayers['freq'][(pad, well)]
        hndl6 = ax.plot(xs[mask], self.fact['freq'][(pad, well)][mask], color=colors[4], label='Fact: freq', **kwargs)

        # kwargs = dict(linewidth=1, alpha=0.9, linestyle=':')
        # hndl4 = ax.plot(xs, well_res_before['p_intake'], color=colors[0], label='Before: intake', **kwargs)
        # ax.plot(xs, well_res_before['p_zab'], color=colors[1], **kwargs)
        # ax.plot(xs, well_res_before['q_well'], color=colors[2], **kwargs)

        kwargs = dict(linewidth=1, alpha=0.9, linestyle='--')
        hndl5 = ax.plot(xs, well_res_after['p_intake'], color=colors[0], label='Predict: intake', **kwargs)
        ax.plot(xs, well_res_after['p_zab'], color=colors[1], **kwargs)
        ax.plot(xs, well_res_after['q_well'], color=colors[2], **kwargs)

        # handles = hndl1 + hndl2 + hndl3 + hndl6 + hndl4 + hndl5
        handles = hndl1 + hndl2 + hndl3 + hndl6 + hndl5
        ax.legend(handles=handles, loc='lower right')
        plt.show()


    def fill_outlayers(self):
        ol_freq = dict()
        ol_intake = dict()
        ol_debit = dict()
        for pad, well in self.pad_well_list:
            freq = self.fact['freq'][(pad, well)]
            mask1 = freq > 60
            mask2 = freq < 40
            mask3 = np.isnan(freq)
            mask = mask1 | mask2 | mask3
            ol_freq[(pad, well)] = mask

            intake = self.fact['p_intake'][(pad, well)]
            mask1 = intake > 70
            mask2 = intake < 20
            mask3 = np.isnan(intake)
            mask = mask1 | mask2 | mask3
            ol_intake[(pad, well)] = mask

            debit = self.fact['q_well'][(pad, well)]
            mask1 = debit > 1000
            mask2 = debit < 0.5
            mask3 = np.isnan(debit)
            mask = mask1 | mask2 | mask3
            ol_debit[(pad, well)] = mask

        self.outlayers['freq'] = ol_freq
        self.outlayers['p_intake'] = ol_intake
        self.outlayers['q_well'] = ol_debit


    def greed_optimization(self):
        random.seed = 42
        self.gimme_wells()
        order = []
        for pad, well in self.pad_well_list:
            order += [(pad, well, 'K_prod'), (pad, well, 'K_pump')]
        random.shuffle(order)
        order = order * 10
        score = 100500100500
        for pad, well, param in order:
            pass






class HE2_PMNetwork_Model():
    def __init__(self, input_df, method=None, use_bounds=False, fit_version=None):
        # 1750023893
        # 1750024074
        # 1750040926
        # 1750026636
        # 1750037686
        # 1750028316
        # 1750037786

        df = input_df
        df.columns = model.do_upcase_columns_adhoc(df.columns)
        df.loc[df.node_id_start == 1750023893, 'start_P'] = np.nan
        df.loc[df.node_id_start == 1750024074, 'start_P'] = np.nan
        df.loc[df.node_id_start == 1750040926, 'start_P'] = np.nan
        df.loc[df.node_id_start == 1750026636, 'start_P'] = np.nan
        df.loc[df.node_id_start == 1750037686, 'start_P'] = np.nan
        df.loc[df.node_id_start == 1750028316, 'start_P'] = np.nan
        df.loc[df.node_id_start == 1750037786, 'start_P'] = np.nan
        self.input_df = df

        self.row_count = self.input_df.shape[0]
        N = self.row_count
        self.d_weights = np.ones(N)
        self.r_weights = np.ones(N)
        self.d_min = np.zeros(N) + 1e-5
        self.d_max = np.ones(N) * 1.2
        self.r_min = np.zeros(N)
        self.r_max = np.ones(N) * 100
        self.reg_d_deviation_keff = 1
        self.reg_r_deviation_keff = 1
        self.reg_d_dispersion_keff = 1
        self.reg_r_dispersion_keff = 1
        self.reg_total_keff = 1
        self.target_columns = {'start_P': 'result_start_P', 'end_P': 'result_end_P'}
        self.max_it = 100
        self.it = 0
        self.y0 = 0
        self.best_x = None
        self.best_y = 100500
        self.minimize_method = method
        self.use_bounds = use_bounds
        self.fit_version = fit_version
        self.best_rez_df = None

    def regularizator(self):
        N = self.row_count
        terms = np.zeros(4)
        terms[0] = np.linalg.norm((np.ones(N) - self.d_weights)) * self.reg_d_deviation_keff
        terms[1] = np.linalg.norm((np.ones(N) - self.r_weights)) * self.reg_r_deviation_keff
        terms[2] = np.std(self.d_weights) * self.reg_d_dispersion_keff
        terms[3] = np.std(self.r_weights) * self.reg_r_dispersion_keff
        return np.sum(terms) * self.reg_total_keff

    def apply_weights_to_model_long(self, x):
        N = self.row_count
        d_w = x[:N]
        r_w = x[N:]

        d_w = np.array([d_w, self.d_min]).max(axis=0)
        d_w = np.array([d_w, self.d_max]).min(axis=0)

        r_w = np.array([r_w, self.r_min]).max(axis=0)
        r_w = np.array([r_w, self.r_max]).min(axis=0)

        self.d_weights = d_w
        self.r_weights = r_w

        df = self.input_df.copy()
        df.D = df.D * d_w
        df.roughness = df.roughness * r_w
        return df

    def apply_weights_to_model(self, x):
        N = self.row_count
        d_w = x

        d_w = np.array([d_w, self.d_min]).max(axis=0)
        d_w = np.array([d_w, self.d_max]).min(axis=0)

        self.d_weights = d_w
        self.r_weights = np.ones(N)

        df = self.input_df.copy()
        df.D = df.D * d_w
        return df

    def evaluate_y(self, df):
        if df is None:
            return 100500100500
        y = 0
        for col_known, col_rez in self.target_columns.items():
            mask = ~df.loc[:, col_known].isna()
            known_y = df.loc[mask, col_known].values
            result_y = df.loc[mask, col_rez].values
            y += np.linalg.norm(known_y - result_y)
        return y

    def target(self, x):
        self.it += 1
        if self.it > self.max_it:
            raise Exception()
        df = self.apply_weights_to_model(x)
        rez_df = model.do_predict(df)
        y_term = self.evaluate_y(rez_df)
        reg_term = self.regularizator()
        rez = reg_term + y_term
        if (rez < self.best_y) or (self.it % 25 == 0):
            print('       ', self.it, rez)
        if rez < self.best_y:
            self.best_x = x
            self.best_y = rez
            self.best_rez_df = rez_df
        return rez

    def fit_v0(self):
        N = self.row_count
        self.it = 0
        # bnds = None
        # if self.use_bounds:
        # bnds = list(zip(self.d_min, self.d_max)) + list(zip(self.r_min, self.r_max))
        # bnds = list(zip(self.d_min, self.d_max))

        bnds = list(zip(list(np.ones(self.row_count) * 0.35), list(np.ones(self.row_count) * 1.2)))

        # baseline_y = self.target(np.ones(2*N))
        print('------------baseline_y = 240.23416053699975---------------')

        target = self.target
        # x0 = np.ones(2*N) * 0.87

        x0 = np.array([0.64338076, 0.82381463, 1., 0.96528834, 0.96071016,
                       0.95635247, 1., 0.92370242, 1., 0.38750445,
                       0.65503074, 0.95287122, 1., 1., 1.,
                       1., 1., 1., 0.97441723, 0.43580423,
                       0.87804864, 0.9425268, 0.94052543, 0.96233901, 0.9806694,
                       0.96689355, 0.4244979, 1., 0.95379705, 0.95644556,
                       0.68955511, 0.60984356, 0.96924162, 0.65517201, 0.97830807,
                       0.95069912, 0.54673536, 0.952645, 0.92537257, 0.7111141,
                       0.97352889, 0.63872855, 0.95075268, 0.92938683, 0.56179554,
                       0.73654833, 0.72238996, 0.76083504, 0.60044676, 0.91387568,
                       0.59788846, 0.74575036, 0.5212663, 0.46514458, 0.69942246,
                       0.77804001, 0.61124401, 0.80702349, 0.92731837, 0.52031628,
                       0.95585672, 0.92850279, 0.97973397, 0.91646619, 0.91254402,
                       0.91254402, 0.93168514, 0.91253429, 0.96657134, 0.98692722,
                       0.91255251, 0.91253575, 0.9456361, 0.98287691, 0.95358789,
                       0.91255343, 0.94552903])

        try:
            op_result = scop.minimize(target, x0, method=self.minimize_method, bounds=bnds, options=dict(nfev=100500))
        except:
            op_result = scop.OptimizeResult(success=True, fun=self.best_y, x=self.best_x, nfev=self.it)
        return op_result

    def fit_v1(self):
        self.it = 0

        # baseline_y = self.target(np.ones(2*N))
        print('------------baseline_y = 54.896212---------------')

        target = self.target
        x0 = np.array([0.64338076, 0.82381463, 1., 0.96528834, 0.96071016,
                       0.95635247, 1., 0.92370242, 1., 0.38750445,
                       0.65503074, 0.95287122, 1., 1., 1.,
                       1., 1., 1., 0.97441723, 0.43580423,
                       0.87804864, 0.9425268, 0.94052543, 0.96233901, 0.9806694,
                       0.96689355, 0.4244979, 1., 0.95379705, 0.95644556,
                       0.68955511, 0.60984356, 0.96924162, 0.65517201, 0.97830807,
                       0.95069912, 0.54673536, 0.952645, 0.92537257, 0.7111141,
                       0.97352889, 0.63872855, 0.95075268, 0.92938683, 0.56179554,
                       0.73654833, 0.72238996, 0.76083504, 0.60044676, 0.91387568,
                       0.59788846, 0.74575036, 0.5212663, 0.46514458, 0.69942246,
                       0.77804001, 0.61124401, 0.80702349, 0.92731837, 0.52031628,
                       0.95585672, 0.92850279, 0.97973397, 0.91646619, 0.91254402,
                       0.91254402, 0.93168514, 0.91253429, 0.96657134, 0.98692722,
                       0.91255251, 0.91253575, 0.9456361, 0.98287691, 0.95358789,
                       0.91255343, 0.94552903])

        # x0 = np.ones(self.row_count) * 0.87

        try:
            op_result = scop.basinhopping(target, x0, 100500, T=100)
        except:
            op_result = scop.OptimizeResult(success=True, fun=self.best_y, x=self.best_x, nfev=self.it)
        return op_result

    def fit_v2(self):
        self.it = 0

        # baseline_y = self.target(np.ones(2*N))
        print('------------baseline_y = 54.896212---------------')

        target = self.target
        bnds = list(zip(self.d_min, self.d_max))

        try:
            op_result = scop.differential_evolution(target, bounds=bnds, workers=2)
        except:
            op_result = scop.OptimizeResult(success=True, fun=self.best_y, x=self.best_x, nfev=self.it)
        return op_result

    def fit_v3(self):
        self.it = 0

        print('------------baseline_y = 54.896212---------------')
        print('op_result = scop.shgo(target, bounds=bnds)')

        target = self.target
        bnds = list(zip(list(np.ones(self.row_count) * 0.35), list(np.ones(self.row_count) * 1.0)))

        try:
            op_result = scop.shgo(target, bounds=bnds)
        except:
            op_result = scop.OptimizeResult(success=True, fun=self.best_y, x=self.best_x, nfev=self.it)
        return op_result

    def fit_v4(self):
        self.it = 0

        print('------------baseline_y = 54.896212---------------')
        print('op_result = scop.dual_annealing')

        target = self.target
        bnds = list(zip(list(np.ones(self.row_count) * 0.35), list(np.ones(self.row_count) * 1.0)))
        x0 = np.array([0.9368659, 0.90908377, 0.66269683, 0.84984735, 0.94433411,
                       0.94433411, 0.81021361, 0.85424077, 0.89137211, 0.84981522,
                       0.94433411, 0.8586045, 0.94433411, 0.94433411, 0.94433411,
                       0.94433411, 0.94433411, 0.94433411, 0.84980174, 0.44306014,
                       0.87813078, 0.84980174, 0.84980174, 0.84980174, 0.84980174,
                       0.93782366, 0.84459252, 0.93500233, 0.84980174, 0.84980174,
                       0.78761383, 0.59268382, 0.84980174, 0.62411044, 0.73925876,
                       0.82262774, 0.55721666, 0.73925876, 0.69122549, 0.59343754,
                       0.84980174, 0.64804504, 0.65627441, 0.84980174, 0.59175647,
                       0.70158817, 0.58099541, 0.61674218, 0.61099233, 0.7366823,
                       0.4614739, 0.66874503, 0.55128615, 0.47035048, 0.61580247,
                       0.71251471, 0.53147553, 0.64558384, 0.64558384, 0.52497384,
                       0.84980174, 0.84980174, 0.83987027, 0.71487156, 0.84980173,
                       0.84980173, 0.756754, 0.53656503, 0.84980173, 0.84980174,
                       0.84980846, 0.84979832, 0.8498047, 0.84980606, 0.84979465,
                       0.84981028, 0.84980687])

        try:
            op_result = scop.dual_annealing(target, bounds=bnds, x0=x0, maxiter=100500)
        except:
            op_result = scop.OptimizeResult(success=True, fun=self.best_y, x=self.best_x, nfev=self.it)
        return op_result

    def fit_v5(self):
        self.it = 0

        print('------------baseline_y = 54.896212---------------')
        print('op_result = scop.dual_annealing')

        target = self.target
        bnds = list(zip(list(np.ones(self.row_count) * 0.35), list(np.ones(self.row_count) * 1.1)))
        x0 = np.array([0.64338076, 0.82381463, 1., 0.96528834, 0.96071016,
                       0.95635247, 1., 0.92370242, 1., 0.38750445,
                       0.65503074, 0.95287122, 1., 1., 1.,
                       1., 1., 1., 0.97441723, 0.43580423,
                       0.87804864, 0.9425268, 0.94052543, 0.96233901, 0.9806694,
                       0.96689355, 0.4244979, 1., 0.95379705, 0.95644556,
                       0.68955511, 0.60984356, 0.96924162, 0.65517201, 0.97830807,
                       0.95069912, 0.54673536, 0.952645, 0.92537257, 0.7111141,
                       0.97352889, 0.63872855, 0.95075268, 0.92938683, 0.56179554,
                       0.73654833, 0.72238996, 0.76083504, 0.60044676, 0.91387568,
                       0.59788846, 0.74575036, 0.5212663, 0.46514458, 0.69942246,
                       0.77804001, 0.61124401, 0.80702349, 0.92731837, 0.52031628,
                       0.95585672, 0.92850279, 0.97973397, 0.91646619, 0.91254402,
                       0.91254402, 0.93168514, 0.91253429, 0.96657134, 0.98692722,
                       0.91255251, 0.91253575, 0.9456361, 0.98287691, 0.95358789,
                       0.91255343, 0.94552903])
        x0 = np.ones(self.row_count) * 0.9

        op_result = None
        for i in range(20):
            op_result = scop.dual_annealing(target, bounds=bnds, x0=x0, maxfun=1000)
            x0 = op_result.x
        return op_result

    def fit(self):
        fits = {0: self.fit_v0, 1: self.fit_v1, 2: self.fit_v2, 3: self.fit_v3, 4: self.fit_v4, 5: self.fit_v5}
        meth = fits.get(self.fit_version, self.fit_v0)
        return meth()


bad_wells = [(738, 33), (567, 39), (4532, 49), (2630, 49), (1579, 57), (3118, 57)]


if __name__ == '__main__':
    model = HE2_OilGatheringNetwork_Model("../../")
    model.fact = model.grab_fact()
    model.fill_outlayers()
    model.bad_wells = bad_wells
    rez = model.prefit_all()
    # f = open('rez.csv', 'w')
    # for item in rez:
    #     print(item, file=f)
    # model.solve_em_all()
    # model.result = model.grab_results()
    # model.plot_fact_and_results(keys_to_plot=('head'))
    # model.plot_fact_and_results(keys_to_plot=('intake'))
    # model.plot_fact_and_results(keys_to_plot=('bottom'))
    # model.plot_fact_and_results(keys_to_plot=('debit'))
    # model.calc_well_score(5, 1523)
