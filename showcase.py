from Fluids.HE2_Fluid import HE2_BlackOil
#from Tests.Optimization_test import model_DNS_2, build_DNS2_graph, gimme_DNS2_inlets_outlets_Q
#from Tests.Optimization_test import model_DNS_2_by_parts
import pandas as pd
from Tools.HE2_ABC import oil_params
from Solver.HE2_Solver import HE2_Solver
from Tools.HE2_tools import check_solution, print_wells_pressures
from GraphEdges.HE2_WellPump import create_HE2_WellPump_instance_from_dataframe
from GraphEdges.HE2_Pipe import HE2_OilPipe
import numpy as np

"""
oil_params - описание ФХС водонефтяной смеси, но данном этапе - усредненные ФХС Тайлаковского месторождения
pressures - перечень пластовых давлений всех используемых в модели скважин

plasts - описание свойств пласта используемых в модели скважин. В настоящий момент задается коэффициентом продуктивности.
В датайрейме для оптимизации он в поле productCoeff.
Коэффициент продуктивности можно вычислить по историческим данным (глубина данных 1-2 месяца должна быть оптимальной) 
как efficiency = Debit / (P_plast - P_zaboi),
где Debit - суточный дебит скважины, P_plast - пластовое давление, бар, P_zaboi - забойное давление, бар

pumps - описание насосов скважин, задается моделью насоса и частотой работы
"""
Tailaki_oil_params = oil_params(sat_P_bar=67, plastT_C=84, gasFactor=36, oildensity_kg_m3=826,
                                        waterdensity_kg_m3=1015, gasdensity_kg_m3=1, oilviscosity_Pa_s=35e-3, volumewater_percent=50, volumeoilcoeff=1.017)

pump_curves = pd.read_csv("../CommonData/PumpChart.csv")

fluid = HE2_BlackOil(Tailaki_oil_params)
pressures = {"PAD_5": {"WELL_1523" : 270.5, "WELL_146" : 270.5, "WELL_142" : 270.5, "WELL_1562" : 268.5},
             "PAD_33":  {"WELL_1385" : 268.5, "WELL_736" : 268.5, "WELL_739" : 270.5, "WELL_1383" : 270.5, "WELL_738" : 268.5,"WELL_725" : 268.5},
             "PAD_34": {"WELL_731" : 270.5, "WELL_196" : 268.5, "WELL_734" : 270.5, "WELL_198" : 270.5, "WELL_199" : 270.5,"WELL_197" : 268.5,
                        "WELL_195" : 268.5, "WELL_191" : 268.5, "WELL_729" : 268.5, "WELL_730" : 268.5, "WELL_192" : 268.5,"WELL_148" : 270.5},
             "PAD_39": {"WELL_3552" : 270.5, "WELL_617" : 270.5, "WELL_567" : 270.5, "WELL_614" : 270.5, "WELL_619" : 270.5,"WELL_609" : 268.5},
             "PAD_49":  {"WELL_1816" : 268.5, "WELL_2630" : 268.5, "WELL_1815" : 268.5, "WELL_676" : 268.5, "WELL_3270" : 268.5,
                         "WELL_3266" : 268.5, "WELL_1814" : 268.5, "WELL_1817" : 268.5, "WELL_4532" : 268.5, "WELL_2631" : 268.5,
                         "WELL_677" : 268.5},
             "PAD_57": {"WELL_3113" : 268.5, "WELL_3118" : 268.5, "WELL_3112" : 268.5, "WELL_4235" : 268.5, "WELL_3117" : 268.5,
                         "WELL_1493" : 268.5, "WELL_1574" : 268.5, "WELL_1579" : 268.5, "WELL_3116" : 268.5}}


plasts = {"PAD_5": {"WELL_1523":0.237, "WELL_146": 0.416, "WELL_142": 0.158,"WELL_1562": 0.276},

"PAD_33": {"WELL_1385" : 0.45, "WELL_736" : 0.3, "WELL_739" : 0.378, "WELL_1383" : 0.5, "WELL_738" : 0.33,"WELL_725" : 0.5},
# "PAD_33": {"WELL_1385" : 0.25, "WELL_736" : 0.2, "WELL_739" : 0.378, "WELL_1383" : 0.25, "WELL_738" : 0.53,"WELL_725" : 0.5},

"PAD_34": {"WELL_731" : 0.347, "WELL_196" : 0.6, "WELL_734" : 0.6, "WELL_198" : 0.4, "WELL_199" : 0.457,"WELL_197" : 0.236,
    "WELL_195" : 0.27, "WELL_191" : 0.227, "WELL_729" : 1.4, "WELL_730" : 0.189, "WELL_192" : 0.445,"WELL_148" : 0.232},

"PAD_39": {"WELL_3552": 0.98, "WELL_617": 0.348, "WELL_567": 0.034, "WELL_614": 0.98, "WELL_619": 0.31,"WELL_609": 0.235},

"PAD_49":  {"WELL_1816" : 1.66, "WELL_2630" : 0.05, "WELL_1815" : 0.516, "WELL_676" : 0.823, "WELL_3270" : 0.288,
    "WELL_3266" : 0.492, "WELL_1814" : 1.259, "WELL_1817" : 0.695, "WELL_4532" : 0.293, "WELL_2631" : 0.68, "WELL_677" : 0.37},

"PAD_57": {"WELL_3113" : 0.515, "WELL_3118" : 0.052, "WELL_3112" : 0.5, "WELL_4235" : 0.17, "WELL_3117" : 1.29,
    "WELL_1493" : 0.792, "WELL_1574" : 0.53, "WELL_1579" : 0.197, "WELL_3116" : 0.98}
}


pumps = {"PAD_5": {"WELL_1523":["ЭЦН5-80-2500", 48], "WELL_146": ["ЭЦН5-60-2450", 58], "WELL_142": ["ЭЦН5-45-2500", 45],"WELL_1562": ["ЭЦН5-50-2500", 50]},

"PAD_33": {"WELL_1385" : ["ЭЦН5-30-2600", 60], "WELL_736" : ["ЭЦН5-30-2600", 60], "WELL_739" : ["ЭЦН5-80-2200", 50],
    "WELL_1383" : ["ЭЦН5-80-2550", 50], "WELL_738" : ["ЭЦН5-80-2500", 43],"WELL_725" : ["ЭЦН5-80-2500", 50]},

"PAD_34": {"WELL_731" : ["ЭЦН5-80-2550", 45], "WELL_196" : ["ЭЦН5-125-2500", 50], "WELL_734" : ["ЭЦН5-80-2200", 55],
    "WELL_198" :["ЭЦН5-80-2550", 50], "WELL_199" : ["ЭЦН5-80-2600", 45],"WELL_197" : ["ЭЦН5-80-2550", 45],
    "WELL_195" : ["ЭЦН5-60-2500", 45], "WELL_191" : ["ЭЦН5-60-2450", 50], "WELL_729" :["ЭЦН5А-250-2400", 50],
    "WELL_730" : ["ЭЦН5-45-2400", 40], "WELL_192" : ["ЭЦН5-80-2650", 45],"WELL_148" : ["ЭЦН5-60-2450", 40]},

"PAD_39": {"WELL_3552": ["ЭЦН5А-160-2400", 50], "WELL_617": ["ЭЦН5-80-2500", 50], "WELL_567": ["ЭЦН5-50-2450", 42],
    "WELL_614": ["ЭЦН5А-160-2500", 55], "WELL_619": ["ЭЦН5-80-2550", 45],"WELL_609": ["ЭЦН5-80-2400", 45]},

"PAD_49":  {"WELL_1816" : ["ЭЦН5А-320-2400", 50], "WELL_2630" : ["ЭЦН5-80-2400", 45], "WELL_1815" : ["ЭЦН5А-160-2500", 45],
            "WELL_676" : ["ЭЦН5А-160-2600", 50], "WELL_3270" : ["ЭЦН5-45-2500", 50], "WELL_3266" : ["ЭЦН5-80-2700", 50],
            "WELL_1814" : ["ЭЦН5А-250-2400", 50], "WELL_1817" : ["ЭЦН5А-160-2600", 50], "WELL_4532" : ["ЭЦН5-60-2400", 50],
            "WELL_2631" : ["ЭЦН5А-160-2400", 50], "WELL_677" : ["ЭЦН5-80-2600", 50]},

"PAD_57": {"WELL_3113" : ["ЭЦН5-125-2500", 45], "WELL_3118" : ["ЭЦН5-80-2500", 45], "WELL_3112" : ["ЭЦН5-80-2400", 57],
           "WELL_4235" : ["ЭЦН5-80-2500", 45], "WELL_3117" : ["ЭЦН5А-250-2500", 50], "WELL_1493" : ["ЭЦН5А-160-2500", 50],
           "WELL_1574" : ["ЭЦН5-125-2450", 50], "WELL_1579" : ["ЭЦН5-80-2500", 45], "WELL_3116" : ["ЭЦН5А-250-2400", 50]}
}

inclination = {
    '1523': (32.6, 2561, 0.125, 267.3, 202.7, 0.143),
    '146': (47.569, 2575.43, 0.125, 161.39, 210.56, 0.143),
    '142': (324.89, 2511.1, 0.125, 242.24, 249.7599, 0.143),
    '1562':(549.92, 2392.08, 0.125, 36.4499, 325.55, 0.143),
    '1385': (395.8, 2139.2, 0.125, 80, 584, 0.143),
    '736': (216, 2532, 0.125, 448, 155, 0.143),
    '739': (323.45, 2356.55, 0.125, 396.35, 290.6, 0.143),
    '1383': (526, 2353.99, 0.125, 30.16, 377.84, 0.143),
    '738': (243.83, 2128.17, 0.125, 402.45, 526.85, 0.143),
    '725': (495.76, 2329.24, 0.125, 28.7, 384.3, 0.143),
    '731': (285.7, 2414.27, 0.125, 13.5, 330, 0.143),
    '196': (172.6, 2504.43, 0.125, 7.38, 245.12, 0.143),
    '734': (32.21, 2362.8, 0.125, 282, 286.88, 0.143),
    '198': (244, 2586, 0.125, 4.82, 169.18, 0.143),
    '199': (228.11, 2520.89, 0.125, 198.34, 241.56, 0.143),
    '197': (668.11, 2345.89, 0.125, 52.91, 403.09, 0.143),
    '195': (610.82, 2372.96, 0.125, 50.06, 357.93, 0.143),
    '191': (254.37, 2585.63, 0.125, 251.19, 166.221, 0.143),
    '729': (92.78, 2345.22, 0.125, 247.95, 399.048, 0.143),
    '730': (75.76, 2218.24, 0.125, 99.72, 532.78, 0.143),
    '192': (337.39, 2177.61, 0.125, 255.33, 531.87, 0.143),
    '148': (337.39, 2177.61, 0.125, 255.33, 531.87, 0.143),
    '3552': (130.37, 2169.63, 0.125, 155.83, 687.06, 0.143),
    '617': (461.4, 2409.6, 0.125, 8, 333, 0.143),
    '567': (674.06, 2165.94, 0.125, 112.83, 554.17, 0.143),
    '614': (128.89, 2529.07, 0.125, 156.93, 215.51, 0.143),
    '619': (269.27, 2343.73, 0.125, 110.95, 384.75, 0.143),
    '609': (117.42, 2291.58, 0.125, 202.87, 535.23, 0.143),
    '1816': (61.69, 2508.31, 0.125, 811.92, 67.67, 0.143),
    '2630': (419.86, 2370.14, 0.125, 570.43, 310.38, 0.143),
    '1815': (113.77, 2516.23, 0.125, 482.82, 333.82, 0.143),
    '676': (248.29, 2433.71, 0.125, 537.87, 336.76, 0.143),
    '3270': (310.83, 2080.17, 0.125, 315.89, 637.11, 0.143),
    '3266': (119.76, 2500.24, 0.125, 280.05, 489.3, 0.143),
    '1814': (205.63, 2455.37, 0.125, 186, 600.24, 0.143),
    '1817': (324.58, 2443.42, 0.125, 690.26, 244.53, 0.143),
    '4532': (363.75, 2482.25, 0.125, 403.62, 509.5, 0.143),
    '2631': (67.5, 2394.5, 0.125, 626.2, 278.96, 0.143),
    '677': (248.2, 2351.8, 0.125, 728.15, 217.82, 0.143),
    '3113': (382.13, 2367.87, 0.125, 652.87, 230.96, 0.143),
    '3118': (152.43, 2458.57, 0.125, 670.7, 109, 0.143),
    '3112': (55.74, 2481.26, 0.125, 753.03, -3, 0.143),
    '4235': (351.06, 2514.94, 0.125, 511.21, 253.89, 0.143),
    '3117': (117.8, 2477.2, 0.125, 527.58, 255.96, 0.143),
    '1493': (403.96, 2442.04, 0.125, 409.48, 399.746, 0.143),
    '1574': (130.42, 2514.58, 0.125, 679.53, -18.46, 0.143),
    '1579': (425.39, 2494.6, 0.125, 345.25, 281.16, 0.143),
    '3116': (167.57, 2453.43, 0.125, 715.74, 81.184, 0.143)
}

def full_test():
    Full_system_daily_debit = 5200
    G, inlets, juncs, outlets = model_DNS_2(pressures=pressures, pumps=pumps, plasts=plasts,
                                            DNS_daily_debit=Full_system_daily_debit, pump_curves=pump_curves, fluid=fluid)

    print_wells_pressures(G, inlets)

    validity = check_solution(G)
    print(validity)



def part_test():
    Full_system_daily_debit = 4750
    # well_list = ['PAD_49_well_1816']
    model_DNS_2_by_parts(pressures=pressures, plasts=plasts, pumps=pumps, pump_curves=pump_curves, fluid=fluid,
                         DNS_daily_debit=Full_system_daily_debit)

def test_with_change_graph_on_the_fly():
    # Выходное условие заменено на выходное давление ДНС-2
    outputpressure = 4.8
    roughness = 1e-5
    real_diam_coefficient = 0.85

    #Получаем расчетный граф с заданными параметрами
    G, inlets, juncs, outlets = build_DNS2_graph(pressures=pressures, plasts=plasts, pumps=pumps,
                                                 pump_curves=pump_curves, fluid=fluid, roughness=roughness,
                                                 real_diam_coefficient=real_diam_coefficient,
                                                 DNS_pressure=outputpressure)
    # Создаем солвер и решаем полученный расчетный граф
    inlets_Q = gimme_DNS2_inlets_outlets_Q()
    #Создаем солвер и решаем полученный расчетный граф
    solver = HE2_Solver(G)
    solver.set_known_Q(inlets_Q)
    solver.solve()

    validity = check_solution(G)
    print(validity)

    print_wells_pressures(G, inlets)
    for n in outlets:
        print(n, G.nodes[n]["obj"].result)

    #Меняем давление на пласте скважины 1523 куста 5

    G.nodes["PAD_5_well_1523"]['obj'].value = 250


    solver.solve()
    print("\nРешение с измененным давлением")

    print_wells_pressures(G, inlets)
    for n in outlets:
        print(n, G.nodes[n]["obj"].result)

    #Меняем частоту насоса скважины 1562 куста 5

    G["Pump_intake_1562"]['Pump_outlet_1562']['obj'].changeFrequency(55)

    solver.solve()
    print("\nРешение с измененной частотой насоса")

    print_wells_pressures(G, inlets)
    for n in outlets:
        print(n, G.nodes[n]["obj"].result)

    #Меняем модель насоса скважины 1562 куста 5
    edge = G['Pump_intake_1562']['Pump_outlet_1562']
    new_pump = create_HE2_WellPump_instance_from_dataframe(pump_curves, 'ЭЦН5-45-2500', fluid, 50)
    edge['obj'] = new_pump
    solver.ready_for_solve = False # Иначе солвер не обновит внутренние edge_func

    solver.solve()
    print("\nРешение с измененной частотой насоса")

    print_wells_pressures(G, inlets)
    for n in outlets:
        print(n, G.nodes[n]["obj"].result)
    print(f'Well 1562 pump power is {new_pump.power} Wt')



import shame_on_me

def test_2pads_with_change_graph():
    # Урезаная модель, ост ДНС2 оставили только 2 куста, к.33 и к.34
    # Выходное условие заменено на выходное давление ДНС-2
    outputpressure = 4.8
    roughness = 1e-5
    real_diam_coefficient = 0.85

    #Получаем расчетный граф с заданными параметрами
    G, inlets, juncs, outlets = shame_on_me.build_DNS2_graph_pads33_34(pressures=pressures, plasts=plasts, pumps=pumps,
                                                                       pump_curves=pump_curves, fluid=fluid,
                                                                       roughness=roughness,
                                                                       real_diam_coefficient=real_diam_coefficient,
                                                                       DNS_pressure=outputpressure)
    # Создаем солвер и решаем полученный расчетный граф
    inlets_Q = gimme_DNS2_inlets_outlets_Q()
    #Создаем солвер и решаем полученный расчетный граф
    solver = HE2_Solver(G)
    solver.set_known_Q(inlets_Q)
    solver.solve()

    print_wells_pressures(G, inlets)
    for n in outlets:
        print(n, G.nodes[n]["obj"].result)

    #Меняем давление на пласте скважины 1523 куста 5

    G.nodes["PAD_33_well_1385"]['obj'].value = 250


    solver.solve()

    print("\nРешение с измененным давлением")
    print_wells_pressures(G, inlets)
    for n in outlets:
        print(n, G.nodes[n]["obj"].result)


    #Меняем частоту насоса скважины 1562 куста 5

    G.edges._adjdict["Pump_intake_725"]['Pump_outlet_725']['obj'].changeFrequency(55)

    solver.solve()
    print("\nРешение с измененной частотой насоса")
    print_wells_pressures(G, inlets)

    for n in outlets:
        print(n, G.nodes[n]["obj"].result)

def fluids_full_test():
    Full_system_daily_debit = 5200
    G, inlets, juncs, outlets = build_DNS2_graph(pressures, plasts, pumps, pump_curves, fluid, 1e-5, 0.8, Full_system_daily_debit, DNS_pressure=4.8)
    np.random.seed(5)
    wcs = np.random.uniform(0, 100, len(inlets))
    for i, n in enumerate(inlets):
        obj = G.nodes[n]['obj']
        op_dict = obj.fluid.oil_params._asdict()
        op_dict.update(volumewater_percent = wcs[i])
        new_op = oil_params(**op_dict)
        obj.fluid = HE2_BlackOil(new_op)

    solver = HE2_Solver(G)
    solver.solve(threshold=0.05)

    print_wells_pressures(G, inlets)
    validity = check_solution(G)
    print(validity.first_CL_OWG_resd)


def full_test_with_fluids_and_random_edges():
    Full_system_daily_debit = 5200
    np.random.seed(5)
    wcs = np.random.uniform(0, 100, 100)

    f = open('randoms.txt')

    diffs = []
    not_solved = []
    invalid_OWG = dict()
    for i, sss in enumerate(f):
        # if not (i in [80, 98, 370, 450, 680]):
        if not (i in [370, 680]):
            continue
        print(f'-----------------------------------------{i}----------------------------------')
        params = sss[:-1].split(';')
        G, inlets, juncs, outlets = build_DNS2_graph(pressures, plasts, pumps, pump_curves, fluid, 1e-5, 0.8,
                                                     Full_system_daily_debit, DNS_pressure=4.8)

        for j, n in enumerate(inlets):
            obj = G.nodes[n]['obj']
            op_dict = obj.fluid.oil_params._asdict()
            op_dict.update(volumewater_percent=wcs[j])
            new_op = oil_params(**op_dict)
            obj.fluid = HE2_BlackOil(new_op)

        u, v = params[0:2]
        dx, dy, diam, rgh = tuple(map(float, params[2:6]))
        G.add_edge(u, v, obj=HE2_OilPipe([dx], [dy], [diam], [rgh], fluid))

        u, v = params[6:8]
        dx, dy, diam, rgh = tuple(map(float, params[8:12]))
        G.add_edge(u, v, obj=HE2_OilPipe([dx], [dy], [diam], [rgh], fluid))

        u1, v1, u2, v2 = params[12:16]

        solver = HE2_Solver(G)
        solver.solve(threshold=0.25, it_limit=100)
        # print_solution(G)
        x1 = solver.op_result.x
        ch1 = list(solver.chordes)
        if not solver.op_result.success:
            not_solved += [i]
            continue

        validity = check_solution(G)
        print(validity.first_CL_OWG_resd)
        if validity.first_CL_OWG_resd > 0.1:
            invalid_OWG[i] = validity.first_CL_OWG_resd

        # G = reverse_edge_in_graph(G, u1, v1)
        # G = reverse_edge_in_graph(G, u2, v2)
        #
        # solver = HE2_Solver(G)
        # solver.solve(threshold=0.25, it_limit=100)
        # # print_solution(G)
        #
        # l1 = sorted(list(abs(x1.flatten())))
        # l2 = []
        # for u, v in ch1:
        #     if (u, v) in solver.edges_x:
        #         l2 += [abs(solver.edges_x[(u, v)])]
        #     elif (v, u) in solver.edges_x:
        #         l2 += [abs(solver.edges_x[(v, u)])]
        #     else:
        #         assert False
        # l2 = sorted(l2)
        #
        # diff = np.linalg.norm(np.array(l1) - np.array(l2))
        # if diff > 5e-3:
        #     diffs += [i]
        #
        # print(diffs)
    print(not_solved)
    print(invalid_OWG)


if __name__ == '__main__':
    # test_2pads_with_change_graph()
    # test_with_change_graph_on_the_fly()
    # fluids_full_test()
    #part_test()
    full_test_with_fluids_and_random_edges()
