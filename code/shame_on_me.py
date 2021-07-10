import networkx as nx
from Fluids.HE2_Fluid import HE2_BlackOil
from GraphNodes import HE2_Vertices as vrtxs
from GraphEdges.HE2_Pipe import HE2_OilPipe
from GraphEdges.HE2_Plast import HE2_Plast
from Solver.HE2_Solver import HE2_Solver
from GraphEdges.HE2_WellPump import HE2_WellPump, create_HE2_WellPump_instance_from_dataframe
from Tools.HE2_ABC import oil_params
import json
from Tools.HE2_Logger import check_for_nan, getLogger
logger = getLogger(__name__)


oil_params = oil_params(sat_P_bar=67, plastT_C=84, gasFactor=36, oildensity_kg_m3=826,
                        waterdensity_kg_m3=1015, gasdensity_kg_m3=1, oilviscosity_Pa_s=35e-3, volumewater_percent=50, volumeoilcoeff=1.017)
fluid = HE2_BlackOil(oil_params)


def build_DNS2_graph_pads33_34(pressures: dict = {}, plasts: dict = {}, pumps=None, pump_curves=None, fluid=None,
                               roughness=0.00001, real_diam_coefficient=1, DNS_daily_debit=0, DNS_pressure=4.8):
    json_str = json.dumps(pumps)
    logger.info(f'Pumps: {json_str}')

    # Давления в источниках
    pressure_1385 = pressures["PAD_33"]["WELL_1385"]
    pressure_736 = pressures["PAD_33"]["WELL_736"]
    pressure_739 = pressures["PAD_33"]["WELL_739"]
    pressure_1383 = pressures["PAD_33"]["WELL_1383"]
    pressure_738 = pressures["PAD_33"]["WELL_738"]
    pressure_725 = pressures["PAD_33"]["WELL_725"]

    pressure_731 = pressures["PAD_34"]["WELL_731"]
    pressure_196 = pressures["PAD_34"]["WELL_196"]
    pressure_734 = pressures["PAD_34"]["WELL_734"]
    pressure_198 = pressures["PAD_34"]["WELL_198"]
    pressure_199 = pressures["PAD_34"]["WELL_199"]
    pressure_197 = pressures["PAD_34"]["WELL_197"]
    pressure_195 = pressures["PAD_34"]["WELL_195"]
    pressure_191 = pressures["PAD_34"]["WELL_191"]
    pressure_729 = pressures["PAD_34"]["WELL_729"]
    pressure_730 = pressures["PAD_34"]["WELL_730"]
    pressure_192 = pressures["PAD_34"]["WELL_192"]
    pressure_148 = pressures["PAD_34"]["WELL_148"]

    # Источники системы
    inlets = dict(PAD_33_well_1385=vrtxs.HE2_Source_Vertex('P', pressure_1385, fluid, 20),  # Куст 33
                  PAD_33_well_736=vrtxs.HE2_Source_Vertex('P', pressure_736, fluid, 20),
                  PAD_33_well_739=vrtxs.HE2_Source_Vertex('P', pressure_739, fluid, 20),
                  PAD_33_well_1383=vrtxs.HE2_Source_Vertex('P', pressure_1383, fluid, 20),
                  PAD_33_well_738=vrtxs.HE2_Source_Vertex('P', pressure_738, fluid, 20),
                  PAD_33_well_725=vrtxs.HE2_Source_Vertex('P', pressure_725, fluid, 20),

                  PAD_34_well_731=vrtxs.HE2_Source_Vertex('P', pressure_731, fluid, 20),  # Куст 34
                  PAD_34_well_196=vrtxs.HE2_Source_Vertex('P', pressure_196, fluid, 20),
                  PAD_34_well_734=vrtxs.HE2_Source_Vertex('P', pressure_734, fluid, 20),
                  PAD_34_well_198=vrtxs.HE2_Source_Vertex('P', pressure_198, fluid, 20),
                  PAD_34_well_199=vrtxs.HE2_Source_Vertex('P', pressure_199, fluid, 20),
                  PAD_34_well_197=vrtxs.HE2_Source_Vertex('P', pressure_197, fluid, 20),
                  PAD_34_well_195=vrtxs.HE2_Source_Vertex('P', pressure_195, fluid, 20),
                  PAD_34_well_191=vrtxs.HE2_Source_Vertex('P', pressure_191, fluid, 20),
                  PAD_34_well_729=vrtxs.HE2_Source_Vertex('P', pressure_729, fluid, 20),
                  PAD_34_well_730=vrtxs.HE2_Source_Vertex('P', pressure_730, fluid, 20),
                  PAD_34_well_192=vrtxs.HE2_Source_Vertex('P', pressure_192, fluid, 20),
                  PAD_34_well_148=vrtxs.HE2_Source_Vertex('P', pressure_148, fluid, 20),
                  )

    inlets.update(
                PAD_5=vrtxs.HE2_Source_Vertex('Q', 2.636, fluid, 20),
                PAD_39=vrtxs.HE2_Source_Vertex('Q', 6.54356, fluid, 20),
                PAD_49=vrtxs.HE2_Source_Vertex('Q', 16.97373, fluid, 20),
                PAD_57=vrtxs.HE2_Source_Vertex('Q', 12.5264, fluid, 20),
                )

    # Узловые точки
    juncs = dict(
                 Zaboi_1385=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_1385=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_1385=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_1385=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_736=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_736=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_736=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_736=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_739=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_739=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_739=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_739=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_1383=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_1383=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_1383=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_1383=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_738=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_738=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_738=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_738=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_725=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_725=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_725=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_725=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_731=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_731=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_731=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_731=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_196=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_196=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_196=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_196=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_734=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_734=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_734=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_734=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_198=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_198=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_198=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_198=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_199=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_199=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_199=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_199=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_197=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_197=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_197=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_197=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_195=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_195=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_195=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_195=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_191=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_191=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_191=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_191=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_729=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_729=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_729=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_729=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_730=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_730=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_730=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_730=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_192=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_192=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_192=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_192=vrtxs.HE2_ABC_GraphVertex(),

                 Zaboi_148=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_intake_148=vrtxs.HE2_ABC_GraphVertex(),
                 Pump_outlet_148=vrtxs.HE2_ABC_GraphVertex(),
                 Wellhead_148=vrtxs.HE2_ABC_GraphVertex(),



                 PAD_33=vrtxs.HE2_ABC_GraphVertex(),
                 PAD_34=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_5=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_9_36=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_35=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_33=vrtxs.HE2_ABC_GraphVertex(),
                 intake_node_33=vrtxs.HE2_ABC_GraphVertex(),
                 intake_node_49=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_59=vrtxs.HE2_ABC_GraphVertex(),
                 node_15=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_46=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_47=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_57=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_41=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_49=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_53=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_134=vrtxs.HE2_ABC_GraphVertex(),
                 intake_pad_59_46=vrtxs.HE2_ABC_GraphVertex(),
                 UDR_1=vrtxs.HE2_ABC_GraphVertex(),
                 UDR_2=vrtxs.HE2_ABC_GraphVertex(),
                 ZKL_98=vrtxs.HE2_ABC_GraphVertex())

    q = DNS_daily_debit * fluid.calc(P_bar=20, T_C=20, X_kgsec=0).CurrentLiquidDensity_kg_m3 / 86400
    outlets = dict(DNS_2=vrtxs.HE2_Boundary_Vertex('P', DNS_pressure))

    G = nx.DiGraph()  # Di = directed
    for k, v in {**inlets, **outlets, **juncs}.items():
        G.add_node(k, obj=v)

    # Продуктивности скважин
    # Куст 33
    productivity_1385 = plasts["PAD_33"]["WELL_1385"]
    productivity_736 = plasts["PAD_33"]["WELL_736"]
    productivity_739 = plasts["PAD_33"]["WELL_739"]
    productivity_1383 = plasts["PAD_33"]["WELL_1383"]
    productivity_738 = plasts["PAD_33"]["WELL_738"]
    productivity_725 = plasts["PAD_33"]["WELL_725"]

    # Куст 34
    productivity_731 = plasts["PAD_34"]["WELL_731"]
    productivity_196 = plasts["PAD_34"]["WELL_196"]
    productivity_734 = plasts["PAD_34"]["WELL_734"]
    productivity_198 = plasts["PAD_34"]["WELL_198"]
    productivity_199 = plasts["PAD_34"]["WELL_199"]
    productivity_197 = plasts["PAD_34"]["WELL_197"]
    productivity_195 = plasts["PAD_34"]["WELL_195"]
    productivity_191 = plasts["PAD_34"]["WELL_191"]
    productivity_729 = plasts["PAD_34"]["WELL_729"]
    productivity_730 = plasts["PAD_34"]["WELL_730"]
    productivity_192 = plasts["PAD_34"]["WELL_192"]
    productivity_148 = plasts["PAD_34"]["WELL_148"]

    # Пласты скважин
    # Куст 33
    G.add_edge('PAD_33_well_1385', 'Zaboi_1385', obj=HE2_Plast(productivity=productivity_1385, fluid=fluid))
    G.add_edge('PAD_33_well_736', 'Zaboi_736', obj=HE2_Plast(productivity=productivity_736, fluid=fluid))
    G.add_edge('PAD_33_well_739', 'Zaboi_739', obj=HE2_Plast(productivity=productivity_739, fluid=fluid))
    G.add_edge('PAD_33_well_1383', 'Zaboi_1383', obj=HE2_Plast(productivity=productivity_1383, fluid=fluid))
    G.add_edge('PAD_33_well_738', 'Zaboi_738', obj=HE2_Plast(productivity=productivity_738, fluid=fluid))
    G.add_edge('PAD_33_well_725', 'Zaboi_725', obj=HE2_Plast(productivity=productivity_725, fluid=fluid))

    # Куст 34
    G.add_edge('PAD_34_well_731', 'Zaboi_731', obj=HE2_Plast(productivity=productivity_731, fluid=fluid))
    G.add_edge('PAD_34_well_196', 'Zaboi_196', obj=HE2_Plast(productivity=productivity_196, fluid=fluid))
    G.add_edge('PAD_34_well_734', 'Zaboi_734', obj=HE2_Plast(productivity=productivity_734, fluid=fluid))
    G.add_edge('PAD_34_well_198', 'Zaboi_198', obj=HE2_Plast(productivity=productivity_198, fluid=fluid))
    G.add_edge('PAD_34_well_199', 'Zaboi_199', obj=HE2_Plast(productivity=productivity_199, fluid=fluid))
    G.add_edge('PAD_34_well_197', 'Zaboi_197', obj=HE2_Plast(productivity=productivity_197, fluid=fluid))
    G.add_edge('PAD_34_well_195', 'Zaboi_195', obj=HE2_Plast(productivity=productivity_195, fluid=fluid))
    G.add_edge('PAD_34_well_191', 'Zaboi_191', obj=HE2_Plast(productivity=productivity_191, fluid=fluid))
    G.add_edge('PAD_34_well_729', 'Zaboi_729', obj=HE2_Plast(productivity=productivity_729, fluid=fluid))
    G.add_edge('PAD_34_well_730', 'Zaboi_730', obj=HE2_Plast(productivity=productivity_730, fluid=fluid))
    G.add_edge('PAD_34_well_192', 'Zaboi_192', obj=HE2_Plast(productivity=productivity_192, fluid=fluid))
    G.add_edge('PAD_34_well_148', 'Zaboi_148', obj=HE2_Plast(productivity=productivity_148, fluid=fluid))

    # Насосы скважин
    # Куст 33
    pump_1385 = pumps["PAD_33"]["WELL_1385"]
    pump_736 = pumps["PAD_33"]["WELL_736"]
    pump_739 = pumps["PAD_33"]["WELL_739"]
    pump_1383 = pumps["PAD_33"]["WELL_1383"]
    pump_738 = pumps["PAD_33"]["WELL_738"]
    pump_725 = pumps["PAD_33"]["WELL_725"]

    # Куст 34
    pump_731 = pumps["PAD_34"]["WELL_731"]
    pump_196 = pumps["PAD_34"]["WELL_196"]
    pump_734 = pumps["PAD_34"]["WELL_734"]
    pump_198 = pumps["PAD_34"]["WELL_198"]
    pump_199 = pumps["PAD_34"]["WELL_199"]
    pump_197 = pumps["PAD_34"]["WELL_197"]
    pump_195 = pumps["PAD_34"]["WELL_195"]
    pump_191 = pumps["PAD_34"]["WELL_191"]
    pump_729 = pumps["PAD_34"]["WELL_729"]
    pump_730 = pumps["PAD_34"]["WELL_730"]
    pump_192 = pumps["PAD_34"]["WELL_192"]
    pump_148 = pumps["PAD_34"]["WELL_148"]

    # Куст 33
    G.add_edge('Pump_intake_1385', 'Pump_outlet_1385',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_1385[0], fluid=fluid,
                                frequency=pump_1385[1]))
    G.add_edge('Pump_intake_736', 'Pump_outlet_736',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_736[0], fluid=fluid,
                                frequency=pump_736[1]))
    G.add_edge('Pump_intake_739', 'Pump_outlet_739',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_739[0], fluid=fluid,
                                frequency=pump_739[1]))
    G.add_edge('Pump_intake_1383', 'Pump_outlet_1383',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_1383[0], fluid=fluid,
                                frequency=pump_1383[1]))
    G.add_edge('Pump_intake_738', 'Pump_outlet_738',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_738[0], fluid=fluid,
                                frequency=pump_738[1]))
    G.add_edge('Pump_intake_725', 'Pump_outlet_725',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_725[0], fluid=fluid,
                                frequency=pump_725[1]))

    # Куст 34
    G.add_edge('Pump_intake_731', 'Pump_outlet_731',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_731[0], fluid=fluid,
                                frequency=pump_731[1]))
    G.add_edge('Pump_intake_196', 'Pump_outlet_196',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_196[0], fluid=fluid,
                                frequency=pump_196[1]))
    G.add_edge('Pump_intake_734', 'Pump_outlet_734',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_734[0], fluid=fluid,
                                frequency=pump_734[1]))
    G.add_edge('Pump_intake_198', 'Pump_outlet_198',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_198[0], fluid=fluid,
                                frequency=pump_198[1]))
    G.add_edge('Pump_intake_199', 'Pump_outlet_199',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_199[0], fluid=fluid,
                                frequency=pump_199[1]))
    G.add_edge('Pump_intake_197', 'Pump_outlet_197',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_197[0], fluid=fluid,
                                frequency=pump_197[1]))
    G.add_edge('Pump_intake_195', 'Pump_outlet_195',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_195[0], fluid=fluid,
                                frequency=pump_195[1]))
    G.add_edge('Pump_intake_191', 'Pump_outlet_191',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_191[0], fluid=fluid,
                                frequency=pump_191[1]))
    G.add_edge('Pump_intake_729', 'Pump_outlet_729',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_729[0], fluid=fluid,
                                frequency=pump_729[1]))
    G.add_edge('Pump_intake_730', 'Pump_outlet_730',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_730[0], fluid=fluid,
                                frequency=pump_730[1]))
    G.add_edge('Pump_intake_192', 'Pump_outlet_192',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_192[0], fluid=fluid,
                                frequency=pump_192[1]))
    G.add_edge('Pump_intake_148', 'Pump_outlet_148',
               obj=create_HE2_WellPump_instance_from_dataframe(full_HPX=pump_curves, model=pump_148[0], fluid=fluid,
                                frequency=pump_148[1]))

    # Инклинометрия скважин
    # Куст 33
    G.add_edge('Pump_outlet_1385', 'Wellhead_1385',
               obj=HE2_OilPipe([395.8], [2139.2], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_1385', 'Pump_intake_1385',
               obj=HE2_OilPipe([80], [584], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_736', 'Wellhead_736',
               obj=HE2_OilPipe([216], [2532], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_736', 'Pump_intake_736',
               obj=HE2_OilPipe([448], [155], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_739', 'Wellhead_739',
               obj=HE2_OilPipe([323.45], [2356.55], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_739', 'Pump_intake_739',
               obj=HE2_OilPipe([396.35], [290.6], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_1383', 'Wellhead_1383',
               obj=HE2_OilPipe([526], [2353.99], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_1383', 'Pump_intake_1383',
               obj=HE2_OilPipe([30.16], [377.84], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_738', 'Wellhead_738',
               obj=HE2_OilPipe([243.83], [2128.17], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_738', 'Pump_intake_738',
               obj=HE2_OilPipe([402.45], [526.85], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_725', 'Wellhead_725',
               obj=HE2_OilPipe([495.76], [2329.24], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_725', 'Pump_intake_725',
               obj=HE2_OilPipe([28.7], [384.3], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Wellhead_1385', 'PAD_33', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_736', 'PAD_33', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_739', 'PAD_33', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_1383', 'PAD_33', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_738', 'PAD_33', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_725', 'PAD_33', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))

    # Куст 34
    G.add_edge('Pump_outlet_731', 'Wellhead_731',
               obj=HE2_OilPipe([285.7], [2414.27], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_731', 'Pump_intake_731',
               obj=HE2_OilPipe([13.5], [330], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_196', 'Wellhead_196',
               obj=HE2_OilPipe([172.6], [2504.43], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_196', 'Pump_intake_196',
               obj=HE2_OilPipe([7.38], [245.12], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_734', 'Wellhead_734',
               obj=HE2_OilPipe([32.21], [2362.8], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_734', 'Pump_intake_734',
               obj=HE2_OilPipe([282], [286.88], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_198', 'Wellhead_198',
               obj=HE2_OilPipe([244], [2586], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_198', 'Pump_intake_198',
               obj=HE2_OilPipe([4.82], [169.18], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_199', 'Wellhead_199',
               obj=HE2_OilPipe([228.11], [2520.89], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_199', 'Pump_intake_199',
               obj=HE2_OilPipe([198.34], [241.56], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_197', 'Wellhead_197',
               obj=HE2_OilPipe([668.11], [2345.89], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_197', 'Pump_intake_197',
               obj=HE2_OilPipe([52.91], [403.09], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_195', 'Wellhead_195',
               obj=HE2_OilPipe([610.82], [2372.96], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_195', 'Pump_intake_195',
               obj=HE2_OilPipe([50.06], [357.93], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_191', 'Wellhead_191',
               obj=HE2_OilPipe([254.37], [2585.63], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_191', 'Pump_intake_191',
               obj=HE2_OilPipe([251.19], [166.221], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_729', 'Wellhead_729',
               obj=HE2_OilPipe([92.78], [2345.22], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_729', 'Pump_intake_729',
               obj=HE2_OilPipe([247.95], [399.048], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_730', 'Wellhead_730',
               obj=HE2_OilPipe([75.76], [2218.24], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_730', 'Pump_intake_730',
               obj=HE2_OilPipe([99.72], [532.78], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_192', 'Wellhead_192',
               obj=HE2_OilPipe([337.39], [2177.61], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_192', 'Pump_intake_192',
               obj=HE2_OilPipe([255.33], [531.87], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Pump_outlet_148', 'Wellhead_148',
               obj=HE2_OilPipe([337.39], [2177.61], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Zaboi_148', 'Pump_intake_148',
               obj=HE2_OilPipe([255.33], [531.87], [0.143 * real_diam_coefficient], [5 * roughness]))

    G.add_edge('Wellhead_731', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_196', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_734', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_198', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_199', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_197', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_195', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_191', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_729', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_730', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_192', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))
    G.add_edge('Wellhead_148', 'PAD_34', obj=HE2_OilPipe([100], [0], [0.125 * real_diam_coefficient], [roughness]))

    # Трубопроводная система
    G.add_edge('PAD_5', 'intake_pad_5', obj=HE2_OilPipe([10], [-0.2], [0.143 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_5', 'intake_pad_9_36',
               obj=HE2_OilPipe([785], [-0.6], [0.199 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_9_36', 'intake_pad_35',
               obj=HE2_OilPipe([2326], [4.6], [0.199 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_35', 'UDR_1', obj=HE2_OilPipe([2648], [13.6], [0.199 * real_diam_coefficient], [roughness]))
    G.add_edge('UDR_1', 'DNS_2', obj=HE2_OilPipe([200], [0.2], [0.305 * real_diam_coefficient], [roughness]))
    G.add_edge('PAD_33', 'intake_pad_33', obj=HE2_OilPipe([440], [-1.9], [0.139 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_33', 'UDR_2', obj=HE2_OilPipe([4394], [-4.6], [0.199 * real_diam_coefficient], [roughness]))
    G.add_edge('UDR_2', 'UDR_1', obj=HE2_OilPipe([1087], [3.4], [0.253 * real_diam_coefficient], [roughness]))
    G.add_edge('PAD_34', 'intake_pad_134', obj=HE2_OilPipe([818], [0.6], [0.143 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_134', 'UDR_2', obj=HE2_OilPipe([3344], [10.2], [0.203 * real_diam_coefficient], [roughness]))
    G.add_edge('PAD_39', 'intake_pad_59', obj=HE2_OilPipe([7568], [1.6], [0.139 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_59', 'intake_pad_59_46',
               obj=HE2_OilPipe([4601], [4.4], [0.143 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_59_46', 'intake_pad_47',
               obj=HE2_OilPipe([2625], [18.4], [0.139 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_47', 'intake_pad_53',
               obj=HE2_OilPipe([2250], [-1.3], [0.203 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_53', 'intake_pad_41',
               obj=HE2_OilPipe([1391], [9], [0.203 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_41', 'ZKL_98',
               obj=HE2_OilPipe([9290], [-13.5], [0.257 * real_diam_coefficient], [roughness]))
    G.add_edge('ZKL_98', 'intake_pad_49', obj=HE2_OilPipe([600], [3], [0.257 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_49', 'intake_node_33',
               obj=HE2_OilPipe([4001], [0], [0.410 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_node_33', 'intake_pad_46',
               obj=HE2_OilPipe([2920], [2.1], [0.410 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_46', 'node_15',
               obj=HE2_OilPipe([1181], [-5.9], [0.410 * real_diam_coefficient], [roughness]))
    G.add_edge('node_15', 'UDR_2', obj=HE2_OilPipe([92], [0.2], [0.199 * real_diam_coefficient], [roughness]))
    G.add_edge('PAD_49', 'intake_node_49', obj=HE2_OilPipe([492], [-0.8], [0.139 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_node_49', 'intake_pad_57',
               obj=HE2_OilPipe([3737], [-5.7], [0.143 * real_diam_coefficient], [roughness]))
    G.add_edge('intake_pad_57', 'intake_pad_49',
               obj=HE2_OilPipe([3852], [-4.3], [0.309 * real_diam_coefficient], [roughness]))
    G.add_edge('PAD_57', 'intake_pad_57', obj=HE2_OilPipe([370], [-1], [0.203 * real_diam_coefficient], [roughness]))

    return G, inlets, juncs, outlets
