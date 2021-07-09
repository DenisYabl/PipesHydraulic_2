from Tools import HE2_ABC as abc
from Fluids.HE2_Fluid import gimme_dummy_BlackOil
import numpy as np
from Tools.HE2_Logger import check_for_nan, getLogger
logger = getLogger(__name__)

class HE2_Plast(abc.HE2_ABC_GraphEdge):
    def __init__(self, productivity = 0, fluid=None):
        if productivity <= 0:
            raise ValueError(f'Productivity = {productivity}')
        self.Productivity = productivity
        self.fluid = fluid
        self._printstr = f"Productivity coefficient: {self.Productivity}"


    def __str__(self):
        return self._printstr

    def perform_calc(self, P_bar, T_C, X_kgsec, unifloc_direction):
        assert unifloc_direction in [0, 1, 10, 11]
        calc_direction = 1 if unifloc_direction >= 10 else -1
        flow_direction = 1 if unifloc_direction % 10 == 1 else - 1
        if calc_direction == 1:
            return self.perform_calc_forward(P_bar, T_C,  X_kgsec)
        else:
            return self.perform_calc_backward(P_bar, T_C, X_kgsec)

    def perform_calc_forward(self, P_bar, T_C, X_kgsec):
        p, t = P_bar, T_C
        p, t = self.calculate_pressure_differrence(p, t, X_kgsec, 1)
        return p, t

    def perform_calc_backward(self, P_bar, T_C, X_kgsec):
        p, t = P_bar, T_C
        p, t = self.calculate_pressure_differrence(p, t, X_kgsec, -1)
        return p, t

    def calculate_pressure_differrence(self, P_bar, T_C, X_kgsec, calc_direction, unifloc_direction=-1):
        check_for_nan(P_bar=P_bar, T_C=T_C, X_kgsec=X_kgsec)
        #Определяем направления расчета
        fric_sign, t_sign = self.decode_direction(X_kgsec, calc_direction, unifloc_direction)

        fl =  self.fluid
        liq = fl.calc(P_bar, T_C, X_kgsec)
        liq_dens = liq.CurrentLiquidDensity_kg_m3
        P_rez_bar = P_bar - calc_direction * (X_kgsec * 86400 /  liq_dens) / self.Productivity
        T_rez_C = T_C
        check_for_nan(P_rez_bar=P_rez_bar, T_rez_C=T_rez_C)
        return P_rez_bar, T_rez_C

    def decode_direction(self, flow, calc_direction, unifloc_direction):
        '''
        :param unifloc_direction - направление расчета и потока относительно  координат.
            11 расчет и поток по координате
            10 расчет по координате, поток против
            00 расчет и поток против координаты
            01 расчет против координаты, поток по координате
            unifloc_direction перекрывает переданные flow, calc_direction
            grav_sign не нужен, поскольку он учитывается в Mukherjee_Brill
        '''
        flow_direction = np.sign(flow)
        if unifloc_direction in [0, 1, 10, 11]:
            calc_direction = 1 if unifloc_direction >= 10 else -1
            flow_direction = 1 if unifloc_direction % 10 == 1 else - 1

        assert calc_direction in [-1, 1]
        fric_sign = flow_direction * calc_direction
        t_sign = calc_direction
        return fric_sign, t_sign