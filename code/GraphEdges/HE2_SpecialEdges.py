from Tools import HE2_ABC as abc
from Fluids.HE2_Fluid import gimme_dummy_BlackOil


class HE2_MockEdge(abc.HE2_ABC_GraphEdge):
    def __init__(self, delta_P=0, fluid=None):
        self.dP = delta_P
        if fluid is None:
            fluid = gimme_dummy_BlackOil()
        self.fluid = fluid

    def perform_calc(self, P_bar, T_C, X_kgsec, unifloc_direction):
        #TODO имплементировать разбор направления по юнифлоку
        return P_bar + self.dP, T_C

    def perform_calc_forward(self, P_bar, T_C, X_kgsec):
        return P_bar + self.dP, T_C

    def perform_calc_backward(self, P_bar, T_C, X_kgsec):
        return P_bar - self.dP, T_C
