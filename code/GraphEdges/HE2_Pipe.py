from Tools import HE2_ABC as abc
from Fluids.HE2_Fluid import HE2_DummyWater, HE2_BlackOil, gimme_dummy_BlackOil
import uniflocpy.uTools.uconst as uc
import numpy as np

import Hydraulics.Methodics.Mukherjee_Brill as mb
from functools import lru_cache
from Tools.HE2_Logger import check_for_nan, getLogger
logger = getLogger(__name__)


class HE2_WaterPipeSegment():
    '''
    Чтобы не запутаться в будущем.
    PipeSegment не должен быть ребром графа и не обязан поддерживать интерфейс HE2_ABC_GraphEdge
    Но поскольку сгемент трубы все равно является трубой, только лишь простой, то есть потребность считать ее в обе стороны.
    Поэтому она умеет считать в обе стороны, но интерфейс для этого отличается, здесь используется calc_direction in [-1,+1]
    '''
    def __init__(self, fluid=None, inner_diam_m=None, roughness_m=None, L_m=None, uphill_m=None):
        fluid = HE2_DummyWater()
        self.fluid = fluid
        self.inner_diam_m = inner_diam_m
        self.roughness_m = roughness_m
        self.L_m = None
        self.uphill_m = None
        self.angle_dgr = None
        self.dx_m = None
        self.set_pipe_geometry(L=L_m, dy=uphill_m)

    def set_pipe_geometry(self, dx=None, dy=None, L=None, angle=None):
        if dx is not None and dy is not None:
            L = (dx*dx + dy*dy) ** 0.5
        elif L is not None and angle is not None:
            dy = L * np.sin(uc.grad2rad(angle))
        elif L is not None and dx is not None:
            logger.error(f'This way to set up pipe geometry is undefined, L={L}, dx={dx}')
            raise ValueError('Undefined sign of dY')
        elif L is not None and dy is not None:
            pass
        elif dx is not None and angle is not None:
            if dx <= 0:
                logger.error(f'This way to set up pipe geometry is undefined, angle={angle}, dx={dx}')
                raise ValueError('')
            dy = dx * np.tan(uc.grad2rad(angle))
            L = (dx * dx + dy * dy) ** 0.5
        elif dy is not None and angle is not None:
            L = abs(dy / np.sin(uc.grad2rad(angle)))
        else:
            return

        angle_dgr = uc.rad2grad(np.arcsin(dy / L))
        dx_m = (L * L - dy * dy) ** 0.5
        self.L_m = L
        self.uphill_m = dy
        self.angle_dgr = angle_dgr
        self.dx_m = dx_m
        check_for_nan(L_m = L, uphill_m = dy, angle_dgr = angle_dgr, dx_m = dx_m)

    @lru_cache(maxsize=None)
    def decode_direction(self, flow, calc_direction, unifloc_direction):
        '''
        :param unifloc_direction - направление расчета и потока относительно  координат.
            11 расчет и поток по координате
            10 расчет по координате, поток против
            00 расчет и поток против координаты
            00 расчет против координаты, поток по координате
            unifloc_direction перекрывает переданные flow, calc_direction
        '''
        # flow_direction = np.sign(flow)
        if unifloc_direction in [0, 1, 10, 11]:
            calc_direction = 1 if unifloc_direction >= 10 else -1
            flow_direction = 1 if unifloc_direction % 10 == 1 else - 1
        else:
            if flow < 0:
                flow_direction = -1
            elif flow > 0:
                flow_direction = 1
            else:
                flow_direction = 0

        # assert calc_direction in [-1, 1]
        grav_sign = calc_direction
        fric_sign = flow_direction * calc_direction
        t_sign = calc_direction
        return grav_sign, fric_sign, t_sign

    def calc_P_friction_gradient_Pam(self, P_bar, T_C, X_kgsec):
        if X_kgsec < 0:
            raise ValueError(f'X_kgsec = {X_kgsec}')
        if X_kgsec == 0:
            return 0
        # Fluid.calc will be optimized at lower level. So we will call it every time
        self.fluid.calc(P_bar, T_C)
        Rho_kgm3 = self.fluid.rho_wat_kgm3
        mu_pasec = uc.cP2pasec(self.fluid.mu_wat_cp) # dynamic viscocity
        Q_m3sec = X_kgsec / Rho_kgm3
        D_m = self.inner_diam_m
        Area_m2 = uc.pi*D_m**2/4
        V_msec = Q_m3sec / Area_m2
        Re = Rho_kgm3 * V_msec * D_m / mu_pasec
        k_m = self.roughness_m
        if Re < 2300:
            lambda_fr = 68/ Re
        else:
            lambda_fr = 0.11 * (k_m/D_m + 68.5/Re) ** 0.25
        P_fric_grad_Pam = 0.5 * lambda_fr * V_msec**2 * Rho_kgm3 / D_m
        return P_fric_grad_Pam

    def calc_T_gradient_Cm(self, P_bar, T_C, X_kgsec):
        return 0

    def calc_segment_pressure_drop_slow(self, P_bar, T_C, X_kgsec, calc_direction, unifloc_direction=-1):
        P_fric_grad_Pam = self.calc_P_friction_gradient_Pam(P_bar, T_C, abs(X_kgsec))
        dP_fric_Pa = P_fric_grad_Pam * self.L_m
        Rho_kgm3 = self.fluid.rho_wat_kgm3
        dP_gravity_Pa = Rho_kgm3 * uc.g * self.uphill_m
        grav_sign, fric_sign, t_sign = self.decode_direction(X_kgsec, calc_direction, unifloc_direction)
        P_drop_bar = uc.Pa2bar(grav_sign * dP_gravity_Pa + fric_sign * dP_fric_Pa)
        P_rez_bar = P_bar - P_drop_bar
        T_grad_Cm = self.calc_T_gradient_Cm(P_bar, T_C, X_kgsec)
        T_rez_C = T_C - t_sign * T_grad_Cm * self.L_m
        return P_rez_bar, T_rez_C


    def calc_segment_pressure_drop(self, P_bar, T_C, X_kgsec, calc_direction, unifloc_direction=-1):
        check_for_nan(P_bar=P_bar, T_C=T_C, X_kgsec=X_kgsec)
        grav_sign, fric_sign, t_sign = self.decode_direction(X_kgsec, calc_direction, unifloc_direction)
        dP_fric_Pa = self.calc_P_friction_gradient_Pam(P_bar, T_C, abs(X_kgsec)) * self.L_m
        P_rez_bar = P_bar - uc.Pa2bar(grav_sign * self.fluid.rho_wat_kgm3 * uc.g * self.uphill_m + fric_sign * dP_fric_Pa)
        check_for_nan(P_fric_grad_Pam=dP_fric_Pa)
        return P_rez_bar, 20


class HE2_WaterPipe(abc.HE2_ABC_GraphEdge):
    def __init__(self, dxs, dys, diams, rghs):
        self.segments = []
        self.intermediate_results = []
        self._printstr = ';\n '.join([' '.join([f'{itm:.2f}' for itm in vec]) for vec in [dxs, dys, diams, rghs]])
        for dx, dy, diam, rgh in zip(dxs, dys, diams, rghs):
            seg = HE2_WaterPipeSegment(None, diam, rgh)
            seg.set_pipe_geometry(dx, dy)
            self.segments += [seg]

    def __str__(self):
        return self._printstr

    def perform_calc(self, P_bar, T_C, X_kgsec, unifloc_direction):
        assert unifloc_direction in [0, 1, 10, 11]
        calc_direction = 1 if unifloc_direction >= 10 else -1
        flow_direction = 1 if unifloc_direction % 10 == 1 else - 1
        if calc_direction == 1:
            return self.perform_calc_forward(P_bar, T_C, flow_direction * abs(X_kgsec))
        else:
            return self.perform_calc_backward(P_bar, T_C, flow_direction * abs(X_kgsec))

    def perform_calc_forward(self, P_bar, T_C, X_kgsec):
        p, t = P_bar, T_C
        self.intermediate_results = []
        for seg in self.segments:
            p, t = seg.calc_segment_pressure_drop(p, t, X_kgsec, 1)
            self.intermediate_results += [(p, t)]
        return p, t

    def perform_calc_backward(self, P_bar, T_C, X_kgsec):
        p, t = P_bar, T_C
        self.intermediate_results = []
        for seg in self.segments[::-1]:
            p, t = seg.calc_segment_pressure_drop(p, t, X_kgsec, -1)
            self.intermediate_results += [(p, t)]
        return p, t



class HE2_OilPipeSegment():
    '''
    Аналог HE2_WaterPipeSegment с реюзом Mishenko и Mukherjee_Brill
    '''
    def __init__(self, fluid:HE2_BlackOil=None, inner_diam_m=None, roughness_m=None, L_m=None, uphill_m=None):
        if fluid is None:
            fluid = gimme_dummy_BlackOil()
        self.fluid = fluid

        self.inner_diam_m = inner_diam_m
        self.roughness_m = roughness_m
        self.L_m = L_m
        self.uphill_m = uphill_m
        self.angle_dgr = 90
        self.dx_m = None
        self.set_pipe_geometry(L=L_m, dy=uphill_m)

    def set_pipe_geometry(self, dx=None, dy=None, L=None, angle=None):
        if dx is not None and dy is not None:
            L = (dx*dx + dy*dy) ** 0.5
        elif L is not None and angle is not None:
            dy = L * np.sin(uc.grad2rad(angle))
        elif L is not None and dx is not None:
            logger.error(f'This way to set up pipe geometry is undefined, L={L}, dx={dx}')
            raise ValueError('Undefined sign of dY')
        elif L is not None and dy is not None:
            pass
        elif dx is not None and angle is not None:
            if dx <= 0:
                logger.error(f'This way to set up pipe geometry is undefined, angle={angle}, dx={dx}')
                raise ValueError()
            dy = dx * np.tan(uc.grad2rad(angle))
            L = (dx * dx + dy * dy) ** 0.5
        elif dy is not None and angle is not None:
            L = abs(dy / np.sin(uc.grad2rad(angle)))
        else:
            return

        angle_dgr = uc.rad2grad(np.arcsin(dy / L))
        dx_m = (L * L - dy * dy) ** 0.5
        self.L_m = L
        self.uphill_m = dy
        self.angle_dgr = angle_dgr
        self.dx_m = dx_m
        check_for_nan(L_m = L, uphill_m = dy, angle_dgr = angle_dgr, dx_m = dx_m)

    def decode_direction(self, flow, calc_direction, unifloc_direction):
        '''
        :param unifloc_direction - направление расчета и потока относительно  координат.
            11 расчет и поток по координате
            10 расчет по координате, поток против
            00 расчет и поток против координаты
            00 расчет против координаты, поток по координате
            unifloc_direction перекрывает переданные flow, calc_direction
        '''
        flow_direction = np.sign(flow)
        if unifloc_direction in [0, 1, 10, 11]:
            calc_direction = 1 if unifloc_direction >= 10 else -1
            flow_direction = 1 if unifloc_direction % 10 == 1 else - 1

        assert calc_direction in [-1, 1]
        grav_sign = calc_direction
        fric_sign = flow_direction * calc_direction
        t_sign = calc_direction
        return grav_sign, fric_sign, t_sign

    def calc_P_friction_gradient_Pam(self, P_bar, T_C, X_kgsec, fric_sign, grav_sign, calc_direction):
        #Уточнить про X_kgsec
        if X_kgsec == 0:
            return 0, self.fluid.calc(P_bar, T_C, 1, self.inner_diam_m).CurrentLiquidDensity_kg_m3 * 9.81
        # Fluid.calc will be optimized at lower level. So we will call it every time
        current_mishenko = self.fluid.calc(P_bar, T_C,abs(X_kgsec), self.inner_diam_m)
        #Определяем угол в зависимости от fric_sign
        angle = self.angle_dgr if calc_direction > 0 else -self.angle_dgr
        P_fric_grad_Pam, P_grav_grad_Pam = mb.calculate(current_mishenko, angle, self.inner_diam_m, self.roughness_m)
        return P_fric_grad_Pam, P_grav_grad_Pam

    def calc_T_gradient_Cm(self, P_bar, T_C, X_kgsec):
        return 0

    def calc_segment_pressure_drop(self, P_bar, T_C, X_kgsec, calc_direction, unifloc_direction=-1):
        check_for_nan(P_bar=P_bar, T_C=T_C, X_kgsec=X_kgsec)
        #Определяем направления расчета
        grav_sign, fric_sign, t_sign = self.decode_direction(X_kgsec, calc_direction, unifloc_direction)
        #Считаем локальный градиент давления по M_B
        P_fric_grad_Pam, P_grav_grad_Pam = self.calc_P_friction_gradient_Pam(P_bar, T_C, abs(X_kgsec), fric_sign, grav_sign, calc_direction)
        #Считаем потери давления на сегменте
        dP_full_Pa = P_fric_grad_Pam * self.L_m
        dP_full_grav_Pa = P_grav_grad_Pam * self.uphill_m
        #Считаем полные потери давления по сегменту
        P_drop_bar = fric_sign * uc.Pa2bar(dP_full_Pa) + grav_sign * uc.Pa2bar(dP_full_grav_Pa)
        P_rez_bar = P_bar - P_drop_bar #temp solution
        T_grad_Cm = self.calc_T_gradient_Cm(P_bar, T_C, X_kgsec)
        T_rez_C = T_C - t_sign * T_grad_Cm * self.L_m
        check_for_nan(P_rez_bar = P_rez_bar, T_rez_C = T_rez_C)
        return P_rez_bar, T_rez_C

class HE2_OilPipe(abc.HE2_ABC_GraphEdge):
    def __init__(self, dxs, dys, diams, rghs, fluid=None):
        self.segments = []
        self.intermediate_results = []
        self._printstr = ';\n '.join([' '.join([f'{itm:.2f}' for itm in vec]) for vec in [dxs, dys, diams, rghs]])
        if fluid is None:
            fluid = gimme_dummy_BlackOil()
        self.fluid = fluid
        for dx, dy, diam, rgh in zip(dxs, dys, diams, rghs):
            seg = HE2_OilPipeSegment(inner_diam_m=diam, roughness_m=rgh, L_m=None, uphill_m=None)
            seg.fluid.oil_params = self.fluid.oil_params
            seg.set_pipe_geometry(dx=dx, dy=dy)
            a = seg
            self.segments += [seg]

    def __str__(self):
        return self._printstr

    def perform_calc(self, P_bar, T_C, X_kgsec, unifloc_direction):
        assert unifloc_direction in [0, 1, 10, 11]
        calc_direction = 1 if unifloc_direction >= 10 else -1
        flow_direction = 1 if unifloc_direction % 10 == 1 else - 1
        if calc_direction == 1:
            return self.perform_calc_forward(P_bar, T_C, flow_direction * abs(X_kgsec)) #!
        else:
            return self.perform_calc_backward(P_bar, T_C, flow_direction * abs(X_kgsec))

    def perform_calc_forward(self, P_bar, T_C, X_kgsec):
        p, t = P_bar, T_C
        self.intermediate_results = []
        for seg in self.segments:
            seg.fluid.oil_params = self.fluid.oil_params
            p, t = seg.calc_segment_pressure_drop(p, t, X_kgsec, 1)
            self.intermediate_results += [(p, t)]
        return p, t

    def perform_calc_backward(self, P_bar, T_C, X_kgsec):
        p, t = P_bar, T_C
        self.intermediate_results = []
        for seg in self.segments[::-1]:
            seg.fluid.oil_params = self.fluid.oil_params
            p, t = seg.calc_segment_pressure_drop(p, t, X_kgsec, -1)
            self.intermediate_results += [(p, t)]
        return p, t

