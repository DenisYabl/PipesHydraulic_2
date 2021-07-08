from numba import njit
# from scipy.interpolate import interp1d
import numpy as np


@njit(cache=True, fastmath=True)
def count_Frud(mishenko, wm, angle, IntDiameter, Roughness):
    return wm ** 2 / (mishenko.g * IntDiameter)

@njit(cache=True, fastmath=True)
def get_dens_freegas(Rgo, a1, a2, a3):
    return a1 * (Rgo - 0.0036 * (1 + a2) * (105.7 + a3 * a2)) * 1.29

@njit(cache=True, fastmath=True)
def get_coefficients(u, mu1, mu2, dens1, dens2, Fr):
    if u <= 1:
        a = 1.04 - 0.03 * u ** 2
    else:
        a = 1 + 0.01 / (u ** 2)
    b = 1.04
    # автомодельное значение критерия Фрауда
    if mu1 <= 26:
        Fra = (5 + 1.6 / mu1) / (a + dens2 / dens1)
    else:
        Fra = 130 * (1 - dens2 / dens1) / mu1
    # Коэффициент k
    if mu2 / mu1 > 0.01:
        k = 0.8 * (1 + 1.5 * (dens2 / dens1) ** 0.5) / (1 + (dens2 / dens1) ** 0.5)
    elif 0.00013 <= mu2 / mu1 <= 0.01:
        k = 0.35 + 1.4 * (mu2 / mu1) ** 0.25
    else:
        k = 0.5
    F = Fr / Fra
    c1 = 1 - np.exp(-4.4 * F ** 0.5)
    kr = k * c1
    return a, b, kr, F

@njit(cache=True, fastmath=True)
def get_flow_structure_MB(Lw, Lg, Lm, angle, IntDiameter, Roughness):
    #if angle > 0:
    x1 = np.log10(Lg) + 0.940 + 0.074 * np.sin(np.radians(angle)) - 0.855 * np.sin(np.radians(angle)) ** 2 + 3.695 * Lm

    Lw_b_s = 10 ** x1
    #else:
    x2 = 0.431 - 3.003 * Lm - 1.138 * np.sin(np.radians(angle)) * np.log10(Lw) - 0.429 * np.sin(
            np.radians(angle)) * (np.log10(Lw)) ** 2 + 1.132 * np.sin(np.radians(angle))
    Lg_b_s = 10 ** x2
    y = 1.4 - 2.694 * Lm + 0.521 * Lw ** 0.329
    Lg_s_m = 10 ** y
    z = 0.321 - 0.017 * Lg - 4.267 * np.sin(np.radians(angle)) - 2.972 * Lm - 0.033 * (
            np.log10(Lg) ** 2) ** 2 - 3.925 * np.sin(np.radians(angle)) ** 2
    Lw_s_t = 10 ** z

    if Lg > Lg_s_m:
        return "annular"

    elif angle > 30:
        if Lw > Lw_s_t:
            if Lw > Lg_b_s:
                return "slug"
            else:
                return "stratified"
        else:
            return "bubble"

    elif angle > 0:
        if Lw > Lw_b_s:
            return "bubble"
        else:
            return "slug"

    elif Lw > Lw_s_t:
        if Lg > Lg_b_s:
            return "slug"
        else:
            return "bubble"
    else:
        return "stratified"

@njit(cache=True, fastmath=True)
def get_coefs_MB(angle, IntDiameter, Roughness, flow):
    coefs = [(-0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657),
             (-1.330283, 4.808139, 4.171584, 56.262268, 0.079951, 0.504887),
             (-0.516644, 0.789805, 0.551627, 15.519214, 0.371771, 0.393952)]
    if angle >= 0:
        return coefs[0]
    elif flow == "stratified":
        return coefs[1]
    else:
        return coefs[2]

HR = np.array((0.01, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 10.00))
fr = np.array((1.00, 0.98, 1.20, 1.25, 1.30, 1.25, 1.00, 1.00))
# _interpolate_function = np.interp(x=HR, y=fr)
# _extrapolate_function = np.interp(x=HR, y=fr, left=1, right=1)

@njit(cache=True, fastmath=True)
def count_dP_MB(mishenko, angle, IntDiameter, Roughness, form, lambda0, dens_true, Ek, phi1, phi2, Lw, Lg, Lm, wm):
    if form in ['slug', 'bubble']:
        #Расчет для пузырькового течения
        dP_fric = (lambda0 * 0.5 * dens_true * wm ** 2 / IntDiameter) / (1 - Ek)
        dP_grav = (dens_true * mishenko.g) / (1 - Ek)
    elif form == 'annular':
        #Расчет для кольцевого течения
        y = mishenko.oil_params.volumeoilcoeff / phi1
        try:
            res = np.interp(x=y, xp=HR, fp=fr)
        except:
            res = 1
        lambdam = lambda0 * res
        dP_fric = (lambdam * 0.5 * dens_true * wm ** 2 / IntDiameter) / (1 - Ek)
        dP_grav = (dens_true * mishenko.g) / (1 - Ek)
    elif form == 'stratified':
        #Расчет для расслоенного течения
        # Заполненность трубы жидкостью
        C1, C2, C3, C4, C5, C6 = get_coefs_MB(angle, IntDiameter, Roughness, form)
        Nl = Lm
        Nlv = Lw
        Ngv = Lg
        try:
            Ht = np.exp((C1 + C2 * np.sin(np.radians(angle)) + C3 * np.sin(np.radians(angle))
                           ** 2 + C4 * Nl ** 2) * (Ngv ** C5 / Nlv ** C6))
        except:
            Ht = 0.5
        delta = 2 * np.cos(1 - 2 * Ht / IntDiameter) ** -1
        Dhg = IntDiameter * (2 * np.pi - (delta - np.sin(delta))) / (
                    2 * np.pi - delta + 2 * np.sin(delta / 2))
        Dhl = IntDiameter * (delta - np.sin(delta)) / (delta + 2 * np.sin(delta / 2))
        Perim = np.pi * IntDiameter
        Perimg = (1 - 0.5 * delta / np.pi) * Perim
        Perimo = Perim - Perimg
        wo = mishenko.oil_params.volumeoilcoeff * wm / phi1
        wg = mishenko.VolumeGas_fraction * wm / phi2
        Re1 = mishenko.SaturatedOilDensity_kg_m3 * wo * Dhl / mishenko.CurrentOilViscosity_Pa_s
        Re2 = mishenko.CurrentFreeGasDensity_kg_m3 * wg * Dhg / mishenko.CurrentFreeGasViscosity_Pa_s
        # Коэффициенты гидравлического сопротивления
        lambdao = 64 / Re1 if Re1 <= 2100 else 1.325 / (
            np.log(Roughness / (3.7 * IntDiameter) + 5.74 / Re1 ** 0.9)) ** 2
        lambdag = 64 / Re2 if Re2 <= 2100 else 1.325 / (
            np.log(Roughness / (3.7 * IntDiameter) + 5.74 / Re2 ** 0.9)) ** 2
        # Касательные напряжения на стенке трубы
        tauo = lambdao * mishenko.SaturatedOilDensity_kg_m3 * wo ** 2 * 0.5 / mishenko.g
        taug = lambdag * mishenko.FreeGasDensity_kg_m3 * wg ** 2 * 0.5 / mishenko.g
        dP_fric = -(tauo * Perimo + taug * Perimg) / mishenko.oil_params.gasFactor
        dP_grav = dens_true * mishenko.g
    return dP_fric, dP_grav


