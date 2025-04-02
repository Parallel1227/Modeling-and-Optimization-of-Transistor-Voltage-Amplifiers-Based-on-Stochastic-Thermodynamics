from pylab import *
import numpy as np
import scipy
from scipy.linalg import null_space
from scipy import integrate
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def Fermi(x):
    return 1.0 / (exp(x) + 1)


def Bose(x):
    if (x > 1e-4):
        return 1.0 / (exp(x) - 1)
    else:
        return 1e5


def Gauss_in0(x, a):
    sigma2 = 0.005
    mu = a
    return 1 / sqrt(2 * np.pi * sigma2) * exp(-1.0 / 2 / sigma2 * (x - mu) * (x - mu))


def PA(V_D, Vin, Gamma_load, gamma):
    A = zeros((4, 4))
    c = zeros(4)

    # gamma = 0.0188
    Gamma_l = gamma
    Gamma_r = gamma
    Gamma = gamma
    Gamma_g = gamma

    E_P = -V_D + Vin
    E_N = -Vin
    mu_r1 = -V_D
    mu_r2 = V_D
    mu_l = 0.0
    kBT = 1.0

    I_sum_before = 1000000000000.0
    Vout_before = 0.0

    for i in range(0, 5001, 1):
        Vout = i / 100 - 25.0
        mu_g = 0.0 - Vout
        k_Nl = Gamma_l * Fermi((E_N - mu_r2) / kBT)
        k_lN = Gamma_l * (1.0 - Fermi((E_N - mu_r2) / kBT))
        k_rP = Gamma_r * (1.0 - Fermi((E_P - mu_r1) / kBT))
        k_Pr = Gamma_r * Fermi((E_P - mu_r1) / kBT)
        k_Ng = Gamma_g * Fermi((E_N - mu_g) / kBT)
        k_gN = Gamma_g * (1.0 - Fermi((E_N - mu_g) / kBT))
        k_Pg = Gamma_g * Fermi((E_P - mu_g) / kBT)
        k_gP = Gamma_g * (1.0 - Fermi((E_P - mu_g) / kBT))
        if (E_N > E_P):
            k_NP = Gamma * Bose((E_N - E_P) / kBT)
            k_PN = Gamma * (1 + Bose((E_N - E_P) / kBT))
        else:
            k_PN = Gamma * Bose((E_P - E_N) / kBT)
            k_NP = Gamma * (1 + Bose((E_P - E_N) / kBT))

        A[1][0] = k_Pr + k_Pg
        A[2][0] = k_Nl + k_Ng
        A[0][0] = -A[1][0] - A[2][0]
        A[0][1] = k_rP + k_gP
        A[2][1] = k_NP
        A[3][1] = k_Nl + k_Ng
        A[1][1] = -A[0][1] - A[2][1] - A[3][1]
        A[0][2] = k_lN + k_gN
        A[1][2] = k_PN
        A[3][2] = k_Pr + k_Pg
        A[2][2] = -A[0][2] - A[1][2] - A[3][2]
        A[1][3] = k_lN + k_gN
        A[2][3] = k_rP + k_gP
        A[3][3] = -A[1][3] - A[2][3]

        p = null_space(A)
        sum = p[0][0] + p[1][0] + p[2][0] + p[3][0]
        p[0][0] = p[0][0] / sum
        p[1][0] = p[1][0] / sum
        p[2][0] = p[2][0] / sum
        p[3][0] = p[3][0] / sum
        p_N = p[2][0] + p[3][0]
        p_P = p[1][0] + p[3][0]
        J1 = k_gN * p_N - k_Ng * (1 - p_N)  # N到g
        J2 = k_gP * p_P - k_Pg * (1 - p_P)  # P到g
        J_r1P = k_rP * p_P - k_Pr * (1 - p_P)
        J_r2N = k_lN * p_N - k_Nl * (1 - p_N)
        J_lg = 0.5 * Gamma_load * (Fermi((mu_g - mu_g) / kBT) - Fermi((mu_g - mu_l) / kBT))
        # print(J1 + J2, J_lg, Vout)

        I_sum = J_lg - J1 - J2
        if abs(I_sum) > abs(I_sum_before):
            Vout = Vout_before
            break
        I_sum_before = I_sum
        Vout_before = Vout

    return Vout, J_r1P, J_r2N, J_lg


T = 40
tint = 1
Ntot = int(T / tint)
adj = 3  # 增加以减少周期数
A = 5  # 幅值
V_D = 20
Gamma_load = 0.01

diss_N = 0
diss_P = 0
# diss_out = 0

time = np.zeros(Ntot)
Vin = np.zeros(Ntot)
Vout = np.zeros(Ntot)
J_r1P = np.zeros(Ntot)
J_r2N = np.zeros(Ntot)
J_lg = np.zeros(Ntot)

gamma = 0.5
for i in range(Ntot):
    time[i] = i * tint
    Vin[i] = sin(i / adj) * A
    Vout[i], J_r1P[i], J_r2N[i], J_lg[i] = PA(V_D, Vin[i], Gamma_load, gamma)


with open('/results/CSVAC.txt', 'w') as f:
    f.write("time\tVin\tVout\tJ_r1P\tJ_r2N\n")
    for i in range(len(time)):
        f.write(f"{time[i]:.6f}\t{Vin[i]:.6f}\t{Vout[i]:.6f}\t{J_r1P[i]:.6f}\t{J_r2N[i]:.6f}\n")

