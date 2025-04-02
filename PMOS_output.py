from pylab import *
import numpy as np
import scipy
from scipy.linalg import null_space
from scipy import integrate
from matplotlib import rcParams
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

def Fermi(x):
		return 1.0/(exp(x)+1)

def Bose(x):
		return 1.0/(exp(x)-1)

def PMOS(V_DD,Vin,R_d):
	A=zeros((2,2))
	Gamma_l=0.2
	Gamma_r=0.2
	Gamma=0.2
	Gamma_g=0.2

	J_d = 0.0
	mu_l=0.0
	kBT=1.0

	tint=10
	T=1000
	Ntot=int(T/tint)


	for i in range(Ntot):
		V_D = V_DD - J_d * R_d
		E_P = Vin
		mu_r = mu_l - V_D

		k_Pl = Gamma_l * Fermi((E_P - mu_l) / kBT)
		k_lP = Gamma_l * (1.0 - Fermi((E_P - mu_l) / kBT))
		k_rP = Gamma_r * (1.0 - Fermi((E_P - mu_r) / kBT))
		k_Pr = Gamma_r * Fermi((E_P - mu_r) / kBT)

		A[1][0] = k_Pr + k_Pl
		A[0][0] = -A[1][0]
		A[0][1] = k_rP + k_lP
		A[1][1] = -A[0][1]

		p = null_space(A)
		sum = p[0][0] + p[1][0]
		p[0][0] = p[0][0] / sum
		p[1][0] = p[1][0] / sum
		p_N = p[1][0]
		J_d = -(k_Pr * (1 - p_N) - k_rP * p_N)

	return J_d, V_D

V_DD_max = 21
V_DD = np.zeros(V_DD_max*10)
J_d1 = np.zeros(V_DD_max*10)
J_d2 = np.zeros(V_DD_max*10)
J_d3 = np.zeros(V_DD_max*10)
J_d4 = np.zeros(V_DD_max*10)
J_d5 = np.zeros(V_DD_max*10)
V_D1 = np.zeros(V_DD_max*10)
V_D2 = np.zeros(V_DD_max*10)
V_D3 = np.zeros(V_DD_max*10)
V_D4 = np.zeros(V_DD_max*10)
V_D5 = np.zeros(V_DD_max*10)

i = 0
for V in np.arange(0,V_DD_max,0.1):
	V_DD[i] = V
	J_d1[i], V_D1[i] = PMOS(V_DD[i], -1.0, 0.0)
	J_d2[i], V_D2[i] = PMOS(V_DD[i], 0.0, 0.0)
	J_d3[i], V_D3[i] = PMOS(V_DD[i], 1.0, 0.0)
	J_d4[i], V_D4[i] = PMOS(V_DD[i], 3.0, 0.0)
	J_d5[i], V_D5[i] = PMOS(V_DD[i], 5.0, 0.0)
	i = i + 1

with open('/results/PMOS_output.txt', 'w') as f:
    f.write("J_d1\tJ_d2\tJ_d3\tJ_d4\tJ_d5\n")
    for i in range(len(V_DD)):
        f.write(f"{J_d1[i]:.6f}\t{J_d2[i]:.6f}\t{J_d3[i]:.6f}\t{J_d4[i]:.6f}\t{J_d5[i]:.6f}\n")
