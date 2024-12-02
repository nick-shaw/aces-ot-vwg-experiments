import numpy as np
import matplotlib.pyplot as plt

L_A = 100.0
Y_b = 20.0
surround_y = 0.59

# pre-calculate viewing condition dependent values
k = 1.0 / (5.0 * L_A + 1.0)
k4 = k*k*k*k
F_L = 0.2 * k4 * (5.0 * L_A) + 0.1 * np.power((1.0 - k4), 2.0) * np.power(5.0 * L_A, 1.0 / 3.0)
n = Y_b / 100.0
z = 1.48 + np.sqrt(n)
F_L_W = np.power(F_L, 0.42)
A_w = (400.0 * F_L_W) / (27.13 + F_L_W)

def Y_to_J(Y, L_A, Y_b, surround_y):
    F_L_Y = np.power(F_L * abs(Y) / 100.0, 0.42)

    return np.sign(Y) * (100.0 * np.power(((400.0 * F_L_Y) / (27.13 + F_L_Y)) / A_w, surround_y * z))

def J_to_Y(J, L_A, Y_b, surround_y):
    A = np.sign(J) * (A_w * np.power(np.abs(J) / 100.0, 1.0 / (surround_y * z)))

    return np.sign(A) * (100.0 / F_L * np.power((27.13 * np.abs(A)) / (400.0 - np.abs(A)), 1.0 / 0.42))

def peak_to_r_hit(peak):
  return 128 + 768 * (np.log10(peak / 100) / np.log10(10000 / 100))

def daniele_evo_fwd(Y, m_2, s_2, g, t_1):
    f = m_2 * np.power(np.maximum(0.0, Y) / (Y + s_2), g)
    h = np.maximum(0.0, f * f / (f + t_1))

    return h

peak = np.array([100, 225, 500, 625, 1000, 2000, 4000])

limitJmax = Y_to_J(peak,  L_A, Y_b, surround_y)

r_hit = peak_to_r_hit(peak)

J = Y_to_J(r_hit*100,  L_A, Y_b, surround_y)

table = np.dstack((peak, r_hit, limitJmax, J))

print(table)

# DanieleEvoCurve (ACES2 candidate) parameters
mmScaleFactor = 100.0      # redundant and equivalent to daniele_n_r
daniele_n_r = 100.0        # Normalized white in nits (what 1.0 should be)
daniele_g = 1.15           # surround / contrast
daniele_c = 0.18           # scene-referred grey
daniele_c_d = 10.013       # display-referred grey (in nits)
daniele_w_g = 0.14         # grey change between different peak luminance
daniele_t_1 = 0.04         # shadow toe, flare/glare compensation - how ever you want to call it
daniele_r_hit_min = 128.0  # Scene-referred value "hitting the roof" at 100 nits
daniele_r_hit_max = 896.0  # Scene-referred value "hitting the roof" at 10,000 nits

for i in range(len(peak)):
    peakLuminance = peak[i]

    daniele_n = peakLuminance  # peak white

    # pre-calculate Daniele Evo constants
    daniele_r_hit = daniele_r_hit_min + (daniele_r_hit_max - daniele_r_hit_min) * (np.log(daniele_n / daniele_n_r) / np.log(10000.0 / 100.0))
    daniele_m_0 = daniele_n / daniele_n_r
    daniele_m_1 = 0.5 * (daniele_m_0 + np.sqrt(daniele_m_0 * (daniele_m_0 + 4.0 * daniele_t_1)))
    daniele_u = pow((daniele_r_hit / daniele_m_1) / ((daniele_r_hit / daniele_m_1) + 1.0), daniele_g)
    daniele_m = daniele_m_1 / daniele_u
    daniele_w_i = np.log(daniele_n / 100.0) / np.log(2.0)
    daniele_c_t = daniele_c_d * (1.0 + daniele_w_i * daniele_w_g) / daniele_n_r
    daniele_g_ip = 0.5 * (daniele_c_t + np.sqrt(daniele_c_t * (daniele_c_t + 4.0 * daniele_t_1)))
    daniele_g_ipp2 = -daniele_m_1 * pow(daniele_g_ip / daniele_m, 1.0 / daniele_g) / (pow(daniele_g_ip / daniele_m, 1.0 / daniele_g) - 1.0)
    daniele_w_2 = daniele_c / daniele_g_ipp2
    daniele_s_2 = daniele_w_2 * daniele_m_1
    daniele_u_2 = pow((daniele_r_hit / daniele_m_1) / ((daniele_r_hit / daniele_m_1) + daniele_w_2), daniele_g)
    daniele_m_2 = daniele_m_1 / daniele_u_2

    J_in = np.linspace(0, J[i], 1000)

    Y_in = J_to_Y(J_in, L_A, Y_b, surround_y) / 100

    Y_out = daniele_evo_fwd(Y_in, daniele_m_2, daniele_s_2, daniele_g, daniele_t_1)

    J_out = Y_to_J(Y_out * 100, L_A, Y_b, surround_y)

    plt.plot(J_in, J_out)

    plt.title('Peak Luminance = {} nits'.format(peakLuminance))

    plt.xlabel('Input J')
    plt.ylabel('Output J')
    plt.grid()

    plt.show()