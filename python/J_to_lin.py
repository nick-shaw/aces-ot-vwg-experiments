import colour
from colour.algebra import vector_dot, lerp
from colour.utilities import tsplit
import numpy as np
import cusp_path
from cusp_path import PLOT_COLOURSPACE, XYZ_to_Hellwig2022, Hellwig2022_to_XYZ, CAM_Specification_Hellwig2022
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

XYZ_w = colour.xy_to_XYZ(PLOT_COLOURSPACE.whitepoint) * 100
L_A = 100.0
Y_b = 20.0
surround = colour.VIEWING_CONDITIONS_HELLWIG2022["Dim"]

samples = 1000

J = np.linspace(0, 500, samples)
M = np.zeros(samples)
h = np.zeros(samples)

JMh = CAM_Specification_Hellwig2022(J=J, M=M, h=h)
XYZ = Hellwig2022_to_XYZ(JMh, XYZ_w, L_A, Y_b, surround, discount_illuminant=True)
X, Y, Z = tsplit(XYZ)

Y_norm = Y / 100
J_norm = J / 100

print("Exponent:")
print(np.log(Y_norm) / np.log(J_norm))

def f(x, p, o):
    return ((x + o)**p - o**p) / ((1 + o)**p - o**p)

p_opt, e = curve_fit(f, J_norm, Y_norm, p0=[2.15, 0.1], bounds=(0, 3))

print("[p, o] =")
print(p_opt)

plt.plot(J, 100*Y_norm)
plt.plot(J, 100 * f(J_norm, p_opt[0], p_opt[1]))

plt.text(50, 3000, r'Luminance = '
                   r'$100\times\frac{(\frac{J}{100}+0.192)^{2.459}-0.192^{2.459}}{(1-0.192)^{2.459}-0.192^{2.459}}$')
plt.xlabel("Hellwig J")
plt.ylabel("Luminance")
plt.title("Mono Hellwig J to Luminance")

plt.show()