import colour
from colour.models import RGB_COLOURSPACE_BT709 as BT709
from colour.algebra import spow, vector_dot
import numpy as np
import cusp_path
from cusp_path import Hellwig2022_to_XYZ, XYZ_to_Hellwig2022, CAM_Specification_Hellwig2022, compress, uncompress
import matplotlib.pyplot as plt

XYZ_w = colour.xy_to_XYZ(BT709.whitepoint) * 100
XYZ_w = [95.05, 100.0, 108.88]
L_A = 100.0
Y_b = 20.0
surround = colour.VIEWING_CONDITIONS_HELLWIG2022["Dim"]
# 
RGB = np.array([0, 0, 100])
XYZ = vector_dot(BT709.matrix_RGB_to_XYZ, RGB)
hellwig = XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround, discount_illuminant=True)
# J, M, h = hellwig.J, hellwig.M, hellwig.h
# print("RGB", RGB)
# print("XYZ", XYZ)
# print("XYZ_w", XYZ_w)
# print("Surround", surround)
# print("JMh", J, M, h)
# print("Compressed", compress(XYZ))
# print("CompEx", uncompress(compress(XYZ)))
# print("XYZ -> JMh -> XYZ", Hellwig2022_to_XYZ(hellwig, XYZ_w, L_A, Y_b, surround, discount_illuminant=True))

# J = np.linspace(0, 100, 256)
# M = cusp_path.find_boundary(180)
# plt.plot(M, J)
# plt.show()