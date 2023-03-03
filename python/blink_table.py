import colour
from colour.algebra import vector_dot, lerp
import numpy as np
import cusp_path
from cusp_path import PLOT_COLOURSPACE, XYZ_to_Hellwig2022, CAM_Specification_Hellwig2022

def HSV_to_RGB( HSV ):
    C = HSV[2]*HSV[1]
    X = C*(1.0-np.abs(((HSV[0]*6.0)%2.0)-1.0))
    m = HSV[2]-C
    RGB = np.zeros(3)
    RGB[0] = (C if HSV[0]<1.0/6.0 else X if HSV[0]<2.0/6.0 else 0.0 if HSV[0]<3.0/6.0 else 0.0 if HSV[0]<4.0/6.0 else X if HSV[0]<5.0/6.0 else C )+m
    RGB[1] = (X if HSV[0]<1.0/6.0 else C if HSV[0]<2.0/6.0 else C if HSV[0]<3.0/6.0 else X if HSV[0]<4.0/6.0 else 0.0 if HSV[0]<5.0/6.0 else 0.0 )+m
    RGB[2] = (0.0 if HSV[0]<1.0/6.0 else 0.0 if HSV[0]<2.0/6.0 else X if HSV[0]<3.0/6.0 else C if HSV[0]<4.0/6.0 else C if HSV[0]<5.0/6.0 else X )+m
    return RGB

def RGB_to_JMh(RGB):
    XYZ = vector_dot(PLOT_COLOURSPACE.matrix_RGB_to_XYZ, np.array(RGB)*100)
    hellwig = XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround, discount_illuminant=True)
    J = hellwig.J
    M = hellwig.M
    h = hellwig.h
    return np.array([J, M, h])


XYZ_w = colour.xy_to_XYZ(PLOT_COLOURSPACE.whitepoint) * 100
L_A = 100.0
Y_b = 20.0
surround = colour.VIEWING_CONDITIONS_HELLWIG2022["Dim"]

gamutCuspTableSize = 360

gamutCuspTableUnsorted = np.zeros((gamutCuspTableSize, 3))
gamutCuspTable = np.zeros((gamutCuspTableSize, 3))

for i in range(gamutCuspTableSize):
    hNorm = i / gamutCuspTableSize
    RGB = HSV_to_RGB([hNorm, 1.0, 1.0])
    gamutCuspTableUnsorted[i] = RGB_to_JMh(RGB)

minhIndex = 0;
for i in range(gamutCuspTableSize):
    if( gamutCuspTableUnsorted[i][2] <  gamutCuspTableUnsorted[minhIndex][2]):
        minhIndex = i


for i in range(gamutCuspTableSize):
  gamutCuspTable[i] = gamutCuspTableUnsorted[(minhIndex+i)%gamutCuspTableSize]

np.savetxt('./data/blink_table_709.txt', gamutCuspTable, fmt='%.8f')

maxDiff = 0.0
index = 0

for i in range(gamutCuspTableSize - 1):
    if (gamutCuspTable[i+1][2] - gamutCuspTable[i][2]) > maxDiff:
        maxDiff = (gamutCuspTable[i+1][2] - gamutCuspTable[i][2])
        index = i

print("Max h difference = {}, index {}-{}:\n{}, {}".format(maxDiff, index, index+1, gamutCuspTable[index][2], gamutCuspTable[index+1][2]))

a = gamutCuspTable[0]
b = gamutCuspTable[359]

b[2] -= 360.0
t = -b[2] / (a[2] - b[2])

c = lerp(t, b, a)

print("cuspFromTable(0):")
print(c)