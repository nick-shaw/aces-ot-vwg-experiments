import numpy as np

# Parameters passed to Blink from the UI

# These first four are the only ones that need to be changed depending on the target

peakLuminance = 100.0
# Primaries of the limiting gamut
# 0: AP0-ACES"
# 1: AP1-ACES
# 2: sRGB/Rec.709
# 3: Rec.2020
# 4: P3
primariesLimit = 2
# White point of the limiting gamut
# effectively the 'creative white'
# 0: ACES white")
# 1: D65")
whiteLimit = 1
# Primaries of the Output Encoding
# 0: AP0-ACES
# 1: AP1-ACES
# 2: sRGB/Rec.709-D65
# 3: Rec.2020-D65
# 4: P3-D65
# 5: P3-DCI
primariesOut = 2

gamutCuspTableSize = 360
referenceLuminance = 100.0
# Primaries of the Gamut reached by the gamut compressor
# 0: AP0-ACES
# 1: AP1-ACES
# 2: sRGB/Rec.709-D65
# 3: Rec.2020-D65
# 4: P3-D65
# 5: P3-DCI
# 6: Spectral Locus
# 7: Chroma Compression Space
primariesReach = 1
# Chroma compression params (limit, k1, k2)
chroma_compress = 3.5
chroma_expand = 1.56
chroma_expand_thr = 0.4
# Blend Between Compressing towards
# Target Gamut Cusp Luminance (0.0)
# and Mid Luminance (1.0)
cuspMidBlend = 1.3
# Focus distance of the compression focal point from the achromatic axis
focusDist = 1.1
focusAdjustGain = 0.35
focusAdjustPeakGain = 1.0
focusGainBlend = 0.3
# How much the edges of the target RGB cube are smoothed when finding the gamut boundary
# in order to reduce visible contours at the gamut cusps
smoothCusps = 0.26
# Hellwig 2022 CAM params
ac_resp = 1.0
surround = (0.9, 0.59, 0.9)
# xy coordintes for custom CAT matrix
rxy = (0.82, 0.175)
gxy = (-1.3, 1.8)
bxy = (0.13, -0.15)
wxy = (0.333, 0.333)
ra = ac_resp * 2
ba = 0.05 + (2.0 - ra)
# Input vars
XYZ_w = (95.05, 100.0, 108.88)  # not used?
XYZ_w_scaler = 100.0
L_A = 100.0
Y_b = 20.0
discount_illuminant = False
# Output vars
L_A_out = 100.0
Y_b_out = 20.0
# DanieleEvoCurve (ACES2 candidate) parameters
mmScaleFactor = 100.0      # redundant and equivalent to daniele_n_r
daniele_n = peakLuminance  # peak white
daniele_n_r = 100.0        # Normalized white in nits (what 1.0 should be)
daniele_g = 1.15           # surround / contrast
daniele_c = 0.18           # scene-referred grey
daniele_c_d = 10.013       # display-referred grey (in nits)
daniele_w_g = 0.14         # grey change between different peak luminance
daniele_t_1 = 0.04         # shadow toe, flare/glare compensation - how ever you want to call it
daniele_r_hit_min = 128.0  # Scene-referred value "hitting the roof" at 100 nits
daniele_r_hit_max = 896.0  # Scene-referred value "hitting the roof" at 10,000 nits

# helper functions for code generation
def format_vector(v, decimals=10):
    return "{" + "{: 0.{}f}f, {: 0.{}f}f, {: 0.{}f}f".format(v[0], decimals, v[1], decimals,  v[2], decimals) + " }"

def format_array3(M, name, decimals=10, indent1=0, indent2=4):
    out = " " * indent1 + "{} = ".format(name) + "\n"
    out = out + " " * indent1 + "{\n"
    n = M.shape[0]
    for i in range(n):
#         print(M[i])
        out = out + " " * indent2 + format_vector(M[i], decimals) + ("," if i < n - 1  else "") + "\n"
    out = out + " " * indent1 + "};\n"
    return out

def format_array(M, name, decimals=10, indent1=0, indent2=4):
    out = " " * indent1 + "{} = ".format(name) + "\n"
    out = out + " " * indent1 + "{\n"
    n = M.shape[0]
    for i in range(n):
        out = out + " " * indent2 + "{:0.{}f}".format(M[i], decimals) + ("f," if i < n - 1  else "f") + "\n"
    out = out + " " * indent1 + "};\n"
    return out

# Function definitions from Blink
def RGBPrimsToXYZMatrix(rxy, gxy, bxy, wxy, Y, direction):
# given r g b chromaticities and whitepoint, convert RGB colors to XYZ
# based on CtlColorSpace.cpp from the CTL source code : 77
# param: inverse - calculate XYZ to RGB instead

    r = rxy
    g = gxy
    b = bxy
    w = wxy

    X = w[0] * Y / w[1]
    Z = (1 - w[0] - w[1]) * Y / w[1]

    # Scale factors for matrix rows
    d = r[0] * (b[1] - g[1]) + b[0] * (g[1] - r[1]) + g[0] * (r[1] - b[1])

    Sr =    (X * (b[1] - g[1]) -
            g[0] * (Y * (b[1] - 1.0) +
            b[1]  * (X + Z)) +
            b[0]  * (Y * (g[1] - 1.0) +
            g[1] * (X + Z))) / d

    Sg =    (X * (r[1] - b[1]) +
            r[0] * (Y * (b[1] - 1.0) +
            b[1] * (X + Z)) -
            b[0] * (Y * (r[1] - 1.0) +
            r[1] * (X + Z))) / d

    Sb =    (X * (g[1] - r[1]) -
            r[0] * (Y * (g[1] - 1.0) +
            g[1] * (X + Z)) +
            g[0] * (Y * (r[1] - 1.0) +
            r[1] * (X + Z))) / d

    # Assemble the matrix
    Mdata = np.array([
        Sr * r[0], Sr * r[1], Sr * (1.0 - r[0] - r[1]),
        Sg * g[0], Sg * g[1], Sg * (1.0 - g[0] - g[1]),
        Sb * b[0], Sb * b[1], Sb * (1.0 - b[0] - b[1])
    ])

    newMatrix = np.array([
        [Mdata[0], Mdata[3], Mdata[6]],
        [Mdata[1], Mdata[4], Mdata[7]],
        [Mdata[2], Mdata[5], Mdata[8]],
    ])

    newMatrixInverse = np.linalg.inv(newMatrix)

    # return forward or inverse matrix
    if (direction == 0):
      return newMatrix
    elif (direction == 1):
      return newMatrixInverse

# multiplies a 3D vector with a 3x3 matrix
def vector_dot(m, v):
    r = np.ones(3)
    for c in range(3):
      r[c] = m[c][0]*v[0] + m[c][1]*v[1] + m[c][2]*v[2]

    return r

# convert achromatic luminance to Hellwig J
def Y_to_J(Y, L_A, Y_b, surround_y):
    k = 1.0 / (5.0 * L_A + 1.0)
    k4 = k*k*k*k
    F_L = 0.2 * k4 * (5.0 * L_A) + 0.1 * np.power((1.0 - k4), 2.0) * np.power(5.0 * L_A, 1.0 / 3.0)
    n = Y_b / 100.0
    z = 1.48 + np.sqrt(n)
    F_L_W = np.power(F_L, 0.42)
    A_w = (400.0 * F_L_W) / (27.13 + F_L_W)

    F_L_Y = np.power(F_L * abs(Y) / 100.0, 0.42)

    return np.sign(Y) * (100.0 * np.power(((400.0 * F_L_Y) / (27.13 + F_L_Y)) / A_w, surround_y * z))

# convert HSV cylindrical projection values to RGB
def HSV_to_RGB( HSV ):
  C = HSV[2] * HSV[1]
  X = C * (1.0 - abs((HSV[0] * 6.0) % 2.0 - 1.0))
  m = HSV[2] - C

  RGB = np.zeros(3)
  RGB[0] = (C if HSV[0] < 1 / 6 else X if HSV[0] < 2 / 6 else 0 if HSV[0] < 3 /6 else 0 if HSV[0] < 4 / 6 else X if HSV[0] < 5 / 6 else C ) + m
  RGB[1] = (X if HSV[0] < 1 / 6 else C if HSV[0] < 2 / 6 else C if HSV[0] < 3 /6 else X if HSV[0] < 4 / 6 else 0 if HSV[0] < 5 / 6 else 0 ) + m
  RGB[2] = (0 if HSV[0] < 1 / 6 else 0 if HSV[0] < 2 / 6 else X if HSV[0] < 3 /6 else C if HSV[0] < 4 / 6 else C if HSV[0] < 5 / 6 else X ) + m

  return RGB

def spow(base, exponent):
    if(base < 0.0 and exponent != np.floor(exponent)):
      return 0.0
    else:
      return pow(base, exponent)

def float3spow(base, exponent):
      return np.array([spow(base[0], exponent), spow(base[1], exponent), spow(base[2], exponent)])

# "safe" div
def sdiv( a, b ):
    if(b == 0.0):
      return 0.0
    else:
      return a / b

def post_adaptation_non_linear_response_compression_forward(RGB, F_L):
  F_L_RGB = float3spow(F_L * np.abs(RGB) / 100.0, 0.42)
  RGB_c = (400.0 * np.sign(RGB) * F_L_RGB) / (27.13 + F_L_RGB)
  return RGB_c

# convert linear RGB values with the limiting primaries to CAM J (lightness), M (colorfulness) and h (hue) correlates
def limit_RGB_to_JMh(RGB):
  luminanceRGB = RGB * boundaryRGB * referenceLuminance
  XYZ = vector_dot(RGB_to_XYZ_limit, luminanceRGB)
  JMh = XYZ_to_Hellwig2022_JMh(XYZ, refWhite, L_A, Y_b, surround)

  return JMh

# convert CAM J (lightness), M (colorfulness) and h (hue) correlates to linear RGB values with the limiting primaries
def JMh_to_limit_RGB(JMh):
  luminanceXYZ = Hellwig2022_JMh_to_XYZ( JMh, refWhite, surround, L_A, Y_b)
  luminanceRGB = vector_dot(XYZ_to_RGB_limit, luminanceXYZ)
  RGB = luminanceRGB / boundaryRGB / referenceLuminance

  return RGB;

# basic 3D hypotenuse function, does not deal with under/overflow
def hypot_float3(xyz):
    return np.sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2])

def compress_bjorn(xyz):
    C = (xyz[0]+xyz[1]+xyz[2])/3

    xyz_temp = xyz - C
    R = hypot_float3(xyz_temp)

    if (R == 0.0 or C == 0.0):
      return xyz

    R = R * 0.816496580927726     # sqrt(2/3)

    xyz_temp = xyz_temp / R

    r = R/C
    r = r*r
    r = np.sqrt(4.0 / r + 1.0) - 1.0

    s = -min(xyz_temp[0], min(xyz_temp[1], xyz_temp[2]))
    s = s -0.5

    t = 0.5 + np.sqrt(s*s + r*r / 4.0)
    t = C / t                                 # t is always >= 0.5

    xyz_temp = xyz_temp * t + C

    return xyz_temp

def uncompress_bjorn(xyz):
    C = (xyz[0]+xyz[1]+xyz[2])/3

    xyz_temp = xyz - C
    R = hypot_float3(xyz_temp)

    if (R == 0.0 or C == 0.0):
      return xyz

    R = R * 0.816496580927726     # sqrt(2/3)

    xyz_temp = xyz_temp / R

    t = C/R
    t = t - 0.5

    s = -min(xyz_temp[0], min(xyz_temp[1], xyz_temp[2]))
    s = s - 0.5

    r = 2 * np.sqrt(np.abs(t*t - s*s)) + 1
    r = np.sqrt(np.abs(r*r - 1))
    if (r == 0.0):
      return xyz
    r = C * 2/r

    xyz_temp = xyz_temp * r + C

    return xyz_temp

# convert radians to degrees
def degrees( radians ):
    return radians * 180.0 / np.pi

# convert degrees to radians
def radians( degrees ):
    return degrees / 180.0 * np.pi

def XYZ_to_Hellwig2022_JMh(XYZ, XYZ_w, L_A, Y_b, surround):
  XYZ_w = XYZ_w * XYZ_w_scaler
  # Step 0
  # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
  MATRIX_16 = CAT_CAT16
  RGB_w = vector_dot(MATRIX_16, XYZ_w)

  # Always discount illuminant so this calculation is omitted
  # D of 1.0 actually cancels out, so could be removed entirely
  D = 1.0

  # Viewing conditions dependent parameters
  k = 1 / (5 * L_A + 1)
  k4 = pow(k,4)
  F_L = 0.2 * k4 * (5.0 * L_A) + 0.1 * pow((1.0 - k4), 2) * pow(5.0 * L_A, 1.0 / 3.0)
  n = Y_b / XYZ_w[1]
  z = 1.48 + np.sqrt(n)

  D_RGB = D * XYZ_w[1] / RGB_w + 1 - D
  RGB_wc = D_RGB * RGB_w
  RGB_aw = post_adaptation_non_linear_response_compression_forward(RGB_wc, F_L)

  # Computing achromatic responses for the whitepoint.
  R_aw = RGB_aw[0]
  G_aw = RGB_aw[1]
  B_aw = RGB_aw[2]

  A_w = ra * R_aw + G_aw + ba * B_aw

  # Step 1
  # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
  RGB = vector_dot(MATRIX_16, XYZ)

  # Step 2
  RGB_c = D_RGB * RGB

  # Step 3
  # Applying forward post-adaptation non-linear response compression.

  # always compressMode
  RGB_c = compress_bjorn(RGB_c)

  RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L)

  # always compressMode
  RGB_a = uncompress_bjorn(RGB_a)

  # Step 4
  # Converting to preliminary cartesian coordinates.
  R_a = RGB_a[0]
  G_a = RGB_a[1]
  B_a = RGB_a[2]
  a = R_a - 12.0 * G_a / 11.0 + B_a / 11.0
  b = (R_a + G_a - 2.0 * B_a) / 9.0

  # Computing the *hue* angle :math:`h`.
  hr = np.arctan2(b, a)
  h = degrees(hr) % 360.0

  # Step 6
  # Computing achromatic responses for the stimulus.
  R_a2 = RGB_a[0]
  G_a2 = RGB_a[1]
  B_a2 = RGB_a[2]

  A = ra * R_a2 + G_a2 + ba * B_a2

  # Step 7
  # Computing the correlate of *Lightness* :math:`J`.
  J = 100.0 * spow(sdiv(A, A_w), surround[1] * z)

  # Step 9
  # Computing the correlate of *colourfulness* :math:`M`.
  M = 43.0 * surround[2] * np.sqrt(a * a + b * b)

  # HK effect block omitted, aas we always have that off

  return np.array([J, M, h])

# convert CAM J (lightness), M (colorfulness) and h (hue) correlates to linear RGB values with the reach primaries
def JMh_to_reach_RGB(JMh):
   luminanceXYZ = Hellwig2022_JMh_to_XYZ( JMh, refWhite, surround, L_A, Y_b)
   luminanceRGB = vector_dot(XYZ_to_RGB_reach, luminanceXYZ)
   RGB = luminanceRGB / boundaryRGB / referenceLuminance
   return RGB

def post_adaptation_non_linear_response_compression_inverse(RGB, F_L):
  RGB_p =  (np.sign(RGB) * 100.0 / F_L * float3spow((27.13 * np.abs(RGB)) / (400.0 - np.abs(RGB)), 1.0 / 0.42) )
  return RGB_p

def Hellwig2022_JMh_to_XYZ( JMh, XYZ_w, surround, L_A, Y_b):
  J = JMh[0]
  M = JMh[1]
  h = JMh[2]
  XYZ_w = XYZ_w * XYZ_w_scaler
  # Step 0
  # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
  MATRIX_16 = CAT_CAT16
  RGB_w = vector_dot(MATRIX_16, XYZ_w)

  # Always discount illuminant so this calculation is omitted
  # D of 1.0 actually cancels out, so could be removed entirely
  D = 1.0

  # Viewing conditions dependent parameters
  k = 1 / (5 * L_A + 1)
  k4 = pow(k,4)
  F_L = 0.2 * k4 * (5.0 * L_A) + 0.1 * pow((1.0 - k4), 2) * pow(5.0 * L_A, 1.0 / 3.0)
  n = Y_b / XYZ_w[1]
  z = 1.48 + np.sqrt(n)

  D_RGB = D * XYZ_w[1] / RGB_w + 1 - D
  RGB_wc = D_RGB * RGB_w
  RGB_aw = post_adaptation_non_linear_response_compression_forward(RGB_wc, F_L)

  # Computing achromatic responses for the whitepoint.
  R_aw = RGB_aw[0]
  G_aw = RGB_aw[1]
  B_aw = RGB_aw[2]

  A_w = ra * R_aw + G_aw + ba * B_aw

  hr = radians(h)

  # HK effect block omitted, aas we always have that off

  # Computing achromatic response :math:`A` for the stimulus.
  A = A_w * spow(J / 100.0, 1.0 / (surround[1] * z))

  # Computing *P_p_1* to *P_p_2*.
  P_p_1 = 43.0 * surround[2]
  P_p_2 = A

  # Step 3
  # Computing opponent colour dimensions :math:`a` and :math:`b`.
  gamma = M / P_p_1
  a = gamma * np.cos(hr)
  b = gamma * np.sin(hr)

  # Step 4
  # Applying post-adaptation non-linear response compression matrix.
  RGB_a = vector_dot(panlrcm, np.array([P_p_2, a, b])) / 1403.0

  # Step 5
  # Applying inverse post-adaptation non-linear response compression.
  # always compressMode
  RGB_a = compress_bjorn(RGB_a)

  RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L)

  # always compressMode
  RGB_c = uncompress_bjorn(RGB_c)

  # Step 6
  RGB = RGB_c / D_RGB

  # Step 7
  MATRIX_INVERSE_16 = np.linalg.inv(CAT_CAT16)
  XYZ = vector_dot(MATRIX_INVERSE_16, RGB)

  return XYZ

# linear interpolation between two values a & b with the bias t
def lerp(a, b, t):
  return a + t * (b - a)

# retrieve the JM coordinates of the limiting gamut cusp at the hue slice 'h'
# cusps are very expensive to compute
# and the DRT is only using them for lightness mapping
# which does not require a high degree of accuracy
# so instead we use a pre-computed table of cusp points
# sampled at 1 degree hue intervals of the the RGB target gamut
# and lerp between them to get the approximate J & M values
def cuspFromTable(h):
  lo = np.zeros(3)
  hi = np.zeros(3)

  if( h <= gamutCuspTable[0][2] ):
    lo = gamutCuspTable[gamutCuspTableSize-1].copy()
    lo[2] = lo[2]-360.0
    hi = gamutCuspTable[0].copy()
  elif( h >= gamutCuspTable[gamutCuspTableSize-1][2] ):
    lo = gamutCuspTable[gamutCuspTableSize-1].copy()
    hi = gamutCuspTable[0].copy()
    hi[2] = hi[2]+360.0
  else:
    for i in range(gamutCuspTableSize):
      if( h <= gamutCuspTable[i][2] ):
        lo = gamutCuspTable[i-1].copy()
        hi = gamutCuspTable[i].copy()
        break

  t = (h - lo[2]) / (hi[2] - lo[2])

  cuspJ = lerp(lo[0], hi[0], t)
  cuspM = lerp(lo[1], hi[1], t)

  return np.array([cuspJ, cuspM])

# Smooth minimum of a and b
def smin(a, b, s):
    h = max(s - abs(a - b), 0.0) / s
    return min(a, b) - h * h * h * s * (1.0 / 6.0)

# reimplemented from https:#github.com/nick-shaw/aces-ot-vwg-experiments/blob/master/python/intersection_approx.py
def solve_J_intersect(JM, focusJ, maxJ, slope_gain):
  a = JM[1] / (focusJ * slope_gain)
  b = 0.0
  c = 0.0
  intersectJ = 0.0

  if (JM[0] < focusJ):
    b = 1.0 - JM[1] / slope_gain
  else:
    b = -(1.0 + JM[1] / slope_gain + maxJ * JM[1] / (focusJ * slope_gain))

  if (JM[0] < focusJ):
    c = -JM[0]
  else:
    c = maxJ * JM[1] / slope_gain + JM[0]

  root = np.sqrt(b*b - 4.0 * a * c)

  if (JM[0] < focusJ):
    intersectJ = 2.0 * c / (-b - root)
  else:
    intersectJ = 2.0 * c / (-b + root)

  return intersectJ

# reimplemented from https:#github.com/nick-shaw/aces-ot-vwg-experiments/blob/master/python/intersection_approx.py
def findGamutBoundaryIntersection(JMh_s, JM_cusp, J_focus, J_max, slope_gain, smoothness, gamma_top, gamma_bottom):
    JM_source = [JMh_s[0], JMh_s[1]]

    slope = 0.0

    s = max(0.000001, smoothness)
    JM_cusp[0] *= 1.0 + 0.06 * s   # J
    JM_cusp[1] *= 1.0 + 0.18 * s   # M

    J_intersect_source = solve_J_intersect(JM_source, J_focus, J_max, slope_gain)
    J_intersect_cusp = solve_J_intersect(JM_cusp, J_focus, J_max, slope_gain)

    if (J_intersect_source < J_focus):
        slope = J_intersect_source * (J_intersect_source - J_focus) / (J_focus * slope_gain)
    else:
        slope = (J_max - J_intersect_source) * (J_intersect_source - J_focus) / (J_focus * slope_gain)

    M_boundary_lower = J_intersect_cusp * pow(J_intersect_source / J_intersect_cusp, 1 / gamma_bottom) / (JM_cusp[0] / JM_cusp[1] - slope)

    M_boundary_upper = JM_cusp[1] * (J_max - J_intersect_cusp) * pow((J_max - J_intersect_source) / (J_max - J_intersect_cusp), 1.0 / gamma_top) / (slope * JM_cusp[1] + J_max - JM_cusp[0])

    M_boundary = JM_cusp[1] * smin(M_boundary_lower / JM_cusp[1], M_boundary_upper / JM_cusp[1], s)

    J_boundary = J_intersect_source + slope * M_boundary

    return np.array([J_boundary, M_boundary, J_intersect_source])


# Blink init() block
HALF_MINIMUM = 0.0000000596046448
HALF_MAXIMUM = 65504.0

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

# 1.0 / (c * z)
model_gamma = 1.0 / (surround[1] * (1.48 + np.sqrt(Y_b_out / L_A_out)))

# Chroma compression scaling for HDR/SDR appearance match
log_peak = np.log10(daniele_n / daniele_n_r)
compr = chroma_compress + (chroma_compress * 5.0) * log_peak
sat = max(0.15, chroma_expand - (chroma_expand * 0.78) * log_peak)
sat_thr = chroma_expand_thr / daniele_n

identity_matrix = np.identity(3)

XYZ_to_AP0_ACES_matrix = np.array([
[ 1.0498110175,  0.0000000000, -0.0000974845],
[-0.4959030231,  1.3733130458,  0.0982400361],
[ 0.0000000000,  0.0000000000,  0.9912520182]
])

XYZ_to_AP1_ACES_matrix = np.array([
[ 1.6410233797, -0.3248032942, -0.2364246952],
[-0.6636628587,  1.6153315917,  0.0167563477],
[ 0.0117218943, -0.0082844420,  0.9883948585]
])

XYZ_to_Rec709_D65_matrix = np.array([
[ 3.2409699419, -1.5373831776, -0.4986107603],
[-0.9692436363,  1.8759675015,  0.0415550574],
[ 0.0556300797, -0.2039769589,  1.0569715142]
])

XYZ_to_Rec2020_D65_matrix = np.array([
[ 1.7166511880, -0.3556707838, -0.2533662814],
[-0.6666843518,  1.6164812366,  0.0157685458],
[ 0.0176398574, -0.0427706133,  0.9421031212]
])

XYZ_to_P3_D65_matrix = np.array([
[ 2.4934969119, -0.9313836179, -0.4027107845],
[-0.8294889696,  1.7626640603,  0.0236246858],
[ 0.0358458302, -0.0761723893,  0.9568845240]
])

XYZ_to_P3_DCI_matrix = np.array([
[ 2.7253940305, -1.0180030062, -0.4401631952],
[-0.7951680258,  1.6897320548,  0.0226471906],
[ 0.0412418914, -0.0876390192,  1.1009293786]
])

# populate the input primaries matrix
XYZ_to_RGB_input = XYZ_to_AP0_ACES_matrix

# populate the limiting primaries matrix
# RGBPrimsToXYZMatrix
limitWhiteForMatrix = (0.0, 0.0)
limitRedForMatrix = (0.0, 0.0)
limitGreenForMatrix = (0.0, 0.0)
limitBlueForMatrix = (0.0, 0.0)
if( whiteLimit == 0):
  limitWhiteForMatrix = (0.32168, 0.33767)
elif( whiteLimit == 1):
  limitWhiteForMatrix = (0.3127, 0.3290)
else:
  limitWhiteForMatrix = (0.333333, 0.333333)

if( primariesLimit == 0 ):
  limitRedForMatrix = (0.7347, 0.2653)
  limitGreenForMatrix = (0.0, 1.0)
  limitBlueForMatrix = (0.0001, -0.077)
elif( primariesLimit == 1 ):
  limitRedForMatrix = (0.713, 0.293)
  limitGreenForMatrix = (0.165, 0.830)
  limitBlueForMatrix = (0.128, 0.044)
elif( primariesLimit == 2 ):
  limitRedForMatrix = (0.64, 0.33)
  limitGreenForMatrix = (0.3, 0.6)
  limitBlueForMatrix = (0.15, 0.06)
elif( primariesLimit == 3 ):
  limitRedForMatrix = (0.708, 0.292)
  limitGreenForMatrix = (0.170, 0.797)
  limitBlueForMatrix = (0.131, 0.046)
elif( primariesLimit == 4 ):
  limitRedForMatrix = (0.680, 0.320)
  limitGreenForMatrix = (0.265, 0.690)
  limitBlueForMatrix = (0.150, 0.060)
else:
  limitRedForMatrix = (1.0, 0.0)
  limitGreenForMatrix = (0.0, 1.0)
  limitBlueForMatrix = (0.0, 0.0)

XYZ_to_RGB_limit = RGBPrimsToXYZMatrix(limitRedForMatrix, limitGreenForMatrix, limitBlueForMatrix, limitWhiteForMatrix, 1.0, 1)

# populate the reach primaries matrix
XYZ_to_RGB_reach = identity_matrix
if( primariesReach == 0 ):
  XYZ_to_RGB_reach = XYZ_to_AP0_ACES_matrix
elif( primariesReach == 1 ):
  XYZ_to_RGB_reach = XYZ_to_AP1_ACES_matrix
elif( primariesReach == 2 ):
  XYZ_to_RGB_reach = XYZ_to_Rec709_D65_matrix
elif( primariesReach == 3 ):
  XYZ_to_RGB_reach = XYZ_to_Rec2020_D65_matrix
elif( primariesReach == 4 ):
  XYZ_to_RGB_reach = XYZ_to_P3_D65_matrix
elif( primariesReach == 5 ):
  XYZ_to_RGB_reach = XYZ_to_P3_DCI_matrix

# populate the output primaries matrix
XYZ_to_RGB_output = identity_matrix
if( primariesOut == 0 ):
  XYZ_to_RGB_output = XYZ_to_AP0_ACES_matrix
elif( primariesOut == 1 ):
  XYZ_to_RGB_output = XYZ_to_AP1_ACES_matrix
elif( primariesOut == 2 ):
  XYZ_to_RGB_output = XYZ_to_Rec709_D65_matrix
elif( primariesOut == 3 ):
  XYZ_to_RGB_output = XYZ_to_Rec2020_D65_matrix
elif( primariesOut == 4 ):
  XYZ_to_RGB_output = XYZ_to_P3_D65_matrix
elif( primariesOut == 5 ):
  XYZ_to_RGB_output = XYZ_to_P3_DCI_matrix

RGB_to_XYZ_input = np.linalg.inv(XYZ_to_RGB_input)
RGB_to_XYZ_limit = np.linalg.inv(XYZ_to_RGB_limit)
RGB_to_XYZ_reach = np.linalg.inv(XYZ_to_RGB_reach)
RGB_to_XYZ_output = np.linalg.inv(XYZ_to_RGB_output)

XYZ_to_AP1 = XYZ_to_AP1_ACES_matrix
AP1_to_XYZ = np.linalg.inv(XYZ_to_AP1)

CAT_CAT16 = RGBPrimsToXYZMatrix(rxy, gxy, bxy, wxy, 1.0, 1)

white = np.array([1.0, 1.0, 1.0])

inWhite = vector_dot(RGB_to_XYZ_input, white)
outWhite = vector_dot(RGB_to_XYZ_output, white)
refWhite = vector_dot(RGB_to_XYZ_limit, white)

boundaryRGB = peakLuminance / referenceLuminance

# Generate the Hellwig2022 post adaptation non-linear compression matrix
# that is used in the inverse of the model (JMh-to-XYZ).
#
# Original:
#  460.0f, 451.0f, 288.0f,
#  460.0f, -891.0f, -261.0f,
#  460.0f, -220.0f, -6300.0f
panlrcm = np.array([
  [ra, 1.0, ba],
  [1.0, -12.0 / 11.0, 1.0 / 11.0],
  [1.0 / 9.0, 1.0 / 9.0, -2.0 / 9.0]
])
panlrcm = np.linalg.inv(panlrcm)

# Normalize rows so that first column is 460
for i in range(3):
  n = 460.0 / panlrcm[i][0]
  panlrcm[i] *= n

# limitJmax (asumed to match limitRGB white)
limitJmax = Y_to_J(peakLuminance, L_A, Y_b, surround[1])

# Cusp table for chroma compression gamut
tmpx = XYZ_to_RGB_limit
tmpr = RGB_to_XYZ_limit
tmpR = XYZ_to_RGB_reach

XYZ_to_RGB_reach = XYZ_to_AP1_ACES_matrix
RGB_to_XYZ_limit = np.linalg.inv(XYZ_to_RGB_reach)

gamutCuspTableUnsorted = np.zeros((gamutCuspTableSize, 3))
for i in range(gamutCuspTableSize):
  hNorm = float(i) / gamutCuspTableSize
  RGB = HSV_to_RGB([hNorm, 1.0, 1.0])
  gamutCuspTableUnsorted[i] = limit_RGB_to_JMh(RGB)

minhIndex = 0
for i in range(1, gamutCuspTableSize):
  if( gamutCuspTableUnsorted[i][2] <  gamutCuspTableUnsorted[minhIndex][2]):
    minhIndex = i

cgamutCuspTable = np.zeros((gamutCuspTableSize, 3))
for i in range(gamutCuspTableSize):
  cgamutCuspTable[i] = gamutCuspTableUnsorted[(minhIndex+i)%gamutCuspTableSize]

# Reach table for the chroma compression reach. If AP1 this is the same as gamutCuspTableReach
cgamutReachTable = np.zeros((gamutCuspTableSize, 3))  # float3 table for parity with Blink. Could just be a float table
for i in range(gamutCuspTableSize):
  cgamutReachTable[i][2] = i
  for M in range(1300):
    sampleM = float(M)
    newLimitRGB = JMh_to_reach_RGB(np.array([limitJmax, sampleM, i]))
    if (newLimitRGB[0] < 0.0 or newLimitRGB[1] < 0.0 or newLimitRGB[2] < 0.0):
      cgamutReachTable[i][1] = sampleM
      break

XYZ_to_RGB_limit = tmpx
RGB_to_XYZ_limit = tmpr

# Cusp table for limiting gamut
gamutCuspTableUnsorted = np.zeros((gamutCuspTableSize, 3))
for i in range(gamutCuspTableSize):
  hNorm = float(i) / gamutCuspTableSize
  RGB = HSV_to_RGB([hNorm, 1.0, 1.0])
  gamutCuspTableUnsorted[i] = limit_RGB_to_JMh(RGB)

minhIndex = 0
for i in range(1, gamutCuspTableSize):
  if( gamutCuspTableUnsorted[i][2] <  gamutCuspTableUnsorted[minhIndex][2]):
    minhIndex = i

gamutCuspTable = np.zeros((gamutCuspTableSize, 3))
for i in range(gamutCuspTableSize):
  gamutCuspTable[i] = gamutCuspTableUnsorted[(minhIndex+i)%gamutCuspTableSize]

# Cusp table for limiting reach gamut, values at a J of 100.  Covers M values
# up to 10000 nits.
gamutCuspTableReach = np.zeros((gamutCuspTableSize, 3))  # float3 table for parity with Blink. Could just be a float table
for i in range(gamutCuspTableSize):
  gamutCuspTableReach[i][2] = i
  for M in range(1300):
    sampleM = float(M)
    newLimitRGB = JMh_to_reach_RGB(np.array([limitJmax, sampleM, i]))
    if (newLimitRGB[0] < 0.0 or newLimitRGB[1] < 0.0 or newLimitRGB[2] < 0.0):
      gamutCuspTableReach[i][1] = sampleM
      break

midJ = Y_to_J(daniele_c_t * mmScaleFactor, L_A, Y_b, surround[1])

# Find upper hull gamma values for the gamut mapper
# start by taking a h angle
# get the cusp J value for that angle
# find a J value halfway to the Jmax
# iterate through gamma values until the approxilate max M is negative through the actual boundry
gamutTopGamma = np.zeros(gamutCuspTableSize)
for i in range(gamutCuspTableSize):
  # get cusp from cusp table at hue position
  JMcusp = cuspFromTable(float(i))
  # create test value halfway betwen the cusp and the Jmax
  # positions between the cusp and Jmax we will check
  testPositions = [0.01, 0.5, 0.99]
  # variables that get set as we iterate through, once all 3 are set to true we break the loop
  gammaFound = [False, False, False]
  # limit value, once we cross this value, we are outside of the top gamut shell 
  maxRGBtestVal = 1.0
  # Tg is Test Gamma. the values are shifted two decimal points to the left. Tg 70 = Gamma 0.7
  for Tg in range(70, 170):
    # topGamma value created from the Tg variable
    topGamma = float(Tg) / 100.0
    # loop to run through each of the positions defined in the testPositions list
    for testIndex in range(3):
      testJmh = np.array([JMcusp[0] + ((limitJmax - JMcusp[0]) * testPositions[testIndex] ), JMcusp[1] , float(i)])
      approxLimit  =  findGamutBoundaryIntersection(testJmh, JMcusp, lerp(JMcusp[0], midJ, cuspMidBlend), limitJmax, 10000.0, 0.0, topGamma, 1.0)
      newLimitRGB = JMh_to_limit_RGB(np.array([approxLimit[0], approxLimit[1], float(i)]))
      # if any channel has broken through the top gamut hull, break
      if (newLimitRGB[0] > maxRGBtestVal or newLimitRGB[1] > maxRGBtestVal or newLimitRGB[2] > maxRGBtestVal):
        gamutTopGamma[i] = topGamma
        gammaFound[testIndex] = True
    # once all 3 of the test
    if (gammaFound[0] and gammaFound[1] and gammaFound[2]):
      break

print()
print(format_array3(XYZ_to_RGB_input, "__CONSTANT__ float3x3 XYZ_to_RGB_input"))
print(format_array3(XYZ_to_RGB_limit, "__CONSTANT__ float3x3 XYZ_to_RGB_limit"))
print(format_array3(XYZ_to_RGB_reach, "__CONSTANT__ float3x3 XYZ_to_RGB_reach"))
print(format_array3(XYZ_to_RGB_output, "__CONSTANT__ float3x3 XYZ_to_RGB_output"))
print(format_array3(RGB_to_XYZ_input, "__CONSTANT__ float3x3 RGB_to_XYZ_input"))
print(format_array3(RGB_to_XYZ_limit, "__CONSTANT__ float3x3 RGB_to_XYZ_limit"))
print(format_array3(RGB_to_XYZ_reach, "__CONSTANT__ float3x3 RGB_to_XYZ_reach"))
print(format_array3(RGB_to_XYZ_output, "__CONSTANT__ float3x3 RGB_to_XYZ_output"))
print(format_array3(XYZ_to_AP1, "__CONSTANT__ float3x3 XYZ_to_AP1"))
print(format_array3(AP1_to_XYZ, "__CONSTANT__ float3x3 AP1_to_XYZ"))
print(format_array3(CAT_CAT16, "__CONSTANT__ float3x3 CAT_CAT16"))
print(format_array3(panlrcm, "__CONSTANT__ float3x3 panlrcm", 1))
print("__CONSTANT__ float3 inWhite = " + format_vector(inWhite) + ";")
print("__CONSTANT__ float3 outWhite = " + format_vector(outWhite) + ";")
print("__CONSTANT__ float3 refWhite = " + format_vector(refWhite) + ";")
print("__CONSTANT__ float limitJmax = {:.2f}f;".format(limitJmax))
print("__CONSTANT__ float midJ = {:.10f}f;".format(midJ))
print()
print(format_array3(gamutCuspTable, "__CONSTANT__ float3 gamutCuspTable[360]"))
print(format_array3(gamutCuspTableReach, "__CONSTANT__ float3 gamutCuspTableReach[360]", 1))
print(format_array3(cgamutCuspTable, "__CONSTANT__ float3 cgamutCuspTable[360]"))
print(format_array3(cgamutReachTable, "__CONSTANT__ float3 cgamutReachTable[360]", 1))
print(format_array(gamutTopGamma, "__CONSTANT__ float gamutTopGamma[360]", 2))
