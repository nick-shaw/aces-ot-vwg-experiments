import numpy as np

# Python implementation of smoothed gamut intersection – https://www.desmos.com/calculator/caou5awk29

# Arbitrary values to test
JM_cusp = np.array([69.9, 51.4])
JM_source = np.array([68.4, 67.5])
gamma_top = 1.12
gamma_bottom = 1.15

smoothness = 20.0

J_max = 100.0

cusp_mid_blend = 0.5
J_focus = cusp_mid_blend * JM_cusp[0] + (1.0 - cusp_mid_blend) * 34.0

focus_dist = 3.5
slope_gain = 50 * J_max * focus_dist / JM_cusp[1]

def solve_J_intersect(JM, focusJ, maxJ):
    a = JM[1] / (focusJ * slope_gain)
    b = 1.0 - JM[1] / slope_gain if JM[0] < focusJ else -(1.0 + JM[1] / slope_gain + maxJ * JM[1] / (focusJ * slope_gain))
    c = -JM[0] if JM[0] < focusJ else maxJ * JM[1] / slope_gain + JM[0]
    root = np.sqrt(b*b - 4 * a * c)
    intersectJ = (-b + root) / (2 * a) if JM[0] < focusJ else (-b - root) / (2 * a)
    return intersectJ

def smin(a, b, s):
    h = max(s - abs(a - b), 0.0) / s
    return min(a, b) - pow(h, 3) * s / 6

J_intersect_source = solve_J_intersect(JM_source, J_focus, J_max)
J_intersect_cusp = solve_J_intersect(JM_cusp, J_focus, J_max)

print("J-axis intersection: J = {:.3f}".format(J_intersect_source))

slope = J_intersect_source * (J_intersect_source - J_focus) / (J_focus * slope_gain) if J_intersect_source < J_focus else (J_max - J_intersect_source) * (J_intersect_source - J_focus) / (J_focus * slope_gain)

M_boundary_lower = J_intersect_cusp * pow(J_intersect_source / J_intersect_cusp, 1 / gamma_bottom) / (JM_cusp[0] / JM_cusp[1] - slope)

M_boundary_upper = JM_cusp[1] * (J_max - J_intersect_cusp) * pow((J_max - J_intersect_source) / (J_max - J_intersect_cusp), 1 / gamma_top) / (slope * JM_cusp[1] + J_max - JM_cusp[0])

M_boundary = smin(M_boundary_lower, M_boundary_upper, smoothness)

# J_boundary is not actually needed, but the calculation would be as follows
J_boundary = J_intersect_source + slope * M_boundary

print("Gamut boundary intersection: JM = [{:.3f}, {:.3f}]".format(J_boundary, M_boundary))