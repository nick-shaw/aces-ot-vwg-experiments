import colour
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import cusp_path
from cusp_path import J_resolution, PLOT_COLOURSPACE, Hellwig2022_to_XYZ, CAM_Specification_Hellwig2022

XYZ_w = colour.xy_to_XYZ(PLOT_COLOURSPACE.whitepoint) * 100
L_A = 100.0
Y_b = 20.0
surround = colour.VIEWING_CONDITIONS_HELLWIG2022["Dim"]

def JMh_to_RGB(J, M, h):
    JMh = CAM_Specification_Hellwig2022(J=J, M=M, h=h)
    XYZ = Hellwig2022_to_XYZ(JMh, XYZ_w, L_A, Y_b, surround, discount_illuminant=True)
    RGB = colour.algebra.vector_dot(PLOT_COLOURSPACE.matrix_XYZ_to_RGB, XYZ) / 100
    RGB = np.nan_to_num(RGB)
    RGB = np.clip(RGB, 0, 1)**(1/2.2)
    return (RGB[0], RGB[1], RGB[2])

def compress(dist, lim=1.6, thr=0.75, power=1.2, invert=False):
    # power(p) compression function plot https://www.desmos.com/calculator/54aytu7hek
    s = (lim-thr)/np.power(np.power((1-thr)/(lim-thr),-power)-1,1/power) # calc y=1 intersect
    if not invert:
        cdist = thr+s*((dist-thr)/s)/(np.power(1+np.power((dist-thr)/s,power),1/power)) # compress
    else:
        cdist = thr+s*np.power(-(np.power((dist-thr)/s,power)/(np.power((dist-thr)/s,power)-1)),1/power) # uncompress

    cdist = np.nan_to_num(cdist)

    cdist[dist < thr] = dist[dist < thr]

    return cdist

def gamut_compress(J, M, M_bound):
    M_cusp = M_bound.max()
    M_thresh = M_bound[int(J_resolution * J / 100.0)]
    print(M_bound[500])
    M_norm = M / M_thresh
    M_comp = compress(np.array([M_norm]))[0]
    J_cusp = 100.0 * M_bound.argmax() / (J_resolution - 1)
    focus = (J_cusp + J_mid) / 2 # cusp to mid blend of 0.5
    J_comp = focus + (J - focus) * M_comp / M_norm
    return J_comp, M_comp * M_thresh

J_range = np.linspace(0, 100, J_resolution)

fig, ax = plt.subplots(figsize=(10,10) )
plt.subplots_adjust(left=0.05, top=0.9, bottom=0.3, right=0.97)

plt.axis([-10, 100, 0, 100])
plt.title("Hellwig JMh Gamut Compression")

ax.plot([0, 0], [0, 100], color='black') # J axis

hue_slider = plt.axes([0.05, 0.1, 0.2, 0.01])
h = Slider(hue_slider, 'h', 0, 360, valinit=180, valfmt="%1.1f")

source_M = plt.axes([0.05, 0.15, 0.2, 0.01])
SM = Slider(source_M, 'M', 0, 100, valinit=50, valfmt="%1.1f")

source_J = plt.axes([0.05, 0.2, 0.2, 0.01])
SJ = Slider(source_J, 'J', 0, 100, valinit=50, valfmt="%1.1f")

M_bound = cusp_path.find_boundary(h.val)
M_cusp = M_bound.max()
J_cusp = 100.0 * M_bound.argmax() / (J_resolution - 1)
J_mid = 34.0 # Hellwig J value for 10 nits
focus = (J_cusp + J_mid) / 2 # cusp to mid blend of 0.5

cusp_J_point, = ax.plot(0, J_cusp, color='black', marker='x')
cusp_J_label = ax.text(1, J_cusp, "J Cusp")

mid_J_point, = ax.plot(0, J_mid, color='black', marker='x')
mid_J_label = ax.text(1, J_mid, "J Mid")

mid_J_line, = ax.plot([0, M_cusp], [J_cusp, J_cusp], color='grey')

focus_point, = ax.plot(0, focus, color='black', marker='x')
focus_label = ax.text(1, focus, "Focus")

CJ, CM = gamut_compress(SJ.val, SM.val, M_bound)

RGB = JMh_to_RGB(CJ, CM, h.val)
compressed, = ax.plot(CM, CJ, color=RGB, marker='o')

check_box = plt.axes([0.8, 0.075, 0.15, 0.15])
check_boxes = CheckButtons(check_box, ['Show Cusp', 'Show Path', 'Show Targets'], [1, 1, 1])

RGB = JMh_to_RGB(SJ.val, SM.val, h.val)
source, = ax.plot(SM.val, SJ.val, color=RGB, marker='o')

curve, = ax.plot( M_bound, J_range, color='blue')

comp_label = ax.text(36, 30, "Compressed:\n  J = {:.1f}\n  M = {:.1f}".format(CJ, CM))

if check_boxes.get_status()[0]==1:
    M_cusp = M_bound.max()
    J_cusp = 100.0 * M_bound.argmax() / (J_resolution - 1)
    RGB = JMh_to_RGB(J_cusp, M_cusp, h.val)
    cusp, = ax.plot(M_cusp, J_cusp, color=RGB, marker='o')

if check_boxes.get_status()[1]==1:
    path, = ax.plot([SM.val, 0], [SJ.val, focus], color='black')

def update(val):
    M_bound = cusp_path.find_boundary(h.val)
    M_cusp = M_bound.max()
    J_cusp = 100.0 * M_bound.argmax() / (J_resolution - 1)
    focus = (J_cusp + J_mid) / 2 # cusp to mid blend of 0.5
    curve.set_xdata( M_bound )
    curve.set_ydata( J_range )
    CJ, CM = gamut_compress(SJ.val, SM.val, M_bound)
    RGB = JMh_to_RGB(CJ, CM, h.val)
    compressed.set_xdata(CM)
    compressed.set_ydata(CJ)
    compressed.set_color(RGB)
    RGB = JMh_to_RGB(SJ.val, SM.val, h.val)
    source.set_xdata(SM.val)
    source.set_ydata(SJ.val)
    source.set_color(RGB)
    comp_label.set_text("Compressed:\n  J = {:.1f}\n  M = {:.1f}".format(CJ, CM))
    comp_label.set_x(CM)
    comp_label.set_y(CJ - 20)
    if check_boxes.get_status()[0]==1:
        RGB = JMh_to_RGB(J_cusp, M_cusp, h.val)
        cusp.set_xdata(M_cusp)
        cusp.set_ydata(J_cusp)
        cusp.set_color(RGB)
    else:
        cusp.set_xdata(200) # Just a large value outside the plot
    if check_boxes.get_status()[1]==1:
        path.set_xdata([SM.val, 0])
        path.set_ydata([SJ.val, focus])
    else:
        # Large values outside plot
        path.set_xdata([200, 200])
        path.set_ydata([200, 200])
    focus_point.set_ydata(focus)
    focus_label.set_y(focus)
    cusp_J_point.set_ydata(J_cusp)
    cusp_J_label.set_y(J_cusp)
    mid_J_line.set_ydata([J_cusp, J_cusp])
    if check_boxes.get_status()[2]==0:
        # Large values outside plot
        focus_point.set_xdata(200)
        focus_label.set_x(200)
        cusp_J_point.set_xdata(200)
        cusp_J_label.set_x(200)
        mid_J_point.set_xdata(200)
        mid_J_label.set_x(200)
        mid_J_line.set_xdata([200, 200])
    else:
        focus_point.set_xdata(0)
        focus_label.set_x(1)
        cusp_J_point.set_xdata(0)
        cusp_J_label.set_x(1)
        mid_J_point.set_xdata(0)
        mid_J_label.set_x(1)
        mid_J_line.set_xdata([0, M_cusp])

    fig.canvas.draw_idle()

h.on_changed(update)
SM.on_changed(update)
SJ.on_changed(update)
check_boxes.on_clicked(update)

plt.show()