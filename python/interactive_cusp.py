import colour
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import cusp_path
from cusp_path import J_resolution, PLOT_COLOURSPACE, Hellwig2022_to_XYZ, CAM_Specification_Hellwig2022, forwardGamutMapper
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassArithmetic,
    as_float,
    as_float_array,
    from_range_100,
    from_range_degrees,
    full,
    has_only_nan,
    ones,
    to_domain_100,
    to_domain_degrees,
    tsplit,
    tstack,
)
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

J_range = np.linspace(0, 100, J_resolution)

fig, ax = plt.subplots(figsize=(10,10) )
plt.subplots_adjust(left=0.05, top=0.9, bottom=0.3, right=0.97)

plt.axis([0, 100, 0, 100])
plt.title("CAM DRT Gamut Mapping")

hue_slider = plt.axes([0.05, 0.1, 0.2, 0.01])
h = Slider(hue_slider, 'h', 0, 360, valinit=30, valfmt="%1.1f")

source_M = plt.axes([0.05, 0.15, 0.2, 0.01])
SM = Slider(source_M, 'M', 0, 100, valinit=80, valfmt="%1.1f")

source_J = plt.axes([0.05, 0.2, 0.2, 0.01])
SJ = Slider(source_J, 'J', 0, 100, valinit=30, valfmt="%1.1f")

check_box = plt.axes([0.8, 0.1, 0.16, 0.12])
check_boxes = CheckButtons(check_box, ['Show Cusp', 'Show Path', 'Approximation'], [1, 1, 1])

M_bound = cusp_path.find_boundary(h.val)
M_cusp = M_bound.max()
J_cusp = 100.0 * M_bound.argmax() / (J_resolution - 1)
compr = forwardGamutMapper(np.array([SJ.val, SM.val, h.val]), np.array([J_cusp, M_cusp]),
                           check_boxes.get_status()[2] == 1)

CJ, CM, hue, focusJ, ixJ, ixM = tsplit(compr);
RGB = JMh_to_RGB(CJ, CM, h.val)
compressed, = ax.plot(CM, CJ, color=RGB, marker='o')
ix, = ax.plot(ixM, ixJ, color="red", marker='o')
focus, = ax.plot(0, focusJ, color="gray", marker='o')

RGB = JMh_to_RGB(SJ.val, SM.val, h.val)
source, = ax.plot(SM.val, SJ.val, color=RGB, marker='o')

curve, = ax.plot( M_bound, J_range, color='blue')

comp_label = ax.text(75, 10,
    "Focus J = {:.1f}\n\n"
    "Intersection:\n  J = {:.1f}\n  M = {:.1f}\n\n"
    "Normalised ratio = {:.2f}\n\n"
    "Compressed:\n  J = {:.1f}\n  M = {:.1f}"
    .format(focusJ, ixJ, ixM, SM.val / ixM, CJ, CM)
)

if check_boxes.get_status()[0]==1:
    M_cusp = M_bound.max()
    J_cusp = 100.0 * M_bound.argmax() / (J_resolution - 1)
    RGB = JMh_to_RGB(J_cusp, M_cusp, h.val)
    cusp, = ax.plot(M_cusp, J_cusp, color=RGB, marker='o')

if check_boxes.get_status()[1]==1:
    path, = ax.plot([SM.val, CM], [SJ.val, CJ], color='black')
    pathix, = ax.plot([SM.val, ixM], [SJ.val, ixJ], color='black')
    pathix0, = ax.plot([ixM, 0], [ixJ, focusJ], color='black')
    if check_boxes.get_status()[2] == 1:
        ixl0, = ax.plot(np.linspace(0, M_cusp), np.linspace(0, 1)**cusp_path.gamma_approx * J_cusp, color='red')
        ixl1, = ax.plot([0, M_cusp], [100, J_cusp], color='red')


def update(val):
    M_bound = cusp_path.find_boundary(h.val)
    M_cusp = M_bound.max()
    J_cusp = 100.0 * M_bound.argmax() / (J_resolution - 1)
    curve.set_xdata( M_bound )
    curve.set_ydata( J_range )

    compr = forwardGamutMapper(np.array([SJ.val, SM.val, h.val]), np.array([J_cusp, M_cusp]),
                               check_boxes.get_status()[2] == 1)
    CJ, CM, hue, focusJ, ixJ, ixM = tsplit(compr);
    RGB = JMh_to_RGB(CJ, CM, h.val)
    compressed.set_xdata(CM)
    compressed.set_ydata(CJ)
    compressed.set_color(RGB)
    ix.set_xdata(ixM)
    ix.set_ydata(ixJ)
    focus.set_xdata(0)
    focus.set_ydata(focusJ)

    RGB = JMh_to_RGB(SJ.val, SM.val, h.val)
    source.set_xdata(SM.val)
    source.set_ydata(SJ.val)
    source.set_color(RGB)
    comp_label.set_text(
        "Focus J = {:.1f}\n\n"
        "Intersection:\n  J = {:.1f}\n  M = {:.1f}\n\n"
        "Normalised ratio = {:.2f}\n\n"
        "Compressed:\n  J = {:.1f}\n  M = {:.1f}"
        .format(focusJ, ixJ, ixM, SM.val / ixM, CJ, CM)
    )
    if check_boxes.get_status()[0]==1:
        RGB = JMh_to_RGB(J_cusp, M_cusp, h.val)
        cusp.set_xdata(M_cusp)
        cusp.set_ydata(J_cusp)
        cusp.set_color(RGB)
    else:
        cusp.set_xdata(200) # Just a large value outside the plot
    if check_boxes.get_status()[1]==1:
        path.set_xdata([SM.val, CM])
        path.set_ydata([SJ.val, CJ])
        pathix.set_xdata([SM.val, ixM])
        pathix.set_ydata([SJ.val, ixJ])
        pathix0.set_xdata([ixM, 0])
        pathix0.set_ydata([ixJ, focusJ])
        if check_boxes.get_status()[2] == 1:
            ixl0.set_xdata(np.linspace(0, M_cusp))
            ixl0.set_ydata(np.linspace(0, 1)**cusp_path.gamma_approx * J_cusp)
            ixl1.set_xdata([0, M_cusp])
            ixl1.set_ydata([100, J_cusp])
        else:
            ixl0.set_xdata([0, 0])
            ixl0.set_ydata([0, 0])
            ixl1.set_xdata([0, 0])
            ixl1.set_ydata([0, 0])
    else:
        # Large values outside plot
        path.set_xdata([200, 200])
        path.set_ydata([200, 200])
        pathix.set_xdata([200, 200])
        pathix.set_ydata([200, 200])
        pathix0.set_xdata([200, 200])
        pathix0.set_ydata([200, 200])

    fig.canvas.draw_idle()

h.on_changed(update)
SM.on_changed(update)
SJ.on_changed(update)
check_boxes.on_clicked(update)

plt.show()
