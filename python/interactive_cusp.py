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

fig, ax = plt.subplots(figsize=(10,10) )
plt.subplots_adjust(left=0.05, top=0.9, bottom=0.3, right=0.97)

plt.axis([0, 100, 0, 100])

hue_slider = plt.axes([0.05, 0.1, 0.2, 0.01])
h = Slider(hue_slider, 'h', 0, 360, valinit=180, valfmt="%1.1f")

source_M = plt.axes([0.05, 0.15, 0.2, 0.01])
SM = Slider(source_M, 'M', 0, 100, valinit=50, valfmt="%1.1f")

source_J = plt.axes([0.05, 0.2, 0.2, 0.01])
SJ = Slider(source_J, 'J', 0, 100, valinit=50, valfmt="%1.1f")

check_box = plt.axes([0.85, 0.1, 0.12, 0.12])
SHOW_CUSP_COLOUR = CheckButtons(check_box, ['Show Cusp'], [1])

RGB = JMh_to_RGB(SJ.val, SM.val, h.val)
source, = ax.plot(SM.val, SJ.val, color=RGB, marker='o')

J = np.linspace(0, 100, J_resolution)
M = cusp_path.find_boundary(h.val)

curve, = ax.plot( M, J, color='blue')

if SHOW_CUSP_COLOUR.get_status()==[1]:
    M_cusp = M.max()
    J_cusp = 100.0 * M.argmax() / (J_resolution - 1)
    RGB = JMh_to_RGB(J_cusp, M_cusp, h.val)
    cusp, = ax.plot(M_cusp, J_cusp, color=RGB, marker='o')

def update(val):
    M = cusp_path.find_boundary(h.val)
    curve.set_xdata( M )
    curve.set_ydata( J )
    RGB = JMh_to_RGB(SJ.val, SM.val, h.val)
    source.set_xdata(SM.val)
    source.set_ydata(SJ.val)
    source.set_color(RGB)
    if SHOW_CUSP_COLOUR.get_status()==[1]:
        M_cusp = M.max()
        J_cusp = 100.0 * M.argmax() / (J_resolution - 1)
        RGB = JMh_to_RGB(J_cusp, M_cusp, h.val)
        cusp.set_xdata(M_cusp)
        cusp.set_ydata(J_cusp)
        cusp.set_color(RGB)
    else:
        cusp.set_xdata(200) # Just a large value outside the plot
        
    fig.canvas.draw_idle()

h.on_changed(update)
SM.on_changed(update)
SJ.on_changed(update)
SHOW_CUSP_COLOUR.on_clicked(update)

plt.show()