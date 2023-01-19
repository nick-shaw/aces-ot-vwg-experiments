import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cusp_path

fig, ax = plt.subplots(figsize=(10,10) )
plt.subplots_adjust(left=0.05, top=0.9, bottom=0.3, right=0.97)

plt.axis([0, 100, 0, 100])
# plt.grid(b=True,which='major',axis='both')
# plt.title('$y=x^p$')

hue_slider = plt.axes([0.05, 0.1, 0.2, 0.01])
h = Slider(hue_slider, 'h', 0, 360, valinit=0, valfmt="%1.1f")

J = np.linspace(0, 100, 256)
M = cusp_path.find_boundary(h.val)

curve, = ax.plot( M, J, color='blue')

def update(val):
    M = cusp_path.find_boundary(h.val)
    curve.set_xdata( M )
    curve.set_ydata( J )
    fig.canvas.draw_idle()

h.on_changed(update)

plt.show()