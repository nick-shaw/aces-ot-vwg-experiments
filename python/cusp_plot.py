import colour
from colour.models import RGB_COLOURSPACE_sRGB as sRGB
from colour.algebra import spow, vector_dot
import numpy as np
import cusp_path
from cusp_path import Hellwig2022_to_XYZ, CAM_Specification_Hellwig2022, find_boundary, find_threshold
import matplotlib.pyplot as plt
import os

if not os.path.exists('./png'):
    os.makedirs('./png')

J = np.linspace(0, 100, 256)
for h in range(360):
	fig, ax = plt.subplots()
	M = find_boundary(h)
	ax.plot(M, J, label='h={}'.format(h))
	plt.xlim(0, 100)
	plt.ylim(0, 100)
	plt.xlabel('Hellwig M')
	plt.ylabel('Hellwig J')
	plt.title('Rec.709 Gamut Boundary')
	plt.legend()
	plt.savefig('./png/cusp_{:0>3}.png'.format(h))
	plt.close()
