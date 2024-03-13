import colour
import numpy as np


# Use to find NaN
# outNaN = np.argwhere(np.isnan(outRGB))

outRGB = colour.read_image("55/Out_sRGB_DigitalLAD.2048x1556.exr")
refRGB = colour.read_image("55/Nuke_sRGB_DigitalLAD.2048x1556.exr")
diffRGB = np.abs(refRGB - outRGB)
print(
    f"SDR stats:\n\tMin: {np.min(diffRGB)}\n\tMean: {np.mean(diffRGB)}\n\tMed: {np.median(diffRGB)}\n\tMax: {np.max(diffRGB)}"
)
np.testing.assert_almost_equal(outRGB, refRGB, decimal=5)

outRGB = colour.read_image("55/Out_Rec2100_DigitalLAD.2048x1556.exr")
outNaN = np.argwhere(np.isnan(outRGB))
refRGB = colour.read_image("55/Nuke_Rec2100_DigitalLAD.2048x1556.exr")
diffRGB = np.abs(refRGB - outRGB)
print(
    f"HDR stats:\n\tMin: {np.min(diffRGB)}\n\tMean: {np.mean(diffRGB)}\n\tMed: {np.median(diffRGB)}\n\tMax: {np.max(diffRGB)}"
)
np.testing.assert_almost_equal(outRGB, refRGB, decimal=5)
