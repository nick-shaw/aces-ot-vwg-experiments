import sys, os
import numpy as np
from colour import read_image
from colour.utilities import tsplit, tstack

def strip_alpha(rgba):
    r, g, b, a = tsplit(rgba)
    return tstack((r, g, b))

def main():
    if len(sys.argv) < 3:
        print(f"Usage:  python3 {sys.argv[0]} <folder1> <folder2> <threshold>")
        print("    folder1: first folder to compare")
        print("    folder1: first folder to compare")
        exit(1)
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    thresh = float(sys.argv[3])
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)
    if (files1.sort() != files2.sort()):
        print("Error: directory contents do not match")
        exit(1)
    print()
    for file in files1:
        if not file.startswith('.'):
            print(file + ":")
            img1 = read_image(os.path.join(path1, file))
            img2 = read_image(os.path.join(path2, file))
            if img1.shape[-1] == 4:
                img1 = strip_alpha(img1)
            if img2.shape[-1] == 4:
                img2 = strip_alpha(img2)
            maxDiff = np.max(np.abs(img1 - img2))
            threshCount = np.count_nonzero(np.abs(img1 - img2) > thresh)
            print("   Max difference:", maxDiff)
            print("   {} pixels difference > {}".format(threshCount, thresh))
            print()

if __name__ == "__main__":
    main()