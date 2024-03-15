# ACES Output Transforms VWG Experiments
This repo contains various experiments related to the ACES 2.0 [Output Transforms Virtual Working Group](https://paper.dropbox.com/doc/Output-Transforms-Architecture-Virtual-Working-Group--BHNkZoNAA~9dfXH1BcmddBLaAg-HKNpj824NA0Z8tn7jiPS0).

## DCTL
* `ACES_LIB_MOD.h` – a modified version of the DCTL conversion of the ACES CTL library by [Paul Dore](https://github.com/baldavenger/ACES_DCTL) for use with the SSTS_OT DCTL

* `SSTS_OT.dctl` – an implementation of the SSTS Output Transform using the above library, with user switches to disable the various "RRT sweeteners" and alter the order of operations.

* `HLG_to_PQ.dctl` – a modified version of the HLG to PQ conversion from the CTL library with `L_b` and `L_w` user controllable, rather than fixed at 0 and 1000.

* `K1S1_LMT_ACEScct_SSTS.cube` - an LMT, to be applied in ACEScct, to emulate the ARRI K1S1 under the Simple SSTS based DRT.

* `K1S1_LMT_ACEScct_v4.cube` - an LMT, to be applied in ACEScct, to emulate the ARRI K1S1 under the Naive DRT.

* `Naive_DRT.dctl` - A naive chromaticity preserving DRT based on the Jed/Daniele tone mapper.

* `Naive_DRT_K1S1.drx` - A Resolve saved grade to set up the Naive DRT and corresponding K1S1 LMT.

* `Simple_DRT_SSTS.dctl` - A naive chromaticity preserving DRT based on the SSTS tone mapper.

* `Simple_SSTS_DRT_K1S1.drx` - A Resolve saved grade to set up the simple SSTS DRT and corresponding K1S1 LMT.

* `ZCAM.dctl` - A basic implementation of ZCAM XYZ to IzMh / JMh transform (not the whole DRT yet) based on the Nuke version by [Matthias Scharfenberg](https://github.com/Tristimulus/aces_vwg_output_transform/blob/master/DRT_ZCAM_IzMh_v02_Blink.nk)

* `ZCAM_DRT.dctl` - A more complete version of [Matthias Scharfenberg](https://github.com/Tristimulus/aces_vwg_output_transform/blob/master/DRT_ZCAM_IzMh_v07_Blink.nk)'s ZCAM based DRT. The inverse transform is implemented, but is not quite right, and needs further work.

* `tonecurves.dctl` - A DCTL for comparing the results of the SSTS and "Daniele Curve" on luminance only

* `ssts_lib.h` - A cutdown of the functions from `ACES_LIB_MOD` including only the SSTS curve, for use in `tonecurves.dctl`

* `DRT_v30_709.dctl` - A port of v30 of the Hellwig CAM based DRT targeting Rec.709 gamma 2.4.

* `DRT_v30_709_sim_in_PQ.dctl` - The same SDR Rec.709 DRT as above, but encoded as BT.2020 PQ to simplify HDR/SDR comparison without needing to change monitor settings.

* `DRT_v30_2100_PQ.dctl` - A 1000 nit P3-D65 limited version of the above DRT encoded as BT.2020 PQ.

* `hellwig_lib.h` - A library of functions and constants used by the above DRTs, to reduce code duplication.

* `DRT_v55_709.dctl` - Pure DCTL implementation of the version 55 DRT, Rec.709 100 nits

* `DRT_v55_P3_D65.dctl` - Pure DCTL implementation of the version 55 DRT, P3-D65 2.6 gamma 48 nits

* `DRT_v55_PQ500.dctl` - Pure DCTL implementation of the version 55 DRT, 500 nit PQ limited to P3-D65 but encoded as Rec.2020

* `DRT_v55_PQ1000.dctl` - Pure DCTL implementation of the version 55 DRT, 1000 nit PQ limited to P3-D65 but encoded as Rec.2020

* `hellwig_lib_v055.h` - Library functions used by the above v55 DRTs

* `ACES Transforms/ODT/DCTL_v55/` - Copies of the above DCTL files with settings and tagging for use in Resolve's `ACES Transforms/ODT/` folder.

* `ACES Transforms/IDT/DCTL_v55/` - Copies of the above DCTL files with settings (including "invert") and tagging for use in Resolve's `ACES Transforms/IDT/` folder.

To load the **Simple** and **Naive** DRTs through the DRX files, all the *.dctl*, *.cube* and *.h* files must be placed in a folder called `AMPAS` in the root of the Resolve LUT folder. Resolve should be in DaVinci YRGB mode, with the timeline colour space set to whatever the connected monitor is expecting. The first node in the node tree of the DRX is a conversion from LogC to ACEScct. This will obviously need to be modified if your source is not LogC.

The DRTs include various presets for targets, based on the SSTS parameters in current ACES, and versions of Jed Smith's presets modified for more consistent black levels.

When a preset is chosen, the numerical values shown are ignored. These are only used when "Custom" is selected. Unfortunately DCTL does not permit display of the actual values used by a given preset (look at the DCTL source code for the values) or greying out of fields which are not currently being used.

## NUKE

* `daniele.nk` -  A blink node implementation of the Daniele curve as used in the Tonecurve DCTL.

* `daniele.blink` - The Blink kernel used in the node above

* `JMh_surr_test.nk` - The Hellwig surround test shown in meeting #82

* `XYZ_JMh.blink` - The Blink kernel used in the script above

* `hellwig_ach.nk` - Converts achromatic luminance to/from Hellwig J

* `hellwig_ach.blink` - The Blink kernel used in the script above

* `v54_cusp_plot.nk` - A visualisation of a JMh hue slice through the true gamut with an overlay of the approximated one.

## PYTHON

* `cusp_path.py` - Code taken from [Thomas Mansencal's Colab](https://colab.research.google.com/drive/1OerRYxnKOYGhiZEZda1QS93JWBfN_WI0) with additional functions for gamut boundary finding. Also generates a plot of the cusp path and saves the source data for that plot in the `data` folder.

* `cusp_plot.py` - Creates a png sequence of plots of the gamut boundary at different hue angles. Uses functions from `cusp_path.py`.

* `interactive_cusp.py` - An interactive plot of the gamut boundary at a selectable hue value. Includes a skeleton framework for showing the path of gamut compression. The compression currently shown is a simple version, not the full version from the current DRT.

* `v55_init_plot.py` - Python port of the `init()` function from the v55 Blink implementation. Generates the lookup tables for the DCTL, and also plots the variable upper hull gamma.

## PYTHON_NUMPY

* NumPy implementation by Rémi Achard (@remia) of the full v55 DRT.