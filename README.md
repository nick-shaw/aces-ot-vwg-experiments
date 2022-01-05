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

To load the **Simple** and **Naive** DRTs through the DRX files, all the *.dctl*, *.cube* and *.h* files must be placed in a folder called `AMPAS` in the root of the Resolve LUT folder. Resolve should be in DaVinci YRGB mode, with the timeline colour space set to whatever the connected monitor is expecting. The first node in the node tree of the DRX is a conversion from LogC to ACEScct. This will obviously need to be modified if your source is not LogC.

The DRTs include various presets for targets, based on the SSTS parameters in current ACES, and versions of Jed Smith's presets modified for more consistent black levels.

When a preset is chosen, the numerical values shown are ignored. These are only used when "Custom" is selected. Unfortunately DCTL does not permit display of the actual values used by a given preset (look at the DCTL source code for the values) or greying out of fields which are not currently being used.