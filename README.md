# ACES Output Transforms VWG Experiments
This repo contains various experiments related to the ACES 2.0 [Output Transforms Virtual Working Group](https://paper.dropbox.com/doc/Output-Transforms-Architecture-Virtual-Working-Group--BHNkZoNAA~9dfXH1BcmddBLaAg-HKNpj824NA0Z8tn7jiPS0).

## DCTL
* `ACES_LIB_MOD.h` – a modified version of the DCTL conversion of the ACES CTL library by [Paul Dore](https://github.com/baldavenger/ACES_DCTL)

* `SSTS_OT.dctl` – an implementation of the SSTS Output Transform using the above library, with user switches to disable the various "RRT sweeteners"

* `HLG_to_PQ.dctl` – a modified version of the HLG to PQ conversion from the CTL library with `L_b` and `L_w` user controllable, rather than fixed at 0 and 1000.
