Code current version is v60.

To apply the display rendering on an image:

.. code-block:: bash

    # For SDR sRGB 100 nits
    python apply.py 1 DigitalLAD.2048x1556.exr Out_sRGB_1000nits_DigitalLAD.2048x1556.exr
    # For SDR BT.1886 100 nits
    python apply.py 2 DigitalLAD.2048x1556.exr Out_Rec1886_1000nits_DigitalLAD.2048x1556.exr
    # For HDR Rec2020 1000 nits
    python apply.py 3 DigitalLAD.2048x1556.exr Out_Rec2100_1000nits_DigitalLAD.2048x1556.exr

To test the roundtrip behaviour:

.. code-block:: bash

    # Test on ColourChecker 24 values
    python roundtrip.py 1
    # Test on image
    python roundtrip.py 1 --img Out_sRGB_1000nits_DigitalLAD.2048x1556.exr
    # Test on RGB triplet
    python roundtrip.py 1 --rgb 0.5 0.5 0.5
    # Test on a regular lattice sampling
    python roundtrip.py 1 --cube 2

To test against Nuke, generate references images from the Blink implementation, then use ``nuke_diff.py`` script.
* Match achieved against Nuke's v58 Blink implementation is improved by lowering the binary search treshold for the reach cusp table to 1e-4.

This is a work in progress.