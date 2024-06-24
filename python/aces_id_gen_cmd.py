import sys

primary_list = ["Rec.709", "Rec709", "Rec.2020", "Rec2020", "P3", "XYZ"]
white_list = ["D65", "D60", "DCI", "E"]
eotf_list = ["BT.1886", "BT1886", "sRGB", "Gamma 2.2", "Gamma22", "Gamma 2.6", "Gamma26", "ST.2084", "ST2084", "HLG", "Linear"]

eotfs = {
    'BT.1886':'BT1886',
    'BT1886':'BT1886',
    'sRGB':'sRGB',
    'Gamma 2.2':'Gamma_2pt2',
    'Gamma22':'Gamma_2pt2',
    'Gamma 2.6':'Gamma_2pt6',
    'Gamma26':'Gamma_2pt6',
    'ST.2084':'ST2084',
    'ST2084':'ST2084',
    'HLG':'HLG',
    'Linear':'Linear'}

def get_primary_name(primaries, white):
    primary_name = primaries.replace(".", "")
    if primary_name[:3] == "Rec" and white == "D65":
        white = ""
    if primary_name == "XYZ" and white == "E":
        white = ""
    return "{}{}".format(primary_name, white)

def generate_aces_id(peakLuminance, limitingPrimaries, limitingWhite, encodingPrimaries, encodingWhite, eotf, inverse=False, explicit=True):
    eotfName = eotfs[eotf]
    limitName = get_primary_name(limitingPrimaries, limitingWhite)
    encodingName = get_primary_name(encodingPrimaries, encodingWhite)
    if limitName != encodingName:
        limitName = "_" + limitName + "limited"
    else:
        limitName = ""
    sim = ""
    if limitingWhite == encodingWhite:
        sim = ""
    elif encodingPrimaries != "XYZ":
        sim = "_" + limitingWhite + "sim"
        if limitingPrimaries == encodingPrimaries:
            limitName = ""
        else:
            limitName = "_{}limited".format(limitingPrimaries.replace(".", ""))
    id = "{}_{}{}{}_{}nit".format(encodingName, eotfName, limitName, sim, peakLuminance)
    if not explicit:
        id = id.replace("Rec709_Gamma_2pt2", "sRGB_Gamma_2pt2")
        id = id.replace("Rec709_sRGB", "sRGB")
        id = id.replace("Rec709_BT1886", "Rec709")
    id = id.replace("XYZ_Gamma_2pt6", "DCDM")
    id = id.replace("XYZ_ST2084", "DCDM_ST2084")
    if inverse:
        style = "InvOutput"
    else:
        style = "Output"
    aces_id = "<ACEStransformID>urn:ampas:aces:transformId:v2.0:{}.Academy.{}.a2.v1</ACEStransformID>".format(style, id)
    return aces_id

def print_help():
    print(f"Usage:  python3 {sys.argv[0]} <peakLuminance> <primariesLimit> <whiteLimit> <primariesEncoding> <whiteEncoding> <EOTF> (optional)<inverse> (optional)<explicit>")
    print("Peak luminance")
    print("\tInteger")
    print("Allowed primaries:")
    for primary in primary_list:
        print("\t{}".format(primary))
    print("Allowed whites:")
    for white in white_list:
        print("\t{}".format(white))
    print("Allowed EOTFs:")
    for eotf in eotf_list:
        print("\t{}".format(eotf))
    print("inverse and explicit:")
    print("\tTrue")
    print("\tFalse")

def main():
    if len(sys.argv) < 7:
        print_help()
        exit(1)
    peakLuminance = sys.argv[1]
    primariesLimit = sys.argv[2]
    whiteLimit = sys.argv[3]
    primariesEncoding = sys.argv[4]
    whiteEncoding = sys.argv[5]
    eotf = sys.argv[6]
    if len(sys.argv) > 7:
        inverse = sys.argv[7]
    else:
        inverse = False
    if len(sys.argv) > 8:
        explicit = sys.argv[8]
    else:
        explicit = False
    if (primariesLimit not in primary_list
        or whiteLimit not in white_list
        or primariesEncoding not in primary_list
        or whiteEncoding not in white_list
        or eotf not in eotf_list):
            print_help()
            exit(1)
    else:
        print(generate_aces_id(peakLuminance, primariesLimit, whiteLimit, primariesEncoding, whiteEncoding, eotf, inverse, explicit,))

if __name__ == "__main__":
    main()