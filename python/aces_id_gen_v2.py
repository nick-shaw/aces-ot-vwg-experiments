from tkinter import *
from tkinter import ttk

import subprocess

# Mac only clipboard hack
def copy2clip(txt):
    cmd='echo "'+txt.strip()+'"|pbcopy'
    try:
       o = subprocess.check_call(cmd, shell=True)
    except:
        pass

def id2clipboard(*args):
    copy2clip(acesId.get())

# EOTF name mappings
eotfs = {
    'BT.1886':'BT1886',
    'sRGB':'sRGB',
    'Gamma 2.2':'Gamma2pt2',
    'Gamma 2.6':'Gamma2pt6',
    'ST.2084':'ST2084',
    'HLG':'HLG',
    'Linear':'Linear'}

Rec709 = [
    [0.6400, 0.3300],
    [0.3000, 0.6000],
    [0.1500, 0.0600],
]

Rec2020 = [
    [0.7080, 0.2920],
    [0.1700, 0.7970],
    [0.1310, 0.0460],
]

P3 = [
    [0.6800, 0.3200],
    [0.2650, 0.6900],
    [0.1500, 0.0600],
]

XYZ = [
    [1, 0],
    [0, 1],
    [0, 0]
]

primaries = {
    "Rec.709" : 0,
    "Rec.2020" : 1,
    "P3" : 2,
    "XYZ": 3}

primaries_list = [Rec709, Rec2020, P3, XYZ]

D65 = [0.3127, 0.3290]
D60 = [0.32168, 0.33767]
DCI = [0.314,  0.351]
EE = [1/3, 1/3]

whites = {
    "D65" : 0,
    "D60" : 1,
    "DCI" : 2,
    "E" : 3}

whites_list = [D65, D60, DCI, EE]

def get_white_name(primaries):
    # Look up white name from chromaticities
    # This is overcomplicated, but has not been changed from the original code where arbitrary chromaticities coudl be entered
    if primaries[3] == D65:
        white_name = "D65"
    elif primaries[3] == D60:
        white_name = "D60"
    elif primaries[3] == DCI:
        white_name = "DCI"
    elif primaries[3] == EE:
        white_name = "E"
    return white_name

def get_primary_name(primaries):
    # Look up primary name from chromaticities
    # This is overcomplicated, but has not been changed from the original code where arbitrary chromaticities coudl be entered
    if primaries[:3] == Rec709:
        primary_name = "Rec709"
    elif primaries[:3] == Rec2020:
        primary_name = "Rec2020"
    elif primaries[:3] == P3:
        primary_name = "P3"
    elif primaries[:3] == XYZ:
        primary_name = "XYZ"
    white_name = get_white_name(primaries)
    if primary_name[:3] == "Rec" and white_name == "D65":
        white_name = ""
    if primary_name[:3] == "XYZ" and white_name == "E":
        white_name = ""
    return "{}{}".format(primary_name, white_name)

def generate(*args):
    # Get chromaticities for limiting primaries
    limit = primaries_list[primaries[limitingPrimaries.get()]].copy()
    # Add limiting white point chromaticities
    limit.append(whites_list[whites[limitingWhite.get()]])
    # Get chromaticities for encoding primaries
    encoding = primaries_list[primaries[encodingPrimaries.get()]].copy()
    # Add encoding white point chromaticities
    encoding.append(whites_list[whites[encodingWhite.get()]])
    # Get peak luminance and cast to int
    peakLuminance = int(float(peak.get()))
    # Map EOTF name to ID version
    eotfName = eotfs[eotf.get()]
    # Combine primary and white names, dropping white name if it is the standard white for the primaries
    limitName = get_primary_name(limit)
    encodingName = get_primary_name(encoding)
    # If names do not match, add D**limited to limitName
    if limitName != encodingName:
        limitName = "_" + limitName + "limited"
    else:
        limitName = ""
    if limitingWhite.get() == encodingWhite.get(): # Matching whites are not a "sim"
        sim = ""
    elif encodingPrimaries.get() != "XYZ": # XYZ encodings are strictly always a "sim" but we just refer to what they are limited to
        sim = "_" + limitingWhite.get() + "sim"
        if limitingPrimaries.get() == encodingPrimaries.get(): # Don't describe something as both ***D**limited and **sim
            limitName = ""
        else:
            limitName = "_{}limited".format(limitingPrimaries.get().replace(".", ""))
    else:
        sim = ""
    id = "{}_{}{}{}_{}nit".format(encodingName, eotfName, limitName, sim, peakLuminance)
    if explicit.get() == "implicit": # List of implicit special cases
        id = id.replace("Rec709_Gamma2pt2", "sRGB_Gamma2pt2")
        id = id.replace("Rec709_sRGB", "sRGB")
        id = id.replace("Rec709_BT1886", "Rec709")
        id = id.replace("P3D65_sRGB", "DisplayP3")
        id = id.replace("XYZ_Gamma2pt6", "DCDM")
        if encodingName[:2] == "P3" and eotfName == "Gamma2pt6":
            id = id.replace("Gamma2pt6_", "")
    # Special cases for DCDM even if "explicit" is selected
    id = id.replace("XYZ_Gamma2pt6", "DCDM_Gamma2pt6")
    id = id.replace("XYZ_ST2084", "DCDM_ST2084")
    # Is it a forward or inverse transform?
    if inverse.get() == 'inverse':
        style = "InvOutput"
    else:
        style = "Output"
    # Put ID together
    aces_id = "<ACEStransformID>urn:ampas:aces:transformId:v2.0:{}.Academy.{}.a2.v1</ACEStransformID>".format(style, id)
    print(aces_id)
    # Show ACES ID and chromaticities
    acesId.set(aces_id)
    redLimitX.set(limit[0][0])
    redLimitY.set(limit[0][1])
    greenLimitX.set(limit[1][0])
    greenLimitY.set(limit[1][1])
    blueLimitX.set(limit[2][0])
    blueLimitY.set(limit[2][1])
    whiteLimitX.set(limit[3][0])
    whiteLimitY.set(limit[3][1])
    redEncodingX.set(encoding[0][0])
    redEncodingY.set(encoding[0][1])
    greenEncodingX.set(encoding[1][0])
    greenEncodingY.set(encoding[1][1])
    blueEncodingX.set(encoding[2][0])
    blueEncodingY.set(encoding[2][1])
    # Replace float value of 1/3 with the string "1/3"
    if encoding[3][0] == 1/3:
        whiteEncodingX.set("1/3")
    else:
        whiteEncodingX.set(encoding[3][0])
    if encoding[3][1] == 1/3:
        whiteEncodingY.set("1/3")
    else:
        whiteEncodingY.set(encoding[3][1])

def change_encoding(*args):
    # Restrict encoding white options based on encoding primaries
    if encodingPrimaries.get() == "XYZ":
        # Set to SDR DCDM settings when XYZ encoding selected
        encoding_white_entry["values"] = ["D65", "E"]
        eotf.set("Gamma 2.6")
        encodingWhite.set("E")
    elif encodingPrimaries.get() == "P3":
        encoding_white_entry["values"] = ["D65", "D60", "DCI"]
        encodingWhite.set(limitingWhite.get()) # Ensure a valid white is selected (in case it was previously set to "E")
    else:
        encoding_white_entry["values"] = ["D65", "D60"]
        if encodingPrimaries.get()[:3] == "Rec":
            encodingWhite.set("D65") # Standard white for Rec.709 and Rec.2020
    generate()

# User Interface
root = Tk()
root.title("ACES Output Transform ID Generator")
root.geometry("960x320")
root.resizable(False, False)

mainframe = ttk.Frame(root, padding="5 5")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

inverse = StringVar()
inverse_entry = ttk.Checkbutton(mainframe, text='Inverse', variable=inverse,
        command=generate, onvalue='inverse', offvalue='forward')
inverse_entry.grid(column=1, row=1,)
inverse.set("forward")

explicit = StringVar()
explicit_entry = ttk.Checkbutton(mainframe, text='Explicit', variable=explicit,
        command=generate, onvalue='explicit', offvalue='implicit')
explicit_entry.grid(column=2, row=1,)
explicit.set("explicit")

peak = StringVar()
peak_entry = ttk.Entry(mainframe, width=10, textvariable=peak)
peak_entry.grid(column=3, row=1, sticky=(W, E))
peak.set(100)

eotf = StringVar()
eotf_entry = ttk.Combobox(mainframe, width=10, textvariable=eotf)
eotf_entry["values"] = ["BT.1886", "sRGB", "Gamma 2.2", "Gamma 2.6", "ST.2084", "HLG", "Linear"]
eotf_entry.state(["readonly"])
eotf_entry.grid(column=5, row=1, sticky=W)
eotf_entry.bind('<<ComboboxSelected>>', generate)
eotf.set("BT.1886")

limitingPrimaries = StringVar()
limiting_primaries_entry = ttk.Combobox(mainframe, width=10, textvariable=limitingPrimaries)
limiting_primaries_entry["values"] = ["Rec.709", "Rec.2020", "P3"]
limiting_primaries_entry.state(["readonly"])
limiting_primaries_entry.grid(column=3, row=2, sticky=W)
limiting_primaries_entry.bind('<<ComboboxSelected>>', generate)
limitingPrimaries.set("Rec.709")

encodingPrimaries = StringVar()
encoding_primaries_entry = ttk.Combobox(mainframe, width=10, textvariable=encodingPrimaries)
encoding_primaries_entry["values"] = ["Rec.709", "Rec.2020", "P3", "XYZ"]
encoding_primaries_entry.state(["readonly"])
encoding_primaries_entry.grid(column=5, row=2, sticky=W)
encoding_primaries_entry.bind('<<ComboboxSelected>>', change_encoding)
encodingPrimaries.set("Rec.709")

limitingWhite = StringVar()
limiting_white_entry = ttk.Combobox(mainframe, width=10, textvariable=limitingWhite)
limiting_white_entry["values"] = ["D65", "D60"]
limiting_white_entry.state(["readonly"])
limiting_white_entry.grid(column=3, row=3, sticky=W)
limiting_white_entry.bind('<<ComboboxSelected>>', generate)
limitingWhite.set("D65")

encodingWhite = StringVar()
encoding_white_entry = ttk.Combobox(mainframe, width=10, textvariable=encodingWhite)
encoding_white_entry["values"] = ["D65", "D60"]
encoding_white_entry.state(["readonly"])
encoding_white_entry.grid(column=5, row=3, sticky=W)
encoding_white_entry.bind('<<ComboboxSelected>>', generate)
encodingWhite.set("D65")

redLimitX = StringVar()
ttk.Label(mainframe, textvariable=redLimitX).grid(column=2, row=5)
redLimitY = StringVar()
ttk.Label(mainframe, textvariable=redLimitY).grid(column=3, row=5)
greenLimitX = StringVar()
ttk.Label(mainframe, textvariable=greenLimitX).grid(column=2, row=6)
greenLimitY = StringVar()
ttk.Label(mainframe, textvariable=greenLimitY).grid(column=3, row=6)
blueLimitX = StringVar()
ttk.Label(mainframe, textvariable=blueLimitX).grid(column=2, row=7)
blueLimitY = StringVar()
ttk.Label(mainframe, textvariable=blueLimitY).grid(column=3, row=7)
whiteLimitX = StringVar()
ttk.Label(mainframe, textvariable=whiteLimitX).grid(column=2, row=8)
whiteLimitY = StringVar()
ttk.Label(mainframe, textvariable=whiteLimitY).grid(column=3, row=8)

redEncodingX = StringVar()
ttk.Label(mainframe, textvariable=redEncodingX).grid(column=4, row=5)
redEncodingY = StringVar()
ttk.Label(mainframe, textvariable=redEncodingY).grid(column=5, row=5)
greenEncodingX = StringVar()
ttk.Label(mainframe, textvariable=greenEncodingX).grid(column=4, row=6)
greenEncodingY = StringVar()
ttk.Label(mainframe, textvariable=greenEncodingY).grid(column=5, row=6)
blueEncodingX = StringVar()
ttk.Label(mainframe, textvariable=blueEncodingX).grid(column=4, row=7)
blueEncodingY = StringVar()
ttk.Label(mainframe, textvariable=blueEncodingY).grid(column=5, row=7)
whiteEncodingX = StringVar()
ttk.Label(mainframe, textvariable=whiteEncodingX).grid(column=4, row=8)
whiteEncodingY = StringVar()
ttk.Label(mainframe, textvariable=whiteEncodingY).grid(column=5, row=8)

ttk.Label(mainframe, text="EOTF").grid(column=4, row=1, sticky=E)
ttk.Label(mainframe, text="Limiting Primaries").grid(column=2, row=2, sticky=E)
ttk.Label(mainframe, text="Encoding Primaries").grid(column=4, row=2, sticky=E)
ttk.Label(mainframe, text="Limiting White").grid(column=2, row=3, sticky=E)
ttk.Label(mainframe, text="Encoding White").grid(column=4, row=3, sticky=E)
ttk.Label(mainframe, text="x").grid(column=2, row=4)
ttk.Label(mainframe, text="y").grid(column=3, row=4)
ttk.Label(mainframe, text="x").grid(column=4, row=4)
ttk.Label(mainframe, text="y").grid(column=5, row=4)

ttk.Button(mainframe, text="Copy to Clipboard", command=id2clipboard).grid(column=1, row=11, columnspan=6)

for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=2)

acesId = StringVar()
ttk.Label(mainframe, textvariable=acesId).grid(column=1, row=10, padx="20 5", pady="20 5", columnspan=6)

ttk.Label(mainframe, text="nit peak").grid(column=4, row=1, padx="0 5", sticky=W)
ttk.Label(mainframe, text="Red").grid(column=1, row=5, padx="64 5", sticky=E)
ttk.Label(mainframe, text="Green").grid(column=1, row=6, padx="64 5", sticky=E)
ttk.Label(mainframe, text="Blue").grid(column=1, row=7, padx="64 5", sticky=E)
ttk.Label(mainframe, text="White").grid(column=1, row=8, padx="64 5", sticky=E)

generate()
peak_entry.focus()
root.bind("<Return>", generate)

root.mainloop()