Adding a Custom ACES IDT or ODT File:
-------------------------------------
- Navigate to the "ACES Transforms" folder in Resolve's main application support folder. Create it if it does not exist.
    - MacOS: "~/Library/Application Support/Blackmagic Design/DaVinci Resolve/ACES Transforms"
    - Windows: "%AppData%\Blackmagic Design\\DaVinci Resolve\\Support\\ACES Transforms"
    - Linux: "~/.local/share/DaVinciResolve/ACES Transforms"
- Place your custom ACES DCTL files for Input Device Transforms (IDTs) in the IDT subfolder.
- Place your custom ACES DCTL files for Output Device Transforms (ODTs) in the ODT subfolder.
- (Re)start Resolve.

Note for Mac users: take care that you use the `Library` folder in your home folder, not the one in the root of your system drive. If in doubt, copy the full path from here and use the ***Go / Go to Folder...*** menu in Finder.