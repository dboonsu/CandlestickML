# David McCormick - DTM190000

import os

def reset():
    dir = "IMGDIR/"

    # Checks if the IMGDIR is there
    #  If not, creates the IMGDIR and all necessary directories
    if (not(os.path.isdir("IMGDIR/"))):
        os.mkdir("IMGDIR/")
        os.mkdir("IMGDIR/CDL2CROWS")
        os.mkdir("IMGDIR/CDL3BLACKCROWS")
        os.mkdir("IMGDIR/CDL3LINESTRIKEBULL")
        os.mkdir("IMGDIR/CDL3LINESTRIKEBEAR")
        os.mkdir("IMGDIR/CDL3WHITESOLDIERS")
        os.mkdir("IMGDIR/CDLABANDONEDBABYBULL")
        os.mkdir("IMGDIR/CDLABANDONEDBABYBEAR")
        os.mkdir("IMGDIR/CDLEVENINGSTAR")
        os.mkdir("IMGDIR/CDLSTICKSANDWICH")

    # Removes all of the images in the directories
    for folder in os.scandir(dir):
        for file in os.scandir(folder):
            os.remove(file)

