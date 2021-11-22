import os

def reset():
    dir = "IMGDIR/"

    if (not(os.path.isdir("IMGDIR/"))):
        os.mkdir("IMGDIR/")
        os.mkdir("IMGDIR/CDL3BLACKCROWS")
        os.mkdir("IMGDIR/CDL3LINESTRIKEBULL")
        os.mkdir("IMGDIR/CDL3LINESTRIKEBEAR")
        os.mkdir("IMGDIR/CDLEVENINGSTAR")
        os.mkdir("IMGDIR/CDLSTICKSANDWICH")

    for folder in os.scandir(dir):
        for file in os.scandir(folder):
            os.remove(file)