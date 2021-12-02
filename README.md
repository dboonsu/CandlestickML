# Candlestick Machine Learning Guide
#### David McCormick (DTM190000)
#### CS 4375.001
This README should help you navigate the Candlestick ML project

## Installation

```bash
git clone https://https://github.com/dboonsu/CandlestickML/
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all of the necessary files

```bash
pip install opencv-python
pip install os
pip install numpy
pip install pandas
pip install yfinance
```
Additionally, you should navigate to this [website](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and download the appropriate version of TA-Lib and move it into the directory.
```bash
python --version
Python 3.9.7

pip install TA_Lib-0.4.21-cp39-cp39-win_amd64.whl
```
## Usage

There are 4 different code pathways.

One to reset the IMGDIR (delete all candlestick images and remake the IMGDIR directory

Two to acquire new candlesticks in the IMGDIR

Three to test trained models (note that you must copy and paste the kernel1, kernel2, and weights
into the testModel.py file

Four to train the model, just leave the three conditional statements at 0

```python
    # 1 if you want to clear all the files in IMGDIR and remake the IMGDIR
    if (0):
        DIRHELP.reset()

    # 1 to read and import candlestick charts
    if (0):
        acquireCandlestickCharts.acquire()

    # 1 for testing trained models
    if (0):
        testModel.test()

    # Stores all of the images in Mx28x28
    #  M is the total number of images
    #  28x28 is the number of pixels in each image
    images=np.zeros((28,28))...
```
## Data
The data for the images has been compressed into "images.csv" and "labels.csv".
It gets reshaped from 2D into 3D properly. Additionally, it saves you from downloading the entirety of the images.
The whole folder for the data can be found [here](https://github.com/dboonsu/CandlestickData).
