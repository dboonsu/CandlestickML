import matplotlib.patches
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import talib

import RESETIMGDIR

def saveCandlestickPattern(hist, numCandles, pattern):
    x = range(0,len(hist))

    # Creates a new matplot figure
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Directly iterates over where the candlesticks are
    for idx_, val_ in hist.loc[idx - (numCandles - 1):idx].iterrows():
        # Plots the low/high lines at on the x axis
        plt.plot([x[idx_], x[idx_]], [val_['Low'], val_['High']], color='black')

        # Determines whether the candle is white or black
        if (val_["Close"] - val_["Open"] > 0):
            curColor = "white"
        else:
            curColor = "black"

        # Adds the candle element to the candle
        rect = matplotlib.patches.Rectangle((x[idx_] + -.1, val_['Open']), .2, (val_['Close'] - val_['Open']),
                                             color=curColor, zorder=10)
        ax.add_patch(rect)

    # Sees how many figures are in the pattern folder
    num = len(os.listdir("IMGDIR/" + pattern))

    # Saves the image as a new file
    name = "IMGDIR/" + pattern + "/" + str(num) + ".jpg"
    plt.savefig(name)

    # Closes the figure
    plt.close(fig)


# def displayPattern(temp, numCandles):
#     x = range(0, len(hist))
#     fig, ax = plt.subplots(nrows=1, ncols=1)
#     ax.set_facecolor((.9, .7, .7))
#     for idx, val in temp.iterrows():
#         for iter in range(0, numCandles):
#             plt.plot([x[iter], x[iter]],
#                      [val['Low ' + str(iter)], val['High ' + str(iter)]],
#                      color='black')
#             if (val["Close " + str(iter)] - val["Open " + str(iter)] > 0):
#                 curColor = "white"
#             else:
#                 curColor = "black"
#             rect1 = matplotlib.patches.Rectangle((x[iter] + -.1, val['Open ' + str(iter)]), .2, (val['Close ' + str(iter)] - val['Open ' + str(iter)]),
#                                                  color=curColor, zorder=10)
#             ax.add_patch(rect1)
#         # plt.show()
#         plt.savefig("C:\\Users\\David\\PycharmProjects\\CandlestickML\\IMG3BLACKCROWS\\test.jpg")
#
#         fig, ax = plt.subplots(nrows=1, ncols=1)
#         ax.set_facecolor((.9, .7, .7))

if __name__ == "__main__":

    # 1 if you want to clear all the files in IMGDIR and remake the IMGDIR
    if (1):
        RESETIMGDIR.reset()
        exit(0)

    # Sets the ticker
    spy = yf.Ticker("SPY")

    # Opens the historical data as a df
    hist = pd.DataFrame(spy.history(period="max"))

    # Drops unnecessary columns
    hist = hist.drop(columns=["Volume", "Dividends", "Stock Splits"])

    # Changes the index from dates to numbers
    hist.reset_index(inplace=True)

    # Identifies candle patterns in historical data
    # Adds several new columns for each pattern
    #  When a pattern is identified, it will add a -100 or 100 for downward trends or positive trends
    #  Otherwise, the value will be 0
    candle_names = talib.get_function_groups()['Pattern Recognition']
    op = hist['Open']
    hi = hist['High']
    lo = hist['Low']
    cl = hist['Close']
    for candle in candle_names:
        hist[candle] = getattr(talib, candle)(op, hi, lo, cl)

    # Detects when a candle pattern has been identified and saves the pattern as an image
    for idx, val in hist.iterrows():
        if (val['CDL3BLACKCROWS'] != 0):
            saveCandlestickPattern(hist, 3, 'CDL3BLACKCROWS')
        if (val['CDL3LINESTRIKE'] != 0):
            if (val['CDL3LINESTRIKE'] == 100):
                saveCandlestickPattern(hist, 3, 'CDL3LINESTRIKEBULL')
            elif (val['CDL3LINESTRIKE'] == -100):
                saveCandlestickPattern(hist, 3, 'CDL3LINESTRIKEBEAR')
        if (val['CDLEVENINGSTAR'] != 0):
            saveCandlestickPattern(hist, 3, 'CDLEVENINGSTAR')
        if (val['CDLSTICKSANDWICH'] != 0):
            saveCandlestickPattern(hist, 3, 'CDLSTICKSANDWICH')
