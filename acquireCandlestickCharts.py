# David McCormick - DTM190000

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yfinance as yf

import DIRHELP

import talib

def saveCandlestickPattern(hist, numCandles, pattern, idx):
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
    plt.savefig(name, bbox_inches = "tight")

    # Closes the figure
    plt.close('all')
    matplotlib.use('Agg')

def acquire():

    # Reads the list of tickers into an iterable list
    listTickets = pd.read_csv("tickerList.csv")
    listTickets = listTickets['Stocks'].values.tolist()

    for ticker in listTickets:
        print(ticker)
        # Sets the ticker
        current = yf.Ticker(ticker)

        # Opens the historical data as a df
        hist = pd.DataFrame(current.history(period="max"))#pd.read_csv("historical.csv")

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

        # Populates the DF with the historical data with each candlestick pattern
        for candle in candle_names:
            hist[candle] = getattr(talib, candle)(op, hi, lo, cl)

        # Detects when a candle pattern has been identified and saves the pattern as an image
        for idx, val in hist.iterrows():
            if (val['CDL3BLACKCROWS'] != 0):
                saveCandlestickPattern(hist, 3, 'CDL3BLACKCROWS', idx)
            if (val['CDL3LINESTRIKE'] != 0):
                if (val['CDL3LINESTRIKE'] == 100):
                    saveCandlestickPattern(hist, 3, 'CDL3LINESTRIKEBULL', idx)
                elif (val['CDL3LINESTRIKE'] == -100):
                    saveCandlestickPattern(hist, 3, 'CDL3LINESTRIKEBEAR', idx)
            if (val['CDL3WHITESOLDIERS'] != 0):
                saveCandlestickPattern(hist, 3, 'CDL3WHITESOLDIERS', idx)
            if (val['CDL2CROWS'] != 0):
                saveCandlestickPattern(hist, 3, 'CDL2CROWS', idx)
            if (val['CDLABANDONEDBABY'] != 0):
                if (val['CDLABANDONEDBABY'] == 100):
                    saveCandlestickPattern(hist, 3, 'CDLABANDONEDBABYBULL', idx)
                elif (val['CDLABANDONEDBABY'] == -100):
                    saveCandlestickPattern(hist, 3, 'CDLABANDONEDBABYBEAR', idx)
            if (val['CDLEVENINGSTAR'] != 0):
                saveCandlestickPattern(hist, 3, 'CDLEVENINGSTAR', idx)
            if (val['CDLSTICKSANDWICH'] != 0):
                saveCandlestickPattern(hist, 3, 'CDLSTICKSANDWICH', idx)

    exit(0)