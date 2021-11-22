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
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor((.9, .7, .7))
    for idx_, val_ in hist.loc[idx - (numCandles - 1):idx].iterrows():
        plt.plot([x[idx_], x[idx_]],
                 [val_['Low'], val_['High']],
                 color='black')
        if (val_["Close"] - val_["Open"] > 0):
            curColor = "white"
        else:
            curColor = "black"
        rect1 = matplotlib.patches.Rectangle((x[idx_] + -.1, val_['Open']), .2, (val_['Close'] - val_['Open']),
                                             color=curColor, zorder=10)
        ax.add_patch(rect1)
    num = len(os.listdir("IMGDIR/" + pattern))
    name = "IMGDIR/" + pattern + "/" + str(num) + ".jpg"
    plt.savefig(name)
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
    if (1):
        RESETIMGDIR.reset()
        exit(0)

    spy = yf.Ticker("SPY")
    hist = pd.DataFrame(spy.history(period="max"))
    hist = hist.drop(columns=["Volume", "Dividends", "Stock Splits"])
    # hist.to_csv("historical.csv")
    # hist = pd.read_csv("historical.csv")
    candle_names = talib.get_function_groups()['Pattern Recognition']
    hist.reset_index(inplace=True)

    CDL3BLACKCROWS = pd.DataFrame()
    CDL3LINESTRIKEBULL = pd.DataFrame()
    CDL3LINESTRIKEBEAR = pd.DataFrame()
    CDLEVENINGSTAR = pd.DataFrame()
    CDLSTICKSANDWICH = pd.DataFrame()

    op = hist['Open']
    hi = hist['High']
    lo = hist['Low']
    cl = hist['Close']

    for candle in candle_names:
        hist[candle] = getattr(talib, candle)(op, hi, lo, cl)
    # Identifies candle patterns in historical data
    # Adds several new columns for each pattern
    #  When a pattern is identified, it will add a -100 or 100 for downward trends or positive trends
    #  Otherwise, the value will be 0

    # Detects when a candle pattern has been identified and
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

    # CDL3BLACKCROWS.to_csv("CDL3BLACKCROWS", mode='a', header=False)
    # CDL3LINESTRIKEBULL.to_csv("CDL3LINESTRIKEBULL", mode='a', header=False)
    # CDL3LINESTRIKEBEAR.to_csv("CDL3LINESTRIKEBEAR", mode='a', header=False)
    # CDLEVENINGSTAR.to_csv("CDLEVENINGSTAR", mode='a', header=False)
    # CDLSTICKSANDWICH.to_csv("CDLSTICKSANDWICH", mode='a', header=False)


    # test = pd.read_csv("CDL3BLACKCROWS")
    # displayPattern(test, 3)

#### BACKUP STUFF ####
# Detects when a candle pattern has been identified and
#     for idx, val in hist.iterrows():
#         # high/low lines
#         # break
#         if (val['CDL3BLACKCROWS'] != 0):
#             saveCandlestickPattern(hist, 3, 'CDL3BLACKCROWS')
#             temp = foundPattern(hist, 3)
#             CDL3BLACKCROWS = CDL3BLACKCROWS.append(temp, ignore_index=True)
#         if (val['CDL3LINESTRIKE'] != 0):
#             temp = foundPattern(hist, 4)
#             if (val['CDL3LINESTRIKE'] == 100):
#                 CDL3LINESTRIKEBULL = CDL3LINESTRIKEBULL.append(temp, ignore_index=True)
#             elif (val['CDL3LINESTRIKE'] == -100):
#                 CDL3LINESTRIKEBEAR = CDL3LINESTRIKEBEAR.append(temp, ignore_index=True)
#             # plt.show()
#         if (val['CDLEVENINGSTAR'] != 0):
#             temp = foundPattern(hist, 3)
#             CDLEVENINGSTAR = CDLEVENINGSTAR.append(temp, ignore_index=True)
#             # displayCandlesticks(hist, 3)
#             # plt.show()
#         if (val['CDLSTICKSANDWICH'] != 0):
#             temp = foundPattern(hist, 3)
#             CDLSTICKSANDWICH = CDLSTICKSANDWICH.append(temp, ignore_index=True)
#             # displayCandlesticks(hist, 3)
#
# def foundPattern(hist, numCandles):
#     temp = {}
#     for idx_, val_ in hist.loc[idx - (numCandles - 1):idx].iterrows():
#         currCandle = idx_ - idx + (numCandles - 1)
#         temp.update({'Open ' + str(currCandle): val_['Open'], 'High ' + str(currCandle): val_['High'],
#                      'Low ' + str(currCandle): val_['Low'], 'Close ' + str(currCandle): val_['Close']})
#     return temp
#
# def displayCandlesticks(hist, numCandles):
#     x = range(0,len(hist))
#     fig, ax = plt.subplots(nrows=1, ncols=1)
#     ax.set_facecolor((.9, .7, .7))
#     for idx_, val_ in hist.loc[idx - (numCandles - 1):idx].iterrows():
#         plt.plot([x[idx_], x[idx_]],
#                  [val_['Low'], val_['High']],
#                  color='black')
#         if (val_["Close"] - val_["Open"] > 0):
#             curColor = "white"
#         else:
#             curColor = "black"
#         rect1 = matplotlib.patches.Rectangle((x[idx_] + -.1, val_['Open']), .2, (val_['Close'] - val_['Open']),
#                                              color=curColor, zorder=10)
#         ax.add_patch(rect1)
#     plt.show()