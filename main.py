import matplotlib.patches
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib

def foundPattern(hist, numCandles):
    temp = {}
    for idx_, val_ in hist.loc[idx - (numCandles - 1):idx].iterrows():
        currCandle = idx_ - idx + (numCandles - 1)
        temp.update({'Open_' + str(currCandle): val_['Open'], 'High_' + str(currCandle): val_['High'],
                     'Low_' + str(currCandle): val_['Low'], 'Close_' + str(currCandle): val_['Close']})
    return temp

def displayCandlesticks(hist, numCandles):
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
    plt.show()


spy = yf.Ticker("IBM")
hist = pd.DataFrame(spy.history(period="max"))
hist = hist.drop(columns=["Volume", "Dividends", "Stock Splits"])
hist.to_csv("historical.csv")
hist = pd.read_csv("historical.csv")
candle_names = talib.get_function_groups()['Pattern Recognition']

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
    print(candle)
    hist[candle] = getattr(talib, candle)(op, hi, lo, cl)
hist.to_csv("historical.csv")

x = range(0,len(hist))
for idx, val in hist.iterrows():
    # high/low lines
    # break
    if (val['CDL3BLACKCROWS'] != 0):
        temp = foundPattern(hist, 3)
        CDL3BLACKCROWS = CDL3BLACKCROWS.append(temp, ignore_index=True)
        # displayCandlesticks(hist, 5)
    if (val['CDL3LINESTRIKE'] != 0):
        temp = foundPattern(hist, 4)
        if (val['CDL3LINESTRIKE'] == 100):
            CDL3LINESTRIKEBULL = CDL3LINESTRIKEBULL.append(temp, ignore_index=True)
        elif (val['CDL3LINESTRIKE'] == -100):
            CDL3LINESTRIKEBEAR = CDL3LINESTRIKEBEAR.append(temp, ignore_index=True)
        # plt.show()
    if (val['CDLEVENINGSTAR'] != 0):
        temp = foundPattern(hist, 3)
        CDLEVENINGSTAR = CDLEVENINGSTAR.append(temp, ignore_index=True)
        # displayCandlesticks(hist, 3)
        # plt.show()
    if (val['CDLSTICKSANDWICH'] != 0):
        temp = foundPattern(hist, 3)
        CDLSTICKSANDWICH = CDLSTICKSANDWICH.append(temp, ignore_index=True)
        # displayCandlesticks(hist, 3)

CDL3BLACKCROWS.to_csv("CDL3BLACKCROWS", mode='a', header=False)
CDL3LINESTRIKEBULL.to_csv("CDL3LINESTRIKEBULL", mode='a', header=False)
CDL3LINESTRIKEBEAR.to_csv("CDL3LINESTRIKEBEAR", mode='a', header=False)
CDLEVENINGSTAR.to_csv("CDLEVENINGSTAR", mode='a', header=False)
CDLSTICKSANDWICH.to_csv("CDLSTICKSANDWICH", mode='a', header=False)