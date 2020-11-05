# All imports
import alpaca_trade_api as tradeapi
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import deque
import pandas as pd
import scipy as sc
import os
import datetime
import ast
import shutil
import itertools
import talib as tl
import talib.abstract as tab
import mplfinance as mpf
from PIL import Image

# API keys
APCA_API_BASE_URL='https://api.alpaca.markets'
APCA_API_KEY_ID='AKLAN58YF9FZJORV1KHW'
APCA_API_SECRET_KEY='XMjBmk5gZD9UIdtk88vceIWiNkkvxI0oymdOCijZ'

# Init apis
api = tradeapi.REST(APCA_API_KEY_ID,APCA_API_SECRET_KEY,APCA_API_BASE_URL,'v2')
pg = api.polygon

# ********************** Orders **********************
def oco_sell(symbol,qty,stop,target):
    api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='limit',
            time_in_force='gtc',
            order_class='oco',
            stop_loss={'stop_price':stop},
            take_profit={'limit_price':target}
            )

# ********************** Functions *******************

# Returns an Aggs of stock symbol back from number of days supplied to now
def get_aggs(symbol,days=30):

    # Get from and to date
    to = datetime.datetime.today()
    _from = to - datetime.timedelta(days=days)

    # Return Aggs
    return pg.historic_agg_v2(symbol,1,timespan='day',_from=_from,to=to)

# Converts timestamps into datetimes objects
def aggs_to_dates(aggs):

    # Check if empty, return empty
    if len(aggs) == 0:
        return aggs

    # Get all timestamps
    return np.array(list(map(lambda agg: 
        datetime.datetime.fromtimestamp(agg._raw.get('timestamp')/1000),aggs)))

# Wrapper to return both dates and bars from aggs
def dates_bars(aggs):
    return aggs_to_dates(aggs),aggs_to_bars(aggs)

# Create a dictionary from the values returned from polygon (List of Aggv2)
def aggs_to_bars(aggs):

    # Check if empty, return empty
    if len(aggs) == 0: 
        return aggs

    # Turn into dataframe
    df = aggs.df

    # Get column names (which will be keys)
    keys = df.columns

    # Return new dictionary
    return dict(zip(keys,np.array(df.T)))

# Calculates average for all 
def bars_avg(bars):
    return dict(zip(bars.keys(),
        np.mean(np.array(list(bars.values())),axis=1)))

# Returns boolean vector for indices not NAN
def notnan(vec):
    return np.logical_not(np.isnan(vec))

# Takes N vectors (of same length) and returns boolean vector for which they all
# have not NAN
def nnreduct(*vecs):
    return np.vstack(tuple(notnan(vecs))).all(axis=0)

# Plots all the points for potential training
# Takes:
#      1) Symbol to use
#      2) Range to get initial values for
#      3) Range to plot successful training ranges for
#      4) How long to the streak should be. Days in a row it goes up (both decreasing and increasing)
#      5) How much increase the stock had, percent wise
def seek_train(symbol,dayrange,plotrange,i_streak,d_streak,percent_increase):

    # Fetch the data for this symbol
    data = get_aggs(symbol,days=dayrange)

    # If there's not enough days return
    if len(data) < plotrange:
        return

    # Convert to dataframe
    df = data.df

    # Create a numpy array of just the open,high,low,close 
    arr = np.array(df)[:,0:4]

    # Take the mean across every instance (row)
    avg = np.mean(arr,axis=1)

    # Take the backwards difference to get if each day is up or down
    # Remember to account for 1 less index
    backward = avg[1:] - avg[0:-1]

    # ------------- Decreasing -----------------------
    # Indentify indices where it's decreasing, that are sandwiching a batch where it's increasing
    decreasing = backward <= 0
    d_indices = np.arange(1,len(decreasing)+1)[decreasing] # Do from 1 to account for the loss of index

    # Grab pair from this indices where's at least the minimum streak between them (this is increasing streaks)
    i_streaks = d_indices[1:] - d_indices[0:-1]
    d_pairs = np.array([[d_indices[i],d_indices[i+1]] for i,s in enumerate(i_streaks) if s > i_streak])

    # Separate out the starts and ends, and subtract one from d_end (so it falls on an increase)
    d_start,d_end = d_pairs.T if len(d_pairs) > 0 else [np.array([]),np.array([])]
    d_end -= 1
    # ------------------------------------------------------

    # ------------ Increasing ------------------------------
    # Do the same, but for the inverse, indentity where increses sandwich decreases
    increasing = backward > 0
    i_indices = np.arange(1,len(increasing)+1)[increasing] # Do from 1 to account for the loss of index

    # Grab pair from this indices where's at least the minimum streak between them (this is decreasing streaks)
    d_streaks = i_indices[1:] - i_indices[0:-1]
    i_pairs = np.array([[i_indices[i],i_indices[i+1]] for i,s in enumerate(d_streaks) if s > d_streak])

    # Separate out the starts and ends, and subtract one from d_end (so it falls on an increase)
    i_start,i_end = i_pairs.T if len(i_pairs) > 0 else [np.array([]),np.array([])]
    i_end -= 1
    #---------------------------------------------------------------

    # Grab indices where d_start somewhere in i_end, and where there's the min percent increase to for filtering
    bools_meet = np.in1d(d_start,i_end)
    bools_percent = (d_end - d_start)/d_start >= percent_increase
    bools = np.all(np.array([bools_meet,bools_percent]),axis=0)
    d_start = d_start[bools]
    d_end = d_end[bools]

    # *** Begin to build the plot ***

    # Do first plot simply (line plot)
    addplots = [mpf.make_addplot(avg)]

    # If possible, prep for second plot
    if len(d_start) > 0:
        X = np.array([np.nan]*len(avg))
        c = np.array([[0,0,0]]*len(avg))
        
        # Iterate the d_start and d_end to fill in correct values 
        for idx in d_start:
            X[idx] = avg[idx]
            c[idx] = [0,1,0]
        for idx in d_end:
            X[idx] = avg[idx]
            c[idx] = [1,0,0]

        # Create second plot
        addplots.append(mpf.make_addplot(X,type='scatter',color=c,marker='^',markersize=100))

    # Final plot
    mpf.plot(df,type='candle',style='charles',addplot=addplots,savefig='ConfirmPlots/' + symbol + '.png')

    # ----------------------------------------------------------------------------------------------------

    # Now if applicable, between to iterate d_start and build more plots
    rc={
        "axes.labelcolor": "none",
        "font.size": 0,
        "xtick.color": "none",
        "ytick.color": "none",
    }
    mc =  mpf.make_marketcolors(up='g',down='r',inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc,gridstyle='',rc=rc,facecolor='black')

    # Plots for, labelled by first character. 1 for buy, 0 for not buy
    for n,idx in enumerate(d_start):

        # First do the not buy, keep rolling random index until it's not within range of anything in d_start
        # After 5 tries if it's not done, just stop
        tries = 0
        idr = idx
        while tries < 5 and np.any(d_streak+i_streak >= np.abs(idr-d_start)):
            idr = np.random.randint(plotrange,len(data))
            tries += 1

        # If we didn't stop cause failing tries, then make the plot
        if tries < 5:

            # Grab the values to use from the dataframe for plots
            df_sub = df.iloc[idr:idr-plotrange:-1][::-1]

            # Shoudn't happen, but leave for safety
            if len(df_sub) == plotrange:
                destination = 'TrainValid/' if np.random.rand() > 0.2 else 'Test/'
                mpf.plot(df_sub,type='candle',style=s,savefig=destination + '0'  + symbol + str(n) + '.png')

        # Same as before, but for the buy labelli88ng
        df_sub = df.iloc[idx:idx-plotrange:-1][::-1]

        # Only plot if we grabbed everything
        if len(df_sub) == plotrange:
            destination = 'TrainValid/' if np.random.rand() > 0.2 else 'Test/'
            mpf.plot(df_sub,type='candle',style=s,savefig=destination + '1' + symbol + str(n) + '.png')

# Function for generating plots
def gen_images(num_symbols,dayrange=3000,plotrange=45,i_streak=6,d_streak=4,percent_increase=0.05):

    # Get assets, filter, and run
    assets = api.list_assets()
    symbols = [asset.symbol for asset in assets if asset.tradable and asset.status == 'active']
    syms = random.sample(symbols,np.minimum(num_symbols,len(symbols)))
    for sym in syms:
        try:
            seek_train(sym,dayrange,plotrange,i_streak,d_streak,percent_increase)
        except Exception:
            continue

# Function for cropping all images to new bounding box in specified folder
def crop_images(directory,crop_dims):

    # Get absolute path of directory
    path = os.path.abspath(directory)

    # Get all files in directory
    files = os.listdir(path)

    # Iterate all files and crop to new dimension, and change from RGBA if it's not
    for _file in files:
        filepath = path + '/' + _file
        img = Image.open(filepath).convert('RGB').crop(crop_dims)
        img.save(filepath)

