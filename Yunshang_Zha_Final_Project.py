
# coding: utf-8

# # Implied Volatility

# The aim of this project is to find the implied volatility and find its relation with maturity and Strike

# ## 1. Background

# A forward-looking measure of volatility, implied volatility is used with option pricing models, such as Black-Scholes-Merton model and the current market price.
# <br><br>
# Black, F., and M. Scholes, 1973, “The Pricing of Options and Corporate Liabilities,” Journal of Political Economy 81, 637-659.
# <br><br>
# It is the expected future volatility of the underlying asset over the remaining life of the option.

# ## 2. Methodology

# Hypothesize that over a period of three expries the implied volatility will exhibit an increasing function of maturity and will give a smile in the plot of implied volatilities and strike
# <br><br>
# My plan for this project
# <ol>
#  <li>Collect the option chain AAPL from Yahoo! Finance over three expiries </li>
#  <li>Compute the implied volatility using Black-Scholes and Newton's method to approximate </li>
#  <li>Plot volatility against strike </li>
#  <li>If there is no volatility smile, analyze it and possibly plot volatility surface </li>
# </ol>

# In[1]:


from scipy.stats import norm
from math import sqrt, exp, log, pi
import pandas as pd
import matplotlib.pyplot as plt
from array import array
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# ## 3. Black-Scholes equation

# We are going to start up with defining the Black-Scholes functions that will give us call price and the put price

# In[2]:


def d(sigma, S, K, r, t):
    d1 = 1 / (sigma * sqrt(t)) * ( log(S/K) + (r + sigma**2/2) * t)
    d2 = d1 - sigma * sqrt(t)
    return d1, d2

def call_price(sigma, S, K, r, t, d1, d2):
    C = norm.cdf(d1) * S - norm.cdf(d2) * K * exp(-r * t)
    return C

def put_price(sigma, S, K, r, t, d1, d2):
    P = norm.cdf(-d2) * K * exp(-r * t) - norm.cdf(-d1) * S
    return P


# In the function, S is the spot price; K is the strike; C is the predected call price; and P is the predected put price; r is the risk-free rate, where we will use the data from Treasury Bill Rate; price will be the derived from option chain

# ## 4. Newton's Method (Newton-Raphson)

# Now to compute the implied volatility, we will apply Newton's method and start with an assumption implied volatility 0.5. We will iterate as many as we need untill 1000 times and use 1.0e-5 as our precision.

# In[3]:


def getVol(S,K,r,t, price):
    vol = 0.5
    epsilon = 1.0 
    precision = 1.0e-5
    i = 0

    while epsilon > precision:
        if i > 1000:
            break

        i = i + 1
        orig = vol
        d1, d2 = d(vol, S, K, r, t)
        function_value = call_price(vol, S, K, r, t, d1, d2) - price
        vega = S * norm.pdf(d1) * sqrt(t)
        vol = -function_value/vega + vol
        epsilon = abs(function_value)

    return vol


# ## 5. Implied Volatility

# Initially it was planned to get the option chain using Yahoo API, but they have retired their service, so I had to grab data from the web page and make csv files

# In[4]:


data = pd.read_csv("AAPL1221C.csv")


# In[5]:


t = (pd.to_datetime('2018-12-21') - pd.to_datetime('today')).days/365
for i in range(data.shape[0]): 
    p =(float(data.Ask[i])+float(data.Bid[i]))/2.0
    x = getVol(168.49,data.Strike[i],0.0002,t,p)
    plt.scatter(data.Strike[i],x)
    plt.ylabel("Implied Volatility")
    plt.xlabel("Strike")


# In the plot a bove, we used the latest spot price, 168.49, with 0.0002 interest rate. The data we used was based on the option chain from December 21. Now let's try to get some option data from another time to test out maturity

# In[6]:


data1 = pd.read_csv("AAPL1228C.csv")


# In[7]:


t1 = (pd.to_datetime('2018-12-28') - pd.to_datetime('today')).days/365
for i in range(data1.shape[0]):   
    p1 =(float(data1.Ask[i])+float(data1.Bid[i]))/2.0
    x1 = getVol(168.49,data1.Strike[i],0.0002, t1 ,p1)
    plt.scatter(data1.Strike[i],x1)
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")


# From the plot above, we can see that the plot got flatten. The implied volatility around 170 strike price is a lot smaller than that from December 21, 7 days younger in maturity.

# Now let's test out the implied volatility from a younger volatility

# In[8]:


data2 = pd.read_csv("AAPL1214C.csv")


# In[9]:


t2 = (pd.to_datetime('2018-12-14') - pd.to_datetime('today')).days/365
for i in range(data2.shape[0]):  
    p =(float(data2.Ask[i])+float(data2.Bid[i]))/2.0
    x2 = getVol(168.49,data2.Strike[i],0.0002,t2,p)
    plt.scatter(data2.Strike[i],x2)
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")


# It is really obvious from the plot above, that the smile is pretty perfect, more curvy than the rest of the plots we have. Thus proved our assumption

# In the industry, call options are used to calculate the implied votilities for Strike that is bigger than the spot prices, and vice versa for puts. So let's use the Black-Schole to caculatet the implied volatility for puts

# In[10]:


def getpVol(S,K,r,t, price):
    vol = 0.5
    epsilon = 1.0 
    precision = 1.0e-5
    i = 0

    while epsilon > precision:
        if i > 1000:
            break

        i = i + 1
        orig = vol
        d1, d2 = d(vol, S, K, r, t)
        function_value = put_price(vol, S, K, r, t, d1, d2) - price
        vega = S * norm.pdf(d1) * sqrt(t)
        vol = -function_value/vega + vol
        epsilon = abs(function_value)

    return vol


# In[11]:


data4 = pd.read_csv("AAPL1221P.csv")


# In[12]:


t1 = (pd.to_datetime('2018-12-21') - pd.to_datetime('today')).days/365
for i in range(data4.shape[0]):   
    p1 =(float(data4.Ask[i])+float(data4.Bid[i]))/2.0
    x1 = getpVol(168.49,data4.Strike[i],0.0002, t1 ,p1)
    plt.scatter(data4.Strike[i],x1, c = 'green')
    for i in range(data.shape[0]): 
        p =(float(data.Ask[i])+float(data.Bid[i]))/2.0
        x = getVol(168.49,data.Strike[i],0.0002,t,p)
        plt.scatter(data.Strike[i],x, c = 'purple')
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")


# The graph above is enhanced by combining both call option values and puts, the graph looks more complete and perfected the smil

# ## 5. Maturity and Implied Volatility

# Now, let's compare the implied volatility from the three different maturities to understand it better. and prepare for a 3D plots

# In[13]:


from scipy.interpolate import griddata
from numpy import *

def make_surf(X,Y,Z):
    XX,YY = meshgrid(linspace(min(X),max(X),230),linspace(min(Y),max(Y),230))
    ZZ = griddata(array([X,Y]).T,array(Z),(XX,YY), method='linear')
    return XX,YY,ZZ

fig = plt.figure()
ax = plt.axes(projection='3d')

def plot3D(X,Y,Z):
    fig = plt.figure()
    ax = Axes3D(fig, azim = -29, elev = 50)
    
    ax.plot(X,Y,Z,'o')
    ax.plot(X,Y,Z,'o', color = 'purple')
    plt.xlabel("maturity")
    plt.ylabel("strike")


# In[14]:


data3 = pd.read_csv("AAPL12C.csv")
vals = []
for i in range(data3.shape[0]):
    x = (pd.to_datetime(data3.Expiry[i]) - pd.to_datetime('today')).days/365
    y = data3.Strike[i]
    p =(float(data3.Ask[i])+float(data3.Bid[i]))/2.0
    z = getVol(168.49,y,0.0002,x,p)
    vals.append([x,float(y),z])

vals = np.array(vals).T
plot3D(vals[0],vals[1],vals[2])


# The graph above shows the three different plots and the implied volatility exhibits an increasing function of maturity

# ## 6. Conclusion

# Our findings strongly support the hypothesis. Our method could be perfected if we use bisection method than Newton and if we could include both calls and puts to plot the implied volatility skew 
