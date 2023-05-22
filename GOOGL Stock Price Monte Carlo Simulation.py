#!/usr/bin/env python
# coding: utf-8

# In[19]:



import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


# In[20]:


df = pd.read_csv('GOOGL.csv')
df.head()


# In[21]:


df.set_index('Date', inplace=True)


# In[22]:


df_2 = df['Close']
df_2


# In[23]:


df_2.plot(figsize=(10,6));


# There are two components to running a Monte Carlo simulation:
# 
# 1) the equation to evaluate
# 
# 2) the random variables for the input
# 
# 
# Lets look in to the formula first to derive the equation to evaluate.
# 
# Today’s Stock Price = Yesterday’s Stock Price x e^r
# 
# where r = periodic daily return

# Monte carlo simulation generates theoretical future ‘r’ values because the rate of return on an asset is a random number. To model the movement and determine possible future ‘r’ values we must use a formula that models random movements. Here we will use a formula of Brownian motion.
# 
# 

# Brownian motion assumes that there are two parts to the random movement. The first is the overall driving force called as ‘drift’ and the second is the random component. Therefore the rate that the asset changes in value each day- the ‘r’ value that the e is raised to can be broken down into two parts-an overall drift and a random stochastic component.
# 
# 

# Amount change in stock Price = Fixed drift rate + Random stochastic variable
# 
# To create a monte carlo simulator to model future outcomes we need to find theses two parts- the drift and the random stochastic component.

# In[24]:


log_returns = np.log(1+df_2.pct_change())


# In[25]:


log_returns.tail()


# In[26]:


log_returns.plot(figsize=(10,6))


# In[27]:


#Drift = Average Daily Return− (Variance​/2)

u = log_returns.mean()
u


# In[28]:


var = log_returns.var()
var


# In[29]:


drift = u-(0.5*var)
drift


# In[30]:


stdev = log_returns.std()
stdev


# In[31]:


norm.ppf(0.95)

#If an event has a 95% chance of occurring, the distance between this event and the mean will be approximately 1.65 standard deviations. we will use the numpy rand function to randomize this component. we want a multidimensional array so we will insert to arguments.


# In[34]:


x = np.random.rand(10,2)


# In[35]:


Z = norm.ppf(np.random.rand(10,2))


# In[36]:


t_intervals = 1000
iterations = 20


# In[37]:


daily_returns = np.exp(np.array(drift)+np.array(stdev)*norm.ppf(np.random.rand(t_intervals, iterations)))


# In[39]:


S0 = df_2.iloc[-1]
S0
#S0= stock price at t day
#S1 = stock price at t+1 day


# In[41]:


price_list = np.zeros_like(daily_returns)


# In[42]:


price_list[0] = S0


# In[43]:


for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1]*daily_returns[t]


# In[44]:


plt.figure(figsize=(10,6))
plt.plot(price_list)


# From the line plots above, we can see the simulated stock prices can spread from about $1000 to $6000. This has given us an idea about the potential price range for the stock based on the same level of volatility. We will do further analysis on our simulated stock prices to gain more insights.
# 
# 

# In[45]:


#lets calculate the Mean of the simulated last prices, Quantile (5%) and Quantile (95%) of the simulated last prices.

print("Expected Price: ", round(np.mean(price_list), 2))
print("Quantile (5%): ", np.percentile(price_list, 5))
print("Quantile (95%): ", np.percentile(price_list, 95))


# From the result above, we can see there is a 5% of probability that the stock price will be below 1059.30 and a 5% of probability the price will be above 387.49. Our expected stock price at the year-end is 1891.07.
# 
# 

# In[ ]:


#https://medium.com/@juee_thete/understanding-monte-carlo-simulation-and-its-implementation-with-python-3ecacb958cd4
#reference for learning monte carlo simulation

