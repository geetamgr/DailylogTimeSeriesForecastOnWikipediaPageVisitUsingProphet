#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Python
import pandas as pd
from prophet import Prophet


# In[16]:


# Python
df = pd.read_csv("https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv")
df.head()


# ##### We fit the model by initantiating a new Prophet object. Any settings to the forecasting procedure are passed into the constructor. Then you call its fit method and pass in the historical dataframe. Fitting should take 1-5 seconds.

# In[21]:


# Python
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

model = Prophet()
model.fit(df)


# ##### Create a future dataframe, that extends into the future a specified number of days using the helper method :
# Prophet.make_future_dataframe (by default it will also include the date from history)

# In[22]:


# Python
future = model.make_future_dataframe(periods = 365)
future.tail()


# ##### The predict method will assign each row in future a predicted value which is names yhat.  The forecast object in new dataframe includes column yhat as well as columns for component and uncertainity intervals.

# In[23]:


# Python
forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# ##### You can plot the forecast by calling Prophet.plot method and passing your forecast dataframe .

# In[24]:


# Python
fig1 = model.plot(forecast)


# ##### If you want to see the forecast components, you can use the  Prophet.plot_components methos. By default you'll see the trend, yearly seasonanlity and weekly seasonality of the time series. If you include holidays, you'll see those here, too.

# In[25]:


# Python
fig2= model.plot_components(forecast)


# ##### An interactive figure of the forecast and components can be created with plotly. You will need to install plotly 4.0 or above seperately, as it will not by default be installed with prophet. You will also need to install the notebook and ipywidgets packages.

# In[26]:


# Python
from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(model, forecast)


# In[27]:


# Python
plot_components_plotly(model, forecast)

