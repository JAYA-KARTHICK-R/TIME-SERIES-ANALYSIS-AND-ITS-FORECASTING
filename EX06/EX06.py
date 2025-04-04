#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# ✅ Load Data
file_path = r"C:\Users\exam\Downloads\FINAL_USO.csv"  # Update path
df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

# ✅ Fix column names (strip spaces)
df.columns = df.columns.str.strip()

# ✅ Ensure "Adjusted Close" column exists
if "Adj Close" not in df.columns:
    print("Error: 'Adj Close' column not found!")
    print("Available columns:", df.columns)
    exit()

# ✅ Define Target Variable
target = "Adj Close"

# ✅ Apply Moving Average Smoothing
df["SMA_10"] = df[target].rolling(window=10).mean()  # 10-day SMA
df["SMA_20"] = df[target].rolling(window=20).mean()  # 20-day SMA

# ✅ Prepare Data for Forecasting
df.dropna(inplace=True)  # Remove NaN values from SMA columns

X = df[["SMA_10", "SMA_20"]]  # Features
y = df[target]  # Target Variable

# ✅ Split Data into Train & Test Sets
train_size = int(len(df) * 0.8)  # 80% train, 20% test
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# ✅ Apply Exponential Smoothing for Forecasting
model = SimpleExpSmoothing(y_train).fit(smoothing_level=0.2, optimized=False)
forecast = model.forecast(len(y_test))  # Predict the same length as test set

# ✅ Align Forecast with Test Data
forecast.index = y_test.index  # Assign same index as test set

# ✅ Plot Forecast vs. Actual Prices
plt.figure(figsize=(12, 5))
plt.plot(y.index, y, label="Actual Price", color="blue", alpha=0.5)
plt.plot(y_test.index, forecast, label="Forecast", color="red", linestyle="dashed")
plt.title("Gold Price Forecast using Moving Average & Exponential Smoothing")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# In[ ]:




