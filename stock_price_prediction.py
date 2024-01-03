"""
Stock price prediction using a Random Forest Regression model
"""

# Import libraries
import pandas as pd
import yfinance as yf
import talib
from datetime import date
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Get user input, retrieve historical data, and initialize DataFrame
ticker = input("Please enter a ticker symbol: ")
df = pd.DataFrame(yf.download(ticker, start="2022-01-01", end=str(date.today())))

# Calculate technical indicators
df["EMA_20"] = talib.EMA(df['Close'], timeperiod=20)
df["RSI_14"] = talib.RSI(df['Close'], timeperiod=14)
macd, signal_line, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df["MACD"] = macd
df = df.dropna()

# Feature selection
feature_col = ["Close", "Volume", "High", "Low", "EMA_20", "RSI_14", "MACD"]
features = df[feature_col]
target = df["Open"]

# Split the test and train data
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and fit the model
regressor = RandomForestRegressor()
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}

grid_search = GridSearchCV(regressor, param_grid, cv=5)
grid_search.fit(feature_train, target_train)
best_regressor = grid_search.best_estimator_
target_predict = best_regressor.predict(feature_test)

# Prediction
features_future = df[feature_col].iloc[-1].values.reshape(1, -1)
predicted_opening_price = best_regressor.predict(features_future)
print(f"Tomorrow's predicted opening price is ${round(predicted_opening_price[0], 2)}")

# Evaluate the model
score = best_regressor.score(feature_test, target_test)
print(f"Accuracy: {(round(score, 3)*100)}%")

# Visualization
df["Forecast"] = np.nan

# Update the DataFrame with the forecasted opening price
next_date = df.index[-1] + pd.DateOffset(days=1)
df.loc[next_date, "Forecast"] = predicted_opening_price[0]

# Visualization
df["Open"].plot(color="lightgreen")
df["Forecast"].plot(color="green", marker='o')
plt.legend(loc="lower left")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
