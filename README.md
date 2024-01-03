# Stock Price Prediction with Random Forest Regression

This Python project utilizes a Random Forest Regression model to predict stock prices based on historical data and technical indicators. The implementation includes fetching historical stock data, calculating technical indicators, feature selection, model training, prediction, evaluation, and visualization.

## Features

- **Technical Indicators**: Utilizes the `talib` library to calculate technical indicators such as Exponential Moving Average (EMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD).
- **Random Forest Regression**: Implements a Random Forest Regression model from the `scikit-learn` library for predicting stock prices.
- **Grid Search for Hyperparameter Tuning**: Uses Grid Search to find the optimal hyperparameters for the Random Forest Regression model.
- **User Input and Data Retrieval**: Prompts the user for a stock ticker symbol, retrieves historical data using `yfinance`, and initializes a Pandas DataFrame.
- **Visualization**: Displays the actual stock opening prices along with the predicted opening prices using matplotlib.

## Code Structure

### `stock_price_prediction.py`
This file contains the main code for stock price prediction.

### Functions
- `fetch_stock_data(ticker)`: Retrieves historical stock data using `yfinance` and calculates technical indicators.
- `feature_selection(df)`: Selects relevant features for training the Random Forest Regression model.
- `train_model(features, target)`: Initializes, fits, and tunes the Random Forest Regression model using Grid Search.
- `predict_opening_price(model, features)`: Predicts the opening price for a given set of features.
- `evaluate_model(model, features_test, target_test)`: Evaluates the accuracy of the trained model.
- `visualize_prediction(df, predicted_opening_price)`: Displays a plot with actual and forecasted opening prices.

## Usage
1. Import the `stock_price_prediction.py` module.
2. Use the `fetch_stock_data` function to retrieve historical stock data.
3. Apply feature selection using the `feature_selection` function.
4. Train the model with the `train_model` function.
5. Make predictions with the `predict_opening_price` function.
6. Evaluate the model with the `evaluate_model` function.
7. Visualize the predictions using the `visualize_prediction` function.

```python
# Example Usage
from stock_price_prediction import fetch_stock_data, feature_selection, train_model, predict_opening_price, evaluate_model, visualize_prediction

# Fetch stock data and calculate technical indicators
df = fetch_stock_data("AAPL")

# Feature selection
features, target = feature_selection(df)

# Split the test and train data
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
best_regressor = train_model(features_train, target_train)

# Make predictions
predicted_opening_price = predict_opening_price(best_regressor, features_test)

# Evaluate the model
accuracy = evaluate_model(best_regressor, features_test, target_test)
print(f"Model Accuracy: {round(accuracy, 3)*100}%")

# Visualize predictions
visualize_prediction(df, predicted_opening_price)
```

## Dependencies
- `pandas`: For handling and manipulating data.
- `yfinance`: For fetching historical stock data.
- `talib`: For calculating technical indicators.
- `numpy`: For numerical operations.
- `scikit-learn`: For implementing the Random Forest Regression model.
- `matplotlib`: For visualizing stock prices.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
