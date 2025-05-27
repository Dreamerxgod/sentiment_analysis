import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def technical_indicators(df):
    df['MA5'] = df['price'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['price'].rolling(window=10, min_periods=1).mean()

    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def load_and_prepare_data():
    df = pd.read_csv('tweets_ilmn_sentiment_output.csv', parse_dates=['date'])
    df = technical_indicators(df)
    df = df.sort_values('date').set_index('date')

    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_idx).ffill()
    return df

def scale_data(data):
    close_scaler = StandardScaler()
    data['scaled_Close'] = close_scaler.fit_transform(data[['price']])

    feature_scaler = StandardScaler()
    features = ['average_sentiment', 'MA5', 'MA10', 'RSI_14']
    data[features] = feature_scaler.fit_transform(data[features])

    return data, close_scaler

def find_best_arima_params(train_data):
    best_aic = np.inf
    best_order = (0, 0, 0)
    exog_features = ['average_sentiment', 'MA5']

    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(train_data['scaled_Close'], exog=train_data[exog_features], order=(p, d, q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order

def predict(model_fit, test_data, close_scaler):
    exog_features = ['average_sentiment', 'MA5']
    forecast = model_fit.get_forecast(steps=len(test_data), exog=test_data[exog_features])

    pred_scaled = forecast.predicted_mean.values.reshape(-1, 1)
    predicted = close_scaler.inverse_transform(pred_scaled)
    actual = close_scaler.inverse_transform(test_data['scaled_Close'].values.reshape(-1, 1))

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)

    actual_diff = np.diff(actual.flatten())
    pred_diff = np.diff(predicted.flatten())
    y_true = (actual_diff > 0).astype(int)
    y_pred = (pred_diff > 0).astype(int)
    f1 = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    return predicted, actual, mse, rmse, mae, f1

def plot_results(test, predicted, actual):
    plt.figure(figsize=(14, 7))
    plt.plot(test.index, actual, label='Actual Price', linewidth=2)
    plt.plot(test.index, predicted, label='Predicted Price', linestyle='--', color='magenta')

    plt.title('ARIMA Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xlim(test.index.min(), test.index.max())
    plt.show()

if __name__ == '__main__':
    df = load_and_prepare_data()
    df, close_scaler = scale_data(df)
    train_size = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    best_order = find_best_arima_params(train_df)
    print(f'Best ARIMA Order: {best_order}')

    model = ARIMA(train_df['scaled_Close'], exog=train_df[['average_sentiment', 'MA5']], order=best_order)
    model_fit = model.fit()

    predicted, actual, mse, rmse, mae, f1 = predict(model_fit, test_df, close_scaler)

    print("\nMetrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"F1 Score: {f1:.2f}")
   

    plot_results(test_df, predicted, actual)
