import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, classification_report
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

    rs = avg_gain / (avg_loss + 1e-10)  #
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


def data():
    data = pd.read_csv('tweets_tsla_sentiment_output.csv', parse_dates=['date'])
    data = technical_indicators(data)
    data = data.sort_values('date').set_index('date')

    date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    data = data.reindex(date_range).ffill()
    return data


def scale_data(data):
    close_scaler = StandardScaler()
    data['scaled_Close'] = close_scaler.fit_transform(data[['price']])

    feature_scaler = StandardScaler()
    features = ['average_sentiment', 'MA5', 'MA10']
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
                    model = ARIMA(train_data['scaled_Close'],
                                  exog=train_data[exog_features],
                                  order=(p, d, q))
                    model_fit = model.fit()

                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                except Exception as e:
                    continue
    return best_order

def predict(model, test_data, close_scaler):
    exog_features = ['average_sentiment', "MA5"]
    forecast = model.get_forecast(steps=len(test_data), exog=test_data[exog_features])

    predicted_scaled = forecast.predicted_mean
    predicted = close_scaler.inverse_transform(predicted_scaled.values.reshape(-1, 1))

    actual = close_scaler.inverse_transform(test_data['scaled_Close'].values.reshape(-1, 1))

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)

    actual_changes = np.diff(actual.flatten())
    predicted_changes = np.diff(predicted.flatten())

    y_true = (actual_changes > 0).astype(int)
    y_pred = (predicted_changes > 0).astype(int)

    f1 = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    return predicted, actual, mse, rmse, mae, f1


def plot_results(train, test, predicted, actual):
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['price'], label='Training Data', alpha=0.5)
    plt.plot(test.index, actual, label='Actual Price', linewidth=2)
    plt.plot(test.index, predicted, label='Predicted Price', linestyle='--', color='red')
    plt.title('ARIMA Stock Price Prediction with F1 Score')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    data = data()
    data, close_scaler = scale_data(data)

    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    best_order = find_best_arima_params(train)
    print(f'Best ARIMA Order: {best_order}')

    model = ARIMA(train['scaled_Close'],
                  exog=train[['average_sentiment', 'RSI_14', "MA5"]],
                  order=best_order)
    model_fit = model.fit()

    predicted, actual, mse, rmse, mae, f1 = predict(model_fit, test, close_scaler)

    print(f"\nMetrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"F1 Score: {f1:.2f}")
    plot_results(train, test, predicted, actual)
