import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

merged = pd.read_csv("output.csv")
merged["average_sentiment_lagged"] = merged["average_sentiment"].shift(1)
merged = merged.dropna()

features = ["average_sentiment_lagged"]
X = merged[features].values
y = merged["Close"].values.reshape(-1, 1)

# Масштабируем признаки и целевую переменную
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 10):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


time_steps = 10
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

split_idx = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

#Предсказание и инверсия скейла ─────────────────────────────────────
y_pred_scaled = model.predict(X_test).reshape(-1, 1)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_true = scaler_y.inverse_transform(y_test).flatten()

# Расчет метрик RMSE и F1 Score по направлению
# Вычисляем RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mse = mean_squared_error(y_true, y_pred)
print(f"LSTM RMSE on test set (original scale): {rmse:.4f}")
print(f"LSTM MSE on test set (original scale): {mse:.4f}")


# направления движения (0 = вниз, 1 = вверх)
def calculate_directions(prices):
    return np.where(np.diff(prices) > 0, 1, 0)


# Реальные и предсказанные направления
y_true_direction = calculate_directions(y_true)
y_pred_direction = calculate_directions(y_pred)

# F1 Score
f1 = f1_score(y_true_direction, y_pred_direction)
print(f"LSTM F1 Score on test set (directional): {f1:.4f}")


dates = merged["date"].iloc[time_steps + split_idx + 1:].reset_index(drop=True)
plt.figure(figsize=(12, 6))
plt.plot(dates, y_true[1:], label='Actual Close')
plt.plot(dates, y_pred[1:], label='Predicted Close', linestyle='--')
plt.title('LSTM: Actual vs Predicted Close Price with Lagged Sentiment')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
