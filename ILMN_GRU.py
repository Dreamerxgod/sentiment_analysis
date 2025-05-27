import pandas as pd
import numpy as np
import random as python_random
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# np.random.seed(42)
# python_random.seed(42)
# tf.random.set_seed(42)


df = pd.read_csv("tweets_ilmn_sentiment_output.csv", parse_dates=["date"])
df["average_sentiment_lagged"] = df["average_sentiment"].shift(0)
df["ma5"]  = df["price"].rolling(5).mean().shift(1)
df["ma10"] = df["price"].rolling(10).mean().shift(1)

delta = df['price'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14, min_periods=1).mean()
avg_loss = loss.rolling(14, min_periods=1).mean()

rs = avg_gain / (avg_loss + 1e-10)  #
df['RSI_14'] = 100 - (100 / (1 + rs))

df = df.dropna().reset_index(drop=True)

prices = df["price"].values.reshape(-1,1)
dates  = df["date"]

scaler_X = StandardScaler()
X_all = scaler_X.fit_transform(df[["average_sentiment_lagged", "ma5","ma10", "RSI_14"]])
scaler_y = MinMaxScaler()
y_all = scaler_y.fit_transform(prices)

def create_sequences(X, y_scaled, prices_orig, tsteps=10):
    Xs, ys_p, ys_d = [], [], []
    for i in range(tsteps, len(X)):
        Xs.append(X[i-tsteps:i])
        ys_p.append(y_scaled[i])
        ys_d.append(1.0 if prices_orig[i] > prices_orig[i-1] else 0.0)
    return np.array(Xs), np.array(ys_p), np.array(ys_d).reshape(-1,1)

time_steps = 10
X_seq, y_price_seq, y_dir_seq = create_sequences(X_all, y_all, prices, time_steps)
dates_seq = dates.iloc[time_steps:].reset_index(drop=True)

n = len(X_seq)
i_test = int(0.7 * n)
i_val  = int(0.5 * i_test)

X_train, X_val, X_test = X_seq[:i_val], X_seq[i_val:i_test], X_seq[i_test:]
y_p_train, y_p_val, y_p_test = y_price_seq[:i_val], y_price_seq[i_val:i_test], y_price_seq[i_test:]
y_d_train, y_d_val, y_d_test = y_dir_seq[:i_val], y_dir_seq[i_val:i_test], y_dir_seq[i_test:]
dates_test = dates_seq[i_test:].reset_index(drop=True)

true_price = scaler_y.inverse_transform(y_p_test).flatten()
true_dir   = y_d_test.flatten().astype(int)

counts = np.bincount(true_dir)
counts_train = np.bincount(y_d_train.flatten().astype(int))
neg, pos = counts_train[0], counts_train[1]
pos_weight = neg / pos

import tensorflow.keras.backend as K
def weighted_bce(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * pos_weight + (1 - y_true)
    return K.mean(bce * weight_vector)

def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = GRU(64, activation='tanh')(inp)
    x = Dropout(0.2)(x)
    price_out = Dense(1, name="price")(x)
    dir_out   = Dense(1, activation='sigmoid', name="direction")(x)
    m = Model(inp, [price_out, dir_out])
    m.compile(
        optimizer="adam",
        loss={
            "price": "mse",
            "direction": weighted_bce
        },
        loss_weights={
            "price": 2.0,
            "direction": 5.0
        }
    )
    return m

model = build_model((time_steps, X_train.shape[2]))
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    {"price": y_p_train, "direction": y_d_train},
    validation_data=(X_val, {"price": y_p_val, "direction": y_d_val}),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

probs_val = model.predict(X_val)[1].flatten()
best_thr, best_f1 = 0.5, 0
for thr in np.linspace(0.3, 0.7, 41):
    preds = (probs_val >= thr).astype(int)
    f = f1_score(y_d_val.flatten().astype(int), preds)
    if f > best_f1:
        best_f1, best_thr = f, thr
print(f"\nЛучший threshold: {best_thr:.2f}, F1_val: {best_f1:.4f}")

pred_p_s, pred_d_p = model.predict(X_test)
pred_price = scaler_y.inverse_transform(pred_p_s).flatten()
pred_dir   = (pred_d_p.flatten() >= best_thr).astype(int)

rmse = np.sqrt(mean_squared_error(true_price, pred_price))
mse  = mean_squared_error(true_price, pred_price)
f1   = f1_score(true_dir, pred_dir)

print(f"\n--- Test @ thr={best_thr:.2f} ---")
print(f"RMSE: {rmse:.4f}, MSE: {mse:.4f}, F1: {f1:.4f}")

plt.figure(figsize=(12,6))
plt.plot(dates_test, true_price, label="Actual Close")
plt.plot(dates_test, pred_price, '--', label="Predicted Close")
plt.title("Actual vs Predicted Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
