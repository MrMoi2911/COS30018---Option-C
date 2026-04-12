import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import os
import yfinance as yf
import mplfinance as mpf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN



def load_and_process_data(
    company,
    start_date,
    end_date,
    feature_columns=["Open", "High", "Low", "Close", "Volume"],
    nan_strategy="drop",
    split_method="ratio",
    train_ratio=0.8,
    split_date=None,
    scale_features=True,
    save_local=True,
    local_dir="data",
    force_download=False,
    prediction_days=60,
):

#------------------------------------------------------------------------------
# Load or Download Data
#------------------------------------------------------------------------------
    os.makedirs(local_dir, exist_ok=True)
    cache_path = os.path.join(local_dir, f"{company}_{start_date}_{end_date}.csv")

    if not force_download and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = yf.download(company, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if save_local:
            df.to_csv(cache_path)

#------------------------------------------------------------------------------
# Filter feature columns
#------------------------------------------------------------------------------
    feature_columns = [col for col in feature_columns if col in df.columns]
    df = df[feature_columns].copy()

#------------------------------------------------------------------------------
# Handle NaN values
#------------------------------------------------------------------------------
    if nan_strategy == "drop":
        df.dropna(inplace=True)
    elif nan_strategy == "fill_ff":
        df.ffill(inplace=True)
        df.dropna(inplace=True)
    elif nan_strategy == "fill_bf":
        df.bfill(inplace=True)
        df.dropna(inplace=True)
    elif nan_strategy == "fill_mean":
        df.fillna(df.mean(), inplace=True)

#------------------------------------------------------------------------------
# Split into train / test
#------------------------------------------------------------------------------
    if split_method == "ratio":
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df  = df.iloc[split_idx:]
    elif split_method == "date":
        train_df = df[df.index < pd.Timestamp(split_date)]
        test_df  = df[df.index >= pd.Timestamp(split_date)]
    elif split_method == "random":
        train_df = df.sample(frac=train_ratio, random_state=42)
        test_df  = df.drop(train_df.index)

#------------------------------------------------------------------------------
# Scale the features
#------------------------------------------------------------------------------
    scalers = {}
    train_scaled = train_df.copy().astype(float)
    test_scaled  = test_df.copy().astype(float)

    if scale_features:
        for col in feature_columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled[col] = scaler.fit_transform(
                train_df[col].values.reshape(-1, 1)
            ).flatten()
            test_scaled[col] = scaler.transform(
                test_df[col].values.reshape(-1, 1)
            ).flatten()
            scalers[col] = scaler

#------------------------------------------------------------------------------
# Create sliding window sequences for LSTM
#------------------------------------------------------------------------------
    def create_sequences(scaled_df):
        X, y = [], []
        data_arr  = scaled_df.values
        close_idx = list(scaled_df.columns).index("Close")
        for i in range(prediction_days, len(data_arr)):
            X.append(data_arr[i - prediction_days:i])
            y.append(data_arr[i, close_idx])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled)
    X_test,  y_test  = create_sequences(test_scaled)

    return {
        "train_df"        : train_df,
        "test_df"         : test_df,
        "X_train"         : X_train,
        "y_train"         : y_train,
        "X_test"          : X_test,
        "y_test"          : y_test,
        "scalers"         : scalers,
        "feature_columns" : feature_columns,
    }

def plot_candlestick(df, n=1, title="Candlestick Chart"):
#------------------------------------------------------------------------------
# Resample data if n > 1 (group n trading days into 1 candle)
#------------------------------------------------------------------------------
    if n > 1:
        df_resampled = df.resample(f"{n}B").agg({
            "Open"  : "first",
            "High"  : "max",
            "Low"   : "min",
            "Close" : "last",
            "Volume": "sum"
        }).dropna()
    else:
        df_resampled = df.copy()
 
#------------------------------------------------------------------------------
# Plot the candlestick chart
#------------------------------------------------------------------------------
    mpf.plot(
        df_resampled,
        type            = "candle",
        style           = "charles",
        title           = title,
        ylabel          = "Price (AUD)",
        volume          = True,
        mav             = (20, 50),
        show_nontrading = False,
    )

def plot_boxplot(df, n=20, title="Boxplot Chart"):
#------------------------------------------------------------------------------
# Slice Close prices into non-overlapping windows of size n
#------------------------------------------------------------------------------
    close_prices = df["Close"].values
 
    windows, labels = [], []
    for i in range(0, len(close_prices) - n + 1, n):
        windows.append(close_prices[i:i+n])
        labels.append(df.index[i].strftime("%Y-%m-%d"))
 
#------------------------------------------------------------------------------
# Plot the boxplot
#------------------------------------------------------------------------------
    plt.figure(figsize=(14, 6))
    plt.boxplot(
        windows,
        labels       = labels,
        patch_artist = True,
        boxprops     = dict(facecolor="lightblue", color="navy"),
        medianprops  = dict(color="red", linewidth=2),
        whiskerprops = dict(color="navy"),
        capprops     = dict(color="navy"),
        flierprops   = dict(marker="o", color="gray", markersize=3),
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(f"Window start date (every {n} days)")
    plt.ylabel("Close Price (AUD)")
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
# Create function to build the model with configurable parameters
#------------------------------------------------------------------------------
def build_model(
    input_shape,
    layer_type="LSTM",
    layer_sizes=[50, 50, 50],
    dropout_rate=0.2,
    optimizer="adam",
    loss="mean_squared_error",
    output_size=1
):

#------------------------------------------------------------------------------
# Map layer type string to the actual Keras layer class
#------------------------------------------------------------------------------
    layer_map = {
        "LSTM"      : LSTM,
        "GRU"       : GRU,
        "RNN"       : SimpleRNN,
    }
 
    if layer_type not in layer_map:
        raise ValueError(f"layer_type need to be one of: {list(layer_map.keys())}")
    RecurrentLayer = layer_map[layer_type]

#------------------------------------------------------------------------------
# Build the Sequential model dynamically
#------------------------------------------------------------------------------
    model = Sequential()
    for i, size in enumerate(layer_sizes):
        is_first = (i == 0)
        is_last_recurrent = (i == len(layer_sizes) - 1)
        if is_first:
            model.add(RecurrentLayer(
                units            = size,
                return_sequences = not is_last_recurrent,
                input_shape      = input_shape,
            ))
        else:
            model.add(RecurrentLayer(
                units = size,
                return_sequences = not is_last_recurrent,
            ))
        model.add(Dropout(dropout_rate))

#------------------------------------------------------------------------------
# Output layer
#------------------------------------------------------------------------------
    model.add(Dense(units=output_size))
    model.compile(optimizer=optimizer, loss=loss)
    return model

#------------------------------------------------------------------------------
# Multistep prediction (predict k days into the future)
#------------------------------------------------------------------------------
def create_multistep_sequences(scaled_df, prediction_days, future_steps):
    X, y = [], []
    data_arr  = scaled_df.values
    close_idx = list(scaled_df.columns).index("Close")
    for i in range(prediction_days, len(data_arr) - future_steps + 1):
        X.append(data_arr[i - prediction_days:i, close_idx])
        y.append(data_arr[i:i + future_steps, close_idx])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return np.array(X), np.array(y)

def predict_multistep(result, future_steps=5, prediction_days=60,
                      layer_type="LSTM", layer_sizes=[50, 50],
                      epochs=25, batch_size=32):
    close_scaler = result["scalers"]["Close"]
    train_df = result["train_df"].copy().astype(float)
    test_df  = result["test_df"].copy().astype(float)
    for col in result["feature_columns"]:
        scaler = result["scalers"][col]
        train_df[col] = scaler.transform(train_df[col].values.reshape(-1, 1)).flatten()
        test_df[col]  = scaler.transform(test_df[col].values.reshape(-1, 1)).flatten()
 
    X_train, y_train = create_multistep_sequences(train_df, prediction_days, future_steps)
    X_test,  y_test  = create_multistep_sequences(test_df,  prediction_days, future_steps)

    model = build_model(
        input_shape  = (prediction_days, 1),
        layer_type   = layer_type,
        layer_sizes  = layer_sizes,
        output_size  = future_steps,
    )
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    predicted_scaled = model.predict(X_test)

    predicted_prices = close_scaler.inverse_transform(
        predicted_scaled.reshape(-1, 1)
    ).reshape(predicted_scaled.shape)

    actual_prices = close_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).reshape(y_test.shape)

    plt.figure(figsize=(14, 5))
    plt.plot(actual_prices.flatten(),    color="black", label="Actual")
    plt.plot(predicted_prices.flatten(), color="green", label=f"Predicted ({future_steps}-step)")
    plt.title(f"Multistep Prediction — {future_steps} days ahead")
    plt.xlabel("Time steps")
    plt.ylabel("Close Price (AUD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    last_window = X_test[-1:]
    next_k = model.predict(last_window)
    next_k = close_scaler.inverse_transform(next_k.reshape(-1, 1)).flatten()
    print(f"\nMultistep — next {future_steps} days prediction: {np.round(next_k, 4)}")

    return model

#------------------------------------------------------------------------------
# Multivariate prediction
#------------------------------------------------------------------------------
def predict_multivariate(result, target_day=1, prediction_days=60,
                         layer_type="LSTM", layer_sizes=[50, 50],
                         epochs=25, batch_size=32):
    close_scaler = result["scalers"]["Close"]
    n_features   = result["X_train"].shape[2]
    train_df = result["train_df"].copy().astype(float)
    test_df  = result["test_df"].copy().astype(float)

    for col in result["feature_columns"]:
        scaler = result["scalers"][col]
        train_df[col] = scaler.transform(train_df[col].values.reshape(-1, 1)).flatten()
        test_df[col]  = scaler.transform(test_df[col].values.reshape(-1, 1)).flatten()

    def create_multivariate_sequences(scaled_df):
            X, y = [], []
            data_arr  = scaled_df.values
            close_idx = list(scaled_df.columns).index("Close")
            for i in range(prediction_days, len(data_arr) - target_day + 1):
                X.append(data_arr[i - prediction_days:i])
                y.append(data_arr[i + target_day - 1, close_idx])
            return np.array(X), np.array(y)

    X_train, y_train = create_multivariate_sequences(train_df)
    X_test,  y_test  = create_multivariate_sequences(test_df)

    model = build_model(
        input_shape  = (prediction_days, n_features),
        layer_type   = layer_type,
        layer_sizes  = layer_sizes,
        output_size  = 1,
    )
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    predicted_scaled = model.predict(X_test)
    predicted_prices = close_scaler.inverse_transform(predicted_scaled)
    actual_prices    = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.figure(figsize=(14, 5))
    plt.plot(actual_prices,    color="black", label="Actual")
    plt.plot(predicted_prices, color="blue",  label=f"Predicted (day +{target_day})")
    plt.title(f"Multivariate Prediction — target day +{target_day}")
    plt.xlabel("Time")
    plt.ylabel("Close Price (AUD)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    last_window = X_test[-1:]
    next_pred   = model.predict(last_window)
    next_pred   = close_scaler.inverse_transform(next_pred)
    print(f"\nMultivariate — day +{target_day} prediction: {next_pred[0][0]:.4f}")
    
    return model

#------------------------------------------------------------------------------
# Multivariate + Multistep
#------------------------------------------------------------------------------
def predict_multivariate_multistep(result, future_steps=5, prediction_days=60,
                                   layer_type="LSTM", layer_sizes=[50, 50],
                                   epochs=25, batch_size=32):
    close_scaler = result["scalers"]["Close"]
    n_features   = result["X_train"].shape[2]
    train_df = result["train_df"].copy().astype(float)
    test_df  = result["test_df"].copy().astype(float)

    for col in result["feature_columns"]:
        scaler = result["scalers"][col]
        train_df[col] = scaler.transform(train_df[col].values.reshape(-1, 1)).flatten()
        test_df[col]  = scaler.transform(test_df[col].values.reshape(-1, 1)).flatten()

    X_train, y_train = create_multistep_sequences(train_df, prediction_days, future_steps)
    X_test,  y_test  = create_multistep_sequences(test_df,  prediction_days, future_steps)
 
    model = build_model(
        input_shape  = (prediction_days, n_features),
        layer_type   = layer_type,
        layer_sizes  = layer_sizes,
        output_size  = future_steps,
    )
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
 
    predicted_scaled = model.predict(X_test)
    predicted_prices = close_scaler.inverse_transform(
        predicted_scaled.reshape(-1, 1)
    ).reshape(predicted_scaled.shape)
    actual_prices = close_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).reshape(y_test.shape)

    plt.figure(figsize=(14, 5))
    plt.plot(actual_prices.flatten(),    color="black",  label="Actual")
    plt.plot(predicted_prices.flatten(), color="purple", label=f"Predicted ({future_steps}-step multivariate)")
    plt.title(f"Multivariate + Multistep Prediction — {future_steps} days ahead")
    plt.xlabel("Time steps")
    plt.ylabel("Close Price (AUD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    last_window = X_test[-1:]
    next_k = model.predict(last_window)
    next_k = close_scaler.inverse_transform(next_k.reshape(-1, 1)).flatten()
    print(f"\nMultivariate+Multistep — next {future_steps} days: {np.round(next_k, 4)}")
 
    return model

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
COMPANY    = "CBA.AX"
START_DATE = "2020-01-01"
END_DATE   = "2024-07-02"
PREDICTION_DAYS = 60

result = load_and_process_data(
    company         = COMPANY,
    start_date      = START_DATE,
    end_date        = END_DATE,
    nan_strategy    = "fill_ff",
    split_method    = "ratio",
    train_ratio     = 0.8,
    scale_features  = True,
    save_local      = True,
    prediction_days = PREDICTION_DAYS,
)

# Plot additional charts
plot_candlestick(result["train_df"], n=5, title="Candlestick Chart")

plot_boxplot(result["train_df"], n=20, title="Boxplot Chart")

X_train      = result["X_train"]
y_train      = result["y_train"]
X_test       = result["X_test"]
y_test       = result["y_test"]
close_scaler = result["scalers"]["Close"]
n_features   = X_train.shape[2]

input_shape = (PREDICTION_DAYS, n_features)

#------------------------------------------------------------------------------
# Experiment with different model configurations
#------------------------------------------------------------------------------
configs = [
    {
        "name"        : "LSTM_3layer_50",
        "layer_type"  : "LSTM",
        "layer_sizes" : [50, 50, 50],
        "dropout_rate": 0.2,
        "optimizer"   : "adam",
        "epochs"      : 25,
        "batch_size"  : 32,
    },
    {
        "name"        : "GRU_2layer_64",
        "layer_type"  : "GRU",
        "layer_sizes" : [64, 64],
        "dropout_rate": 0.3,
        "optimizer"   : "adam",
        "epochs"      : 25,
        "batch_size"  : 32,
    },
    {
        "name"        : "RNN_2layer_32",
        "layer_type"  : "RNN",
        "layer_sizes" : [32, 32],
        "dropout_rate": 0.2,
        "optimizer"   : "rmsprop",
        "epochs"      : 25,
        "batch_size"  : 64,
    },
]

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Training: {cfg['name']}")
    print(f"{'='*60}")

    model = build_model(
        input_shape  = input_shape,
        layer_type   = cfg["layer_type"],
        layer_sizes  = cfg["layer_sizes"],
        dropout_rate = cfg["dropout_rate"],
        optimizer    = cfg["optimizer"]
    )

#------------------------------------------------------------------------------
# Traning the model
#------------------------------------------------------------------------------
    model.fit(
            X_train, y_train,
            epochs     = cfg["epochs"],
            batch_size = cfg["batch_size"],
            verbose    = 1,
        )

#------------------------------------------------------------------------------
# Testing the model
#------------------------------------------------------------------------------
    predicted_prices = model.predict(X_test)
    predicted_prices = close_scaler.inverse_transform(predicted_prices)
    actual_prices    = close_scaler.inverse_transform(y_test.reshape(-1, 1))

#------------------------------------------------------------------------------
# Visualise the results
#------------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))
    plt.plot(actual_prices,    color="black", label=f"Actual {COMPANY} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted — {cfg['name']}")
    plt.title(f"{COMPANY} — {cfg['name']}")
    plt.xlabel("Time")
    plt.ylabel("Price (AUD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------
    last_window = X_test[-1:]  # shape: (1, 60, n_features)
    prediction  = model.predict(last_window)
    prediction  = close_scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")

#------------------------------------------------------------------------------
# Multistep, Multivariate, and Combined predictions
#------------------------------------------------------------------------------
print("\n" + "="*60)
print("Multistep Prediction (5 days ahead)")
print("="*60)
predict_multistep(
    result          = result,
    future_steps    = 5,
    prediction_days = PREDICTION_DAYS,
    layer_type      = "LSTM",
    layer_sizes     = [50, 50],
    epochs          = 25,
    batch_size      = 32,
)
 
print("\n" + "="*60)
print("Multivariate Prediction (day +3)")
print("="*60)
predict_multivariate(
    result          = result,
    target_day      = 3,
    prediction_days = PREDICTION_DAYS,
    layer_type      = "LSTM",
    layer_sizes     = [50, 50],
    epochs          = 25,
    batch_size      = 32,
)
 
print("\n" + "="*60)
print("Multivariate + Multistep (5 days ahead)")
print("="*60)
predict_multivariate_multistep(
    result          = result,
    future_steps    = 5,
    prediction_days = PREDICTION_DAYS,
    layer_type      = "LSTM",
    layer_sizes     = [50, 50],
    epochs          = 25,
    batch_size      = 32,
)