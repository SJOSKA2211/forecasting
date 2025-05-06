# demand_forecast_app.py
import pandas as pd
import numpy as np
import os
import time # Import time for potential timing or pauses
# import matplotlib.pyplot as plt # Matplotlib will be used with Streamlit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA # For ARIMA and MA models
from pmdarima import auto_arima # For automatic ARIMA selection
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, Bidirectional # Include Dense for NN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam # Import Adam optimizer
import joblib # For saving/loading ARIMA model and scalers
import warnings
import sys
import traceback # For detailed error reporting
import streamlit as st # Import Streamlit
import matplotlib.pyplot as plt # Import Matplotlib for Streamlit plotting
import seaborn as sns # Import Seaborn
import plotly.express as px # Import Plotly
import plotly.graph_objects as go # Import Plotly Graph Objects
# import altair as alt # Import Altair if needed
from concurrent.futures import ProcessPoolExecutor, as_completed # Import for parallel processing
import plotly.io as pio # Import plotly.io for saving images

warnings.filterwarnings('ignore') # Ignore some common warnings

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Disable TensorFlow verbosity for cleaner process output
os.environ['TF_CPP_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow logger

st.set_page_config(layout="wide") # Use wide layout for Streamlit app

st.title("Demand Forecasting Application")
st.write("This application forecasts demand using ARIMA, MA, NN, LSTM, and Hybrid ARIMA-LSTM models.")

# --- Check for GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    st.success(f"Detected GPUs: {gpus[0].name}. TensorFlow will attempt to use the GPU for faster training.")
else:
    st.warning("No GPU detected. TensorFlow will use the CPU, which may be slower for deep learning models.")

# --- Step 0: Configuration ---
st.header("Configuration")

# Use Streamlit inputs for configuration
data_full_file_path = st.text_input("Enter the full path to the sales data file (CSV or Excel):", "sales_data.csv") # Added a default
date_column = st.text_input("Enter the exact name of the date column:", "Date") # Added a default
sales_column = st.text_input("Enter the exact name of the sales/demand column:", "Sales") # Added a default
grouping_column = st.text_input("Enter the exact name of the grouping column (e.g., Store, Product ID):", "StoreID") # Added a default

time_series_frequency = st.text_input("Enter the time series frequency (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly - used for ARIMA/MA seasonality). Leave empty to auto-detect:", "") # Added frequency input

test_size_days = st.number_input("Enter the number of days for the test set (e.g., 90):", min_value=1, value=90)
lstm_look_back = st.number_input("Enter the look-back period for LSTM/NN (e.g., 30 days):", min_value=1, value=30) # Lookback used for NN too

st.subheader("Model Selection")
use_arima_model = st.checkbox("Include ARIMA Model?", value=True)
use_ma_model = st.checkbox("Include MA (Moving Average) Model?", value=True)
ma_q_order = st.number_input("MA (q) order (typically based on seasonality, e.g., 7 for daily data with weekly seasonality):", min_value=1, value=7, disabled=not use_ma_model)
use_nn_model = st.checkbox("Include NN (Neural Network) Model?", value=True)
nn_layers = st.number_input("NN Layers (e.g., 2):", min_value=1, value=2, disabled=not use_nn_model)
nn_units = st.number_input("NN Units per layer (e.g., 64):", min_value=1, value=64, disabled=not use_nn_model)
use_lstm_model = st.checkbox("Include LSTM Model?", value=True)
use_bidirectional_lstm = st.checkbox("Use Bidirectional LSTM?", value=True, disabled=not use_lstm_model)
use_hybrid_model = st.checkbox("Include Hybrid ARIMA-LSTM Model?", value=True and use_arima_model and use_lstm_model, disabled=not (use_arima_model and use_lstm_model)) # Hybrid requires ARIMA and LSTM

st.subheader("Training Parameters")
lstm_epochs = st.number_input("Enter the number of epochs for NN/LSTM training (Early Stopping will likely stop sooner):", min_value=1, value=100) # Increased default epochs, relying on ES
lstm_batch_size = st.number_input("Enter the batch size for NN/LSTM training (e.g., 32):", min_value=1, value=32)

st.subheader("Other Settings")
add_lagged_features = st.checkbox("Add simple lagged features (lag 1 and lag 7)?", value=True) # Option for lagged features
max_workers = st.slider("Number of parallel processes (adjust based on your CPU cores):", 1, os.cpu_count(), os.cpu_count() // 2 or 1) # Use half of available cores by default

output_dir = st.text_input("Enter the directory path to save results (models, forecasts, plots):", "forecast_results") # Added a default

# Button to start the process
start_button = st.button("Start Forecasting")

# Global lists/dicts to store results from parallel processing
all_evaluation_metrics_list = []
all_forecasts_dict = {}
all_test_data_dict = {}
all_residual_plots_data = {} # To store data for residual plots
all_lstm_history_data = {} # To store LSTM training history
all_nn_history_data = {} # To store NN training history
all_errors_list = [] # To store errors from parallel processes

df = None # Initialize df

# --- Helper Functions for Sequence Models (LSTM/NN) ---
def create_sequence_dataset(dataset, look_back=1):
    """Create input dataset for sequence models (LSTM/NN)."""
    dataX, dataY = [], []
    # Ensure dataset is numpy array before slicing
    dataset = np.array(dataset)
    # Check if dataset has enough samples for look_back
    if len(dataset) <= look_back:
        return np.array([]), np.array([]) # Return empty arrays if not enough data

    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :] # Capture all features for input sequence
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0]) # Predict only the sales column (index 0)
    # Reshape X for sequence models (n_samples, look_back, n_features)
    return np.array(dataX), np.array(dataY)


def build_lstm_model(look_back, features=1, bidirectional=False):
    """Builds the LSTM model."""
    model = Sequential(name=f"{'Bi' if bidirectional else ''}LSTM_Model")
    model.add(InputLayer(input_shape=(look_back, features)))

    if bidirectional:
        model.add(Bidirectional(LSTM(50, return_sequences=True)))
    else:
        model.add(LSTM(50, return_sequences=True))

    model.add(Dropout(0.2))

    if bidirectional:
         model.add(Bidirectional(LSTM(50)))
    else:
         model.add(LSTM(50))

    model.add(Dropout(0.2))
    model.add(Dense(1)) # Output is a single value (sales forecast)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def build_nn_model(look_back, features=1, num_layers=2, units=64):
    """Builds a Feedforward Neural Network (MLP) model for time series."""
    model = Sequential(name="NN_Model")
    # Flatten the look_back window and features into a single input layer
    model.add(InputLayer(input_shape=(look_back, features)))
    model.add(tf.keras.layers.Flatten()) # Flatten the input sequence

    # Add Dense layers
    for _ in range(num_layers):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.2))

    model.add(Dense(1)) # Output layer
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# --- Feature Engineering Function ---
def create_lagged_features(df, sales_col, date_col, lags=[1, 7]):
    """Creates lagged features for a single group DataFrame."""
    df_lagged = df.copy()
    df_lagged = df_lagged.sort_values(by=date_col) # Ensure sorted by date

    for lag in lags:
        df_lagged[f'{sales_col}_Lag_{lag}'] = df_lagged[sales_col].shift(lag)

    # Handle NaNs created by lagging - fill with a strategy suitable for your data
    # Using mean of the respective lag column as a slightly better approach than 0 if data allows
    for lag in lags:
        lag_col = f'{sales_col}_Lag_{lag}'
        if df_lagged[lag_col].isnull().any():
             mean_val = df_lagged[lag_col].mean()
             df_lagged[lag_col].fillna(mean_val if not np.isnan(mean_val) else 0, inplace=True)


    return df_lagged


# --- Function to Process a Single Group (for Parallel Execution) ---
def train_and_forecast_group(group_id, group_df_json, config):
    """
    Trains models for a single group and returns results.
    Designed to be run in a separate process. Includes error handling.
    """
    group_forecasts = {}
    metrics = {'group_id': group_id}
    test_data_json = None
    error_message = None
    residual_data = {}
    lstm_history = None
    nn_history = None


    try:
        # Reconstruct DataFrame from JSON string
        group_df = pd.read_json(group_df_json, orient='split')
        group_df[config['date_column']] = pd.to_datetime(group_df[config['date_column']])
        # Keep the date column for feature engineering before setting index
        # group_df.set_index(config['date_column'], inplace=True) # Set index later if needed


        sales_column = config['sales_column']
        date_column = config['date_column']
        config_time_series_frequency = config['time_series_frequency'] # Get from config
        test_size_days = config['test_size_days']
        lstm_look_back = config['lstm_look_back'] # Used for NN lookback too
        lstm_epochs = config['lstm_epochs']
        lstm_batch_size = config['lstm_batch_size']
        use_bidirectional_lstm = config['use_bidirectional_lstm']
        add_lagged_features = config['add_lagged_features']
        output_dir = config['output_dir']

        # Model Inclusion Flags and Parameters
        use_arima_model = config['use_arima_model']
        use_ma_model = config['use_ma_model']
        ma_q_order = config['ma_q_order']
        use_nn_model = config['use_nn_model']
        nn_layers = config['nn_layers']
        nn_units = config['nn_units']
        use_lstm_model = config['use_lstm_model']
        use_hybrid_model = config['use_hybrid_model'] # Depends on ARIMA and LSTM


        # --- Feature Engineering (inside process for that group) ---
        features_to_scale = [sales_column] # Always include sales column
        if add_lagged_features:
             group_df = create_lagged_features(group_df, sales_column, date_column, lags=[1, 7])
             # Add lagged feature columns to features_to_scale
             features_to_scale.extend([col for col in group_df.columns if '_Lag_' in col])

        # Add other general time-based features (Year, Month, etc.) to features_to_scale if present
        # Assuming these were added to the main df before passing to the process
        # Need to make sure these columns exist in the group_df and are numeric
        general_time_features = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'WeekOfYear']
        for feature in general_time_features:
            if feature in group_df.columns and pd.api.types.is_numeric_dtype(group_df[feature]):
                 if feature not in features_to_scale: # Avoid duplicates
                     features_to_scale.append(feature)


        # Set index AFTER feature engineering that might need the date column
        group_df.set_index(date_column, inplace=True)
        ts_data = group_df[features_to_scale] # Use ALL selected features for LSTM/NN input


        # --- Data Split ---
        min_data_points_arima_ma = test_size_days + 1 # ARIMA/MA only need data for split
        min_data_points_sequence = test_size_days + lstm_look_back + 1 # LSTM/NN need lookback

        min_required_points = min_data_points_arima_ma
        if use_lstm_model or use_nn_model or use_hybrid_model:
             min_required_points = max(min_required_points, min_data_points_sequence)


        if len(ts_data) < min_required_points:
             error_message = f"Insufficient data points ({len(ts_data)}) for test split ({test_size_days}) and sequence look-back ({lstm_look_back}). Required: {min_required_points}"
             metrics.update({k: np.nan for k in ['ARIMA_RMSE', 'ARIMA_MAE', 'MA_RMSE', 'MA_MAE', 'NN_RMSE', 'NN_MAE', 'LSTM_RMSE', 'LSTM_MAE', 'Hybrid_RMSE', 'Hybrid_MAE']})
             return group_id, None, None, metrics, error_message, None, None, None # Return error message and None for residuals/histories


        train_data = ts_data[:-test_size_days]
        test_data = ts_data[-test_size_days:]

        # Convert test_data index to a common format before JSON conversion
        test_data_reset = test_data.reset_index()
        test_data_reset[date_column] = test_data_reset[date_column].astype(str) # Convert date to string for JSON
        test_data_json = test_data_reset.to_json(orient='split')


        # Determine ARIMA/MA seasonality (m) based on frequency
        # pmdarima uses integer for 'm'. Common frequencies:
        # D=1 (no seasonality), W=7, M=12, Q=4, Y=1. A value of 7 is often used for daily data assuming weekly seasonality.
        arima_ma_m = 1 # Default to non-seasonal if frequency is not standard or detectable

        # If user provided frequency, use that first
        if config_time_series_frequency:
            try:
                 user_freq_offset = pd.tseries.frequencies.to_offset(config_time_series_frequency)
                 if user_freq_offset:
                    st_user = user_freq_offset.freqstr
                    if st_user == 'D': arima_ma_m = 7 # Assuming weekly seasonality for Daily data
                    elif st_user == 'W': arima_ma_m = 52 # Assuming yearly seasonality for Weekly data
                    elif st_user == 'M': arima_ma_m = 12
                    elif st_user == 'Q' or st_user == 'QS' : arima_ma_m = 4
                    elif st_user == 'Y' or st_user == 'YS': arima_ma_m = 1
                    # print(f"Using user-specified frequency '{config_time_series_frequency}' (m={arima_ma_m}) for group {group_id}")
            except Exception as e:
                 error_message = (error_message + "\n" if error_message else "") + f"Frequency Conversion Error: {e}. Attempting auto-inference."
                 arima_ma_m = 1 # Fallback if user input is invalid


        # If no valid user frequency or user left it empty, attempt to infer
        if arima_ma_m == 1 and not config_time_series_frequency:
            try:
                inferred_freq = pd.infer_freq(train_data.index)
                if inferred_freq:
                     st_inf = pd.tseries.frequencies.to_offset(inferred_freq).freqstr # Convert to freq string
                     if st_inf == 'D': arima_ma_m = 7 # Assuming weekly seasonality for Daily data
                     elif st_inf == 'W': arima_ma_m = 52 # Assuming yearly seasonality for Weekly data
                     elif st_inf == 'M': arima_ma_m = 12
                     elif st_inf == 'Q' or st_inf == 'QS' : arima_ma_m = 4
                     elif st_inf == 'Y' or st_inf == 'YS': arima_ma_m = 1
                     # print(f"Using inferred frequency '{inferred_freq}' (m={arima_ma_m}) for group {group_id}")
                # else:
                   # print(f"Could not infer frequency for group {group_id}. Using m=1.")

            except Exception as e:
                 # print(f"Error inferring frequency for group {group_id}: {e}. Using m=1.")
                 arima_ma_m = 1 # Fallback to non-seasonal

        # print(f"Final ARIMA/MA seasonality (m) for group {group_id}: {arima_ma_m}")


        # --- Model 1: ARIMA ---
        arima_model = None
        arima_forecast = None
        arima_residuals_test = None
        if use_arima_model:
            try:
                # Fine-tuned auto_arima parameters for potentially faster search and using determined 'm'
                arima_model = auto_arima(train_data[sales_column],
                                         start_p=1, start_q=1,
                                         test='adf',
                                         max_p=2, max_q=2, # Reduced max p, q
                                         m=arima_ma_m,    # Use determined seasonality
                                         start_P=0, start_Q=0,
                                         max_P=1, max_Q=1, # Reduced max P, Q
                                         seasonal=True, d=None, D=None, trace=False,
                                         error_action='ignore', suppress_warnings=True, stepwise=True)

                arima_forecast = arima_model.predict(n_periods=len(test_data))
                group_forecasts['ARIMA'] = arima_forecast.values.tolist()

                arima_residuals_test = test_data[sales_column].values - arima_forecast.values
                residual_data['ARIMA'] = arima_residuals_test.tolist()

                arima_rmse = np.sqrt(np.mean(arima_residuals_test**2))
                arima_mae = np.mean(np.abs(arima_residuals_test))
                metrics['ARIMA_RMSE'] = arima_rmse
                metrics['ARIMA_MAE'] = arima_mae
                # print(f"  ARIMA Forecast generated for {group_id}. RMSE: {arima_rmse:.4f}, MAE: {arima_mae:.4f}")
            except Exception as e:
                error_message = (error_message + "\n" if error_message else "") + f"ARIMA Error: {e}"
                # print(f"  Error training/forecasting ARIMA for group {group_id}: {e}")
                metrics['ARIMA_RMSE'] = np.nan
                metrics['ARIMA_MAE'] = np.nan
                group_forecasts['ARIMA'] = None
                residual_data['ARIMA'] = None # No residuals if error
        else:
             metrics['ARIMA_RMSE'] = np.nan # Mark as NaN if model not selected
             metrics['ARIMA_MAE'] = np.nan


        # --- Model 2: MA (Moving Average) ---
        ma_model = None
        ma_forecast = None
        ma_residuals_test = None
        if use_ma_model:
            try:
                 # Implement MA(q) model using statsmodels ARIMA with p=0, d=0
                 # Use the specified ma_q_order and seasonality (m)
                 # Check if ma_q_order is valid and training data has enough points
                 if ma_q_order <= 0:
                      raise ValueError(f"MA order (q) must be positive, got {ma_q_order}")
                 if len(train_data) < ma_q_order + 1: # Need at least q+1 points to estimate MA(q)
                      raise ValueError(f"Insufficient training data points ({len(train_data)}) for MA({ma_q_order}) model. Need at least {ma_q_order + 1}.")


                 ma_model = ARIMA(train_data[sales_column], order=(0, 0, ma_q_order),
                                  seasonal_order=(0, 0, 0, arima_ma_m if arima_ma_m > 1 else 0)) # Only add seasonal component if m > 1
                 ma_results = ma_model.fit()

                 # Generate forecast for the test period
                 ma_forecast = ma_results.predict(start=len(train_data), end=len(ts_data)-1)

                 group_forecasts['MA'] = ma_forecast.values.tolist()

                 # Calculate test residuals for MA
                 ma_residuals_test = test_data[sales_column].values - ma_forecast.values
                 residual_data['MA'] = ma_residuals_test.tolist()

                 ma_rmse = np.sqrt(np.mean(ma_residuals_test**2))
                 ma_mae = np.mean(np.abs(ma_residuals_test))
                 metrics['MA_RMSE'] = ma_rmse
                 metrics['MA_MAE'] = ma_mae
                 # print(f"  MA Forecast generated for {group_id}. RMSE: {ma_rmse:.4f}, MAE: {ma_mae:.4f}")

            except Exception as e:
                 error_message = (error_message + "\n" if error_message else "") + f"MA Error: {e}"
                 # print(f"  Error training/forecasting MA for group {group_id}: {e}")
                 metrics['MA_RMSE'] = np.nan
                 metrics['MA_MAE'] = np.nan
                 group_forecasts['MA'] = None
                 residual_data['MA'] = None # No residuals if error
        else:
             metrics['MA_RMSE'] = np.nan # Mark as NaN if model not selected
             metrics['MA_MAE'] = np.nan


        # --- Data Scaling and Sequence Creation (for NN and LSTM) ---
        scaler = None
        scaled_train_data = None
        scaled_ts_data = None
        X_train_seq = None
        y_train_seq = None

        # Prepare sequence data only if at least one sequence model is selected
        if use_nn_model or use_lstm_model or use_hybrid_model:
            try:
                 scaler = StandardScaler()
                 # Scale all selected features using only training data for fitting
                 scaled_train_data = scaler.fit_transform(train_data)
                 scaled_ts_data = scaler.transform(ts_data) # Transform entire time series

                 # Create sequence dataset for NN/LSTM
                 if len(scaled_train_data) <= lstm_look_back:
                      raise ValueError(f"Not enough training data points ({len(scaled_train_data)}) for the specified look_back period ({lstm_look_back}).")

                 X_train_seq, y_train_seq = create_sequence_dataset(scaled_train_data, lstm_look_back)

                 if X_train_seq.shape[0] == 0:
                      raise ValueError("Sequence dataset creation resulted in zero training samples.")

                 # Reshape X_train_seq for Keras input (n_samples, look_back, n_features)
                 # NN will flatten this later, LSTM expects this shape
                 X_train_seq = np.reshape(X_train_seq, (X_train_seq.shape[0], lstm_look_back, len(features_to_scale)))


                 # Save scaler - only save if sequence data prep was successful
                 group_model_dir = os.path.join(output_dir, f"group_models_{group_id}")
                 os.makedirs(group_model_dir, exist_ok=True)
                 scaler_filename = os.path.join(group_model_dir, f'scaler_group_{group_id}.joblib')
                 joblib.dump(scaler, scaler_filename)


            except Exception as e:
                 error_message = (error_message + "\n" if error_message else "") + f"Scaling/Sequence Data Prep Error: {e}"
                 # print(f"  Error during scaling or sequence data preparation for group {group_id}: {e}")
                 # Ensure related models are marked as failed
                 if use_nn_model: metrics['NN_RMSE'] = np.nan; metrics['NN_MAE'] = np.nan; group_forecasts['NN'] = None; residual_data['NN'] = None; nn_history = None
                 if use_lstm_model: metrics['LSTM_RMSE'] = np.nan; metrics['LSTM_MAE'] = np.nan; group_forecasts['LSTM'] = None; residual_data['LSTM'] = None; lstm_history = None
                 if use_hybrid_model: metrics['Hybrid_RMSE'] = np.nan; metrics['Hybrid_MAE'] = np.nan; group_forecasts['Hybrid'] = None; residual_data['Hybrid'] = None
                 # Ensure training is skipped
                 X_train_seq = None
                 scaled_ts_data = None # Ensure test data is also marked as unavailable for sequence models


        # --- Model 3: NN (Neural Network) ---
        nn_model = None
        nn_forecast = None
        nn_residuals_test = None
        nn_history = None # Renamed to avoid conflict with lstm_history
        if use_nn_model and X_train_seq is not None: # Only train if data prep was successful
            try:
                # Use a subdirectory for each group's models
                group_model_dir = os.path.join(output_dir, f"group_models_{group_id}")
                os.makedirs(group_model_dir, exist_ok=True)
                nn_checkpoint_path = os.path.join(group_model_dir, f"nn_model_group_{group_id}.keras")

                # --- Check for and load existing model checkpoint ---
                if os.path.exists(nn_checkpoint_path):
                    try:
                        nn_model = load_model(nn_checkpoint_path)
                        # print(f"  Loaded existing NN model for group {group_id} from checkpoint.")
                        loaded_from_checkpoint = True
                    except Exception as load_e:
                        # print(f"  Error loading existing NN model for group {group_id}: {load_e}. Building new model.")
                        nn_model = build_nn_model(lstm_look_back, features=len(features_to_scale), num_layers=nn_layers, units=nn_units)
                        loaded_from_checkpoint = False
                else:
                    # print(f"  No existing NN checkpoint found for group {group_id}. Building new model.")
                    nn_model = build_nn_model(lstm_look_back, features=len(features_to_scale), num_layers=nn_layers, units=nn_units)
                    loaded_from_checkpoint = False
                # --- End of checkpoint loading ---


                nn_callbacks = [
                    EarlyStopping(monitor='loss', patience=10, verbose=0, restore_best_weights=True),
                    ModelCheckpoint(filepath=nn_checkpoint_path, monitor='loss', save_best_only=True, verbose=0, save_weights_only=False),
                    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0)
                ]

                # Train NN model
                history_nn = nn_model.fit(X_train_seq, y_train_seq, epochs=lstm_epochs, batch_size=lstm_batch_size,
                                          verbose=0, callbacks=nn_callbacks, shuffle=False) # Shuffle=False important for time series
                nn_history = history_nn.history['loss'] # Collect loss history

                # Ensure best model is loaded after training if checkpoint was used
                if os.path.exists(nn_checkpoint_path):
                    try:
                        nn_model = load_model(nn_checkpoint_path)
                        # print(f"  Loaded best NN model after training for group {group_id}.")
                    except Exception as load_e:
                        # print(f"  Error loading best NN model after training for group {group_id}: {load_e}.")
                        pass # Continue with the model object from fit if loading fails


                # Prepare test data for NN prediction (rolling forecast)
                # Similar to LSTM, need the last `look_back` points of scaled data
                if scaled_ts_data is None or len(scaled_ts_data) < len(test_data) + lstm_look_back:
                     raise ValueError("Insufficient scaled test data for NN rolling forecast.")

                nn_input_sequence_scaled = scaled_ts_data[-(len(test_data) + lstm_look_back):]

                nn_forecast_scaled_list = []
                current_input_window_nn = nn_input_sequence_scaled[:lstm_look_back] # Shape (look_back, n_features)

                for i in range(len(test_data)):
                     # Reshape for prediction (1, look_back, n_features)
                     input_for_pred_nn = np.reshape(current_input_window_nn, (1, lstm_look_back, len(features_to_scale)))
                     pred_scaled_nn = nn_model.predict(input_for_pred_nn, verbose=0)[0, 0] # Predict the next sales value
                     nn_forecast_scaled_list.append(pred_scaled_nn)

                     # Prepare input for the next step: roll and update sales position
                     next_input_window_nn = np.roll(current_input_window_nn, -1, axis=0)
                     next_input_window_nn[-1, features_to_scale.index(sales_column)] = pred_scaled_nn

                     current_input_window_nn = next_input_window_nn


                nn_forecast_scaled = np.array(nn_forecast_scaled_list).reshape(-1, 1)
                # Inverse transform
                temp_scaled_array_nn = np.zeros((nn_forecast_scaled.shape[0], len(features_to_scale)))
                temp_scaled_array_nn[:, features_to_scale.index(sales_column)] = nn_forecast_scaled.flatten()
                # Use the main scaler for inverse transform
                nn_forecast = scaler.inverse_transform(temp_scaled_array_nn)[:, features_to_scale.index(sales_column)].flatten()


                group_forecasts['NN'] = nn_forecast.tolist() # Convert to list

                # Calculate test residuals for NN
                nn_residuals_test = test_data[sales_column].values - nn_forecast
                residual_data['NN'] = nn_residuals_test.tolist()

                nn_rmse = np.sqrt(np.mean(nn_residuals_test**2))
                nn_mae = np.mean(np.abs(nn_residuals_test))
                metrics['NN_RMSE'] = nn_rmse
                metrics['NN_MAE'] = nn_mae
                # print(f"  NN Forecast generated for {group_id}. RMSE: {nn_rmse:.4f}, MAE: {nn_mae:.4f}")

            except Exception as e:
                error_message = (error_message + "\n" if error_message else "") + f"NN Error: {e}"
                # print(f"  Error training/forecasting NN for group {group_id}: {e}")
                metrics['NN_RMSE'] = np.nan
                metrics['NN_MAE'] = np.nan
                group_forecasts['NN'] = None
                residual_data['NN'] = None
                nn_history = None
        else:
             metrics['NN_RMSE'] = np.nan # Mark as NaN if model not selected or data prep failed
             metrics['NN_MAE'] = np.nan


        # --- Model 4: LSTM ---
        lstm_model = None
        lstm_forecast = None
        lstm_residuals_test = None
        group_lstm_history = None
        if use_lstm_model and X_train_seq is not None: # Only train if data prep was successful
            try:
                # Use a subdirectory for each group's models
                group_model_dir = os.path.join(output_dir, f"group_models_{group_id}")
                os.makedirs(group_model_dir, exist_ok=True)
                # Keras checkpoint path
                checkpoint_path = os.path.join(group_model_dir, f"lstm_model_group_{group_id}.keras")

                # --- Check for and load existing model checkpoint ---
                if os.path.exists(checkpoint_path):
                    try:
                        lstm_model = load_model(checkpoint_path)
                        # print(f"  Loaded existing LSTM model for group {group_id} from checkpoint.")
                        loaded_from_checkpoint = True
                    except Exception as load_e:
                        # print(f"  Error loading existing LSTM model for group {group_id}: {load_e}. Building new model.")
                        lstm_model = build_lstm_model(lstm_look_back, features=len(features_to_scale), bidirectional=use_bidirectional_lstm)
                        loaded_from_checkpoint = False
                else:
                    # print(f"  No existing LSTM checkpoint found for group {group_id}. Building new model.")
                    lstm_model = build_lstm_model(lstm_look_back, features=len(features_to_scale), bidirectional=use_bidirectional_lstm)
                    loaded_from_checkpoint = False
                # --- End of checkpoint loading ---


                callbacks = [
                    EarlyStopping(monitor='loss', patience=10, verbose=0, restore_best_weights=True),
                    ModelCheckpoint(filepath=checkpoint_path, monitor='loss', save_best_only=True, verbose=0, save_weights_only=False),
                    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0)
                ]

                # Train LSTM model (relying on Early Stopping)
                history = lstm_model.fit(X_train_seq, y_train_seq, epochs=lstm_epochs, batch_size=lstm_batch_size,
                                        verbose=0, callbacks=callbacks, shuffle=False)
                group_lstm_history = history.history['loss'] # Collect loss history


                # Ensure best model is loaded after training if checkpoint was used
                if os.path.exists(checkpoint_path):
                    try:
                         lstm_model = load_model(checkpoint_path)
                         # print(f"  Loaded best LSTM model after training for group {group_id}.")
                    except Exception as load_e:
                         # print(f"  Error loading best LSTM model after training for group {group_id}: {load_e}.")
                         pass # Continue with the model object from fit if loading fails


                # Prepare test data for LSTM prediction (rolling forecast)
                # Need the last `lstm_look_back` points of the scaled data (all features)
                # from the END OF TRAINING to start the forecasting sequence.
                if scaled_ts_data is None or len(scaled_ts_data) < len(test_data) + lstm_look_back:
                     raise ValueError("Insufficient scaled test data for LSTM rolling forecast.")

                lstm_input_sequence_scaled = scaled_ts_data[-(len(test_data) + lstm_look_back):]

                lstm_forecast_scaled_list = []
                # Get the initial input window from the end of the combined scaled data
                current_input_window_lstm = lstm_input_sequence_scaled[:lstm_look_back] # Shape (look_back, n_features)


                for i in range(len(test_data)):
                    # Reshape for prediction (1, look_back, n_features)
                    input_for_pred = np.reshape(current_input_window_lstm, (1, lstm_look_back, len(features_to_scale)))
                    pred_scaled = lstm_model.predict(input_for_pred, verbose=0)[0, 0] # Predict the next sales value (the first feature)
                    lstm_forecast_scaled_list.append(pred_scaled)

                    # Prepare input for the next step: roll the sequence and add the predicted value
                    next_input_window_lstm = np.roll(current_input_window_lstm, -1, axis=0) # Roll along the time dimension
                    # Replace the last step's sales feature with the prediction
                    next_input_window_lstm[-1, features_to_scale.index(sales_column)] = pred_scaled

                    # Similar limitations for recursive forecasting with features apply here

                    current_input_window_lstm = next_input_window_lstm # Update input for the next step


                lstm_forecast_scaled = np.array(lstm_forecast_scaled_list).reshape(-1, 1)
                # Inverse transform
                temp_scaled_array_lstm = np.zeros((lstm_forecast_scaled.shape[0], len(features_to_scale)))
                temp_scaled_array_lstm[:, features_to_scale.index(sales_column)] = lstm_forecast_scaled.flatten()

                # Use the main scaler for inverse transform
                lstm_forecast = scaler.inverse_transform(temp_scaled_array_lstm)[:, features_to_scale.index(sales_column)].flatten()


                group_forecasts['LSTM'] = lstm_forecast.tolist() # Convert to list

                # Calculate test residuals for LSTM
                lstm_residuals_test = test_data[sales_column].values - lstm_forecast
                residual_data['LSTM'] = lstm_residuals_test.tolist()

                lstm_rmse = np.sqrt(np.mean(lstm_residuals_test**2))
                lstm_mae = np.mean(np.abs(lstm_residuals_test))
                metrics['LSTM_RMSE'] = lstm_rmse
                metrics['LSTM_MAE'] = lstm_mae
                # print(f"  LSTM Forecast generated for {group_id}. RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}")

            except Exception as e:
                error_message = (error_message + "\n" if error_message else "") + f"LSTM Error: {e}"
                # print(f"  Error training/forecasting LSTM for group {group_id}: {e}")
                metrics['LSTM_RMSE'] = np.nan
                metrics['LSTM_MAE'] = np.nan
                group_forecasts['LSTM'] = None
                residual_data['LSTM'] = None
                group_lstm_history = None
        else:
             metrics['LSTM_RMSE'] = np.nan # Mark as NaN if model not selected or data prep failed
             metrics['LSTM_MAE'] = np.nan

        # --- Model 5: Hybrid ARIMA-LSTM ---
        hybrid_lstm_model = None
        hybrid_scaler = None
        hybrid_forecast = None
        hybrid_residuals_test = None
        # Hybrid history is for residual model, could store if needed but maybe less critical than main LSTM
        # group_hybrid_lstm_history = None
        if use_hybrid_model and use_arima_model and use_lstm_model: # Only train if Hybrid, ARIMA, and LSTM are selected
            try:
                # Check if ARIMA was successful AND produced test residuals
                if arima_model is not None and group_forecasts.get('ARIMA') is not None and residual_data.get('ARIMA') is not None:

                    # Need the ARIMA *training* predictions to calculate training residuals
                    # ARIMA model object should ideally have a method for in-sample prediction.
                    # If not directly available, a common workaround is to refit on the training data
                    # and predict over the training period, or use the stored model object if available.
                    # Assuming arima_model object is available and fitted on train_data

                    # Generate in-sample predictions from the fitted ARIMA model on the training data
                    # Check if arima_model exists and is fitted
                    if arima_model and hasattr(arima_model, 'predict_in_sample'):
                        arima_train_pred = arima_model.predict_in_sample()

                        # Align residuals with train_data
                        # The length of predict_in_sample output can be less than train_data due to differencing.
                        # Need to align the residuals correctly.
                        start_index_residual = len(train_data) - len(arima_train_pred)
                        if start_index_residual < 0: # Should not be negative for in-sample
                            start_index_residual = 0
                        residuals_train = train_data[sales_column].iloc[start_index_residual:] - arima_train_pred


                        hybrid_scaler = StandardScaler()
                        # Reshape for scaler fit_transform
                        scaled_residuals_train = hybrid_scaler.fit_transform(residuals_train.values.reshape(-1, 1))

                        # Create dataset for LSTM on residuals
                        # Check if scaled_residuals_train has enough data points for the look_back period
                        if len(scaled_residuals_train) <= lstm_look_back:
                             raise ValueError(f"Not enough ARIMA residuals ({len(scaled_residuals_train)}) for the specified LSTM look_back period ({lstm_look_back}) for Hybrid model. Need at least {lstm_look_back + 1}.")

                        X_hybrid_train, y_hybrid_train = create_sequence_dataset(scaled_residuals_train, lstm_look_back)

                        if X_hybrid_train.shape[0] > 0:
                             # print(f"  Training LSTM on ARIMA residuals for {group_id}...")
                             # LSTM on residuals always has 1 feature
                             # Use a subdirectory for each group's models
                             group_model_dir = os.path.join(output_dir, f"group_models_{group_id}")
                             os.makedirs(group_model_dir, exist_ok=True) # Ensure directory exists
                             hybrid_checkpoint_path = os.path.join(group_model_dir, f"hybrid_lstm_model_group_{group_id}.keras")

                             # --- Check for and load existing hybrid model checkpoint ---
                             if os.path.exists(hybrid_checkpoint_path):
                                 try:
                                     hybrid_lstm_model = load_model(hybrid_checkpoint_path)
                                     # print(f"  Loaded existing Hybrid LSTM model for group {group_id} from checkpoint.")
                                     loaded_from_checkpoint = True
                                 except Exception as load_e:
                                     # print(f"  Error loading existing Hybrid LSTM model for group {group_id}: {load_e}. Building new model.")
                                     hybrid_lstm_model = build_lstm_model(lstm_look_back, features=1, bidirectional=use_bidirectional_lstm)
                                     loaded_from_checkpoint = False
                             else:
                                 # print(f"  No existing Hybrid LSTM checkpoint found for group {group_id}. Building new model.")
                                 hybrid_lstm_model = build_lstm_model(lstm_look_back, features=1, bidirectional=use_bidirectional_lstm)
                                 loaded_from_checkpoint = False
                             # --- End of checkpoint loading ---


                             hybrid_callbacks = [
                                 EarlyStopping(monitor='loss', patience=10, verbose=0, restore_best_weights=True),
                                 ModelCheckpoint(filepath=hybrid_checkpoint_path, monitor='loss', save_best_only=True, verbose=0, save_weights_only=False),
                                 ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0)
                             ]

                             # Train Hybrid LSTM model (relying on Early Stopping)
                             hybrid_history = hybrid_lstm_model.fit(X_hybrid_train, y_hybrid_train, epochs=lstm_epochs, batch_size=lstm_batch_size,
                                                                    verbose=0, callbacks=hybrid_callbacks, shuffle=False)
                             # Hybrid history is for residual model, could store if needed but maybe less critical than main LSTM
                             # group_hybrid_lstm_history = hybrid_history.history['loss']

                             # Ensure best model is loaded after training if checkpoint was used
                             if os.path.exists(hybrid_checkpoint_path):
                                 try:
                                     hybrid_lstm_model = load_model(hybrid_checkpoint_path)
                                     # print(f"  Loaded best Hybrid LSTM model after training for group {group_id}.")
                                 except Exception as load_e:
                                     # print(f"  Error loading best Hybrid LSTM model after training for group {group_id}: {load_e}.")
                                     pass # Continue with the model object from fit if loading fails


                             # Generate residual forecast using the trained residual LSTM model
                             # Need the last `lstm_look_back` scaled residuals from the training set
                             # Ensure there are enough residuals
                             if len(scaled_residuals_train) < lstm_look_back:
                                  raise ValueError("Not enough scaled ARIMA residuals to form the initial LSTM input sequence for forecasting.")

                             hybrid_input_sequence = scaled_residuals_train[-lstm_look_back:].flatten()

                             residual_forecast_scaled = []
                             for _ in range(len(test_data)):
                                 current_input = np.reshape(hybrid_input_sequence, (1, lstm_look_back, 1))
                                 pred_scaled = hybrid_lstm_model.predict(current_input, verbose=0)[0, 0]
                                 residual_forecast_scaled.append(pred_scaled)

                                 # Update the input sequence for the next prediction (rolling forecast)
                                 hybrid_input_sequence = np.roll(hybrid_input_sequence, -1)
                                 hybrid_input_sequence[-1] = pred_scaled # Feed the prediction back in

                             residual_forecast = hybrid_scaler.inverse_transform(np.array(residual_forecast_scaled).reshape(-1, 1)).flatten()

                             # ARIMA forecast was already calculated earlier
                             arima_test_forecast_values = np.array(group_forecasts.get('ARIMA')) # Get ARIMA forecast from results dict safely

                             if arima_test_forecast_values is None or len(arima_test_forecast_values) != len(residual_forecast):
                                 # This case should ideally not happen if ARIMA was successful, but safety check
                                 raise ValueError(f"ARIMA forecast missing or length ({len(arima_test_forecast_values) if arima_test_forecast_values is not None else 'None'}) does not match residual forecast length ({len(residual_forecast)}) for Hybrid model.")

                             hybrid_forecast = arima_test_forecast_values + residual_forecast # Combine ARIMA forecast and residual forecast
                             group_forecasts['Hybrid'] = hybrid_forecast.tolist() # Convert to list

                             # Calculate test residuals for Hybrid
                             hybrid_residuals_test = test_data[sales_column].values - hybrid_forecast
                             residual_data['Hybrid'] = hybrid_residuals_test.tolist()

                             hybrid_rmse = np.sqrt(np.mean(hybrid_residuals_test**2))
                             hybrid_mae = np.mean(np.abs(hybrid_residuals_test))
                             metrics['Hybrid_RMSE'] = hybrid_rmse
                             metrics['Hybrid_MAE'] = hybrid_mae
                             # print(f"  Hybrid Forecast generated for {group_id}. RMSE: {hybrid_rmse:.4f}, MAE: {hybrid_mae:.4f}")

                             # Save hybrid scaler
                             hybrid_scaler_filename = os.path.join(output_dir, f'hybrid_scaler_group_{group_id}.joblib')
                             joblib.dump(hybrid_scaler, hybrid_scaler_filename)

                        else:
                            error_message = (error_message + "\n" if error_message else "") + "Hybrid Error: Not enough residual data after creating look-back dataset for LSTM training."
                            # print(f"  Skipping Hybrid LSTM for {group_id}: Not enough residual data after creating look-back...")
                            metrics['Hybrid_RMSE'] = np.nan
                            metrics['Hybrid_MAE'] = np.nan
                            group_forecasts['Hybrid'] = None
                            residual_data['Hybrid'] = None

                    else:
                        error_message = (error_message + "\n" if error_message else "") + "Hybrid Error: ARIMA model not available or not fitted correctly to calculate residuals."
                        # print(f"  Skipping Hybrid for {group_id}: ARIMA model not available or not fitted.")
                        metrics['Hybrid_RMSE'] = np.nan
                        metrics['Hybrid_MAE'] = np.nan
                        group_forecasts['Hybrid'] = None
                        residual_data['Hybrid'] = None

                else:
                    error_message = (error_message + "\n" if error_message else "") + "Hybrid Error: ARIMA model did not run successfully or did not produce necessary outputs for hybrid modeling."
                    # print(f"  Skipping Hybrid for {group_id}: ARIMA prerequisites not met.")
                    metrics['Hybrid_RMSE'] = np.nan
                    metrics['Hybrid_MAE'] = np.nan
                    group_forecasts['Hybrid'] = None
                    residual_data['Hybrid'] = None

            except Exception as e:
                error_message = (error_message + "\n" if error_message else "") + f"Hybrid ARIMA-LSTM Error: {e}"
                # print(f"  Error training/forecasting Hybrid ARIMA-LSTM for group {group_id}: {e}")
                metrics['Hybrid_RMSE'] = np.nan
                metrics['Hybrid_MAE'] = np.nan
                group_forecasts['Hybrid'] = None
                residual_data['Hybrid'] = None
                # group_hybrid_lstm_history = None # Ensure history is None if error
        else:
             metrics['Hybrid_RMSE'] = np.nan # Mark as NaN if model not selected or prerequisites not met
             metrics['Hybrid_MAE'] = np.nan


        # --- Return Results ---
        return group_id, group_forecasts, test_data_json, metrics, error_message, residual_data, group_lstm_history, nn_history # Include NN history


    except Exception as e:
        # Catch any unexpected errors during the group processing
        tb_str = traceback.format_exc() # Get detailed traceback
        error_message = (error_message + "\n" if error_message else "") + f"Unexpected Error in Group Process {group_id}: {e}\n{tb_str}"
        # print(f"  FATAL Error processing group {group_id}: {e}\n{tb_str}")
        # Ensure metrics reflect the failure
        metrics.update({k: np.nan for k in ['ARIMA_RMSE', 'ARIMA_MAE', 'MA_RMSE', 'MA_MAE', 'NN_RMSE', 'NN_MAE', 'LSTM_RMSE', 'LSTM_MAE', 'Hybrid_RMSE', 'Hybrid_MAE']})
        return group_id, None, None, metrics, error_message, None, None, None # Return None for all results on fatal error


# --- Main Function to Run Forecasting ---
def run_forecasting(data_full_file_path, date_column, sales_column, grouping_column,
                    time_series_frequency, test_size_days, lstm_look_back, lstm_epochs,
                    lstm_batch_size, use_arima_model, use_ma_model, ma_q_order, use_nn_model,
                    nn_layers, nn_units, use_lstm_model, use_bidirectional_lstm, use_hybrid_model,
                    add_lagged_features, max_workers, output_dir):
    """
    Loads data, performs feature engineering, splits data, trains models
    in parallel, and collects results.
    """

    st.subheader("Step 1: Data Loading and Preparation")
    try:
        if not os.path.exists(data_full_file_path):
            st.error(f"Error: Data file not found at '{data_full_file_path}'")
            return # Stop execution if file not found

        # Determine file type and read
        file_ext = os.path.splitext(data_full_file_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(data_full_file_path)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(data_full_file_path)
        else:
            st.error(f"Unsupported file format: {file_ext}. Please use .csv or .xlsx.")
            return

        st.success(f"Successfully loaded data from '{data_full_file_path}'")

        # Validate required columns exist
        required_cols = [date_column, sales_column, grouping_column]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Error: Missing required columns in data file: {', '.join(missing)}")
            return

        # Data Cleaning and Preparation
        df = df.dropna(subset=[date_column, sales_column, grouping_column]) # Drop rows with missing essential data
        df[date_column] = pd.to_datetime(df[date_column]) # Convert date column to datetime objects
        df[sales_column] = pd.to_numeric(df[sales_column], errors='coerce') # Ensure sales is numeric, coerce errors
        df = df.dropna(subset=[sales_column]) # Drop rows where sales could not be converted to numeric

        # Sort by group and date
        df = df.sort_values(by=[grouping_column, date_column])

        # Add general time-based features *before* splitting into groups
        df['Year'] = df[date_column].dt.year
        df['Month'] = df[date_column].dt.month
        df['Day'] = df[date_column].dt.day
        df['DayOfWeek'] = df[date_column].dt.dayofweek # Monday=0, Sunday=6
        df['DayOfYear'] = df[date_column].dt.dayofyear
        df['WeekOfYear'] = df[date_column].dt.isocalendar().week.astype(int) # Use isocalendar for week number

        st.success("Data cleaning and preparation complete.")
        st.write("Data Head:", df.head())
        st.write("Data Info:", df.info())

    except Exception as e:
        st.error(f"Error during data loading and preparation: {e}")
        st.exception(e)
        return


    st.subheader("Step 2: Training Models (Parallel Processing)")
    group_ids = df[grouping_column].unique()
    total_groups = len(group_ids)
    st.info(f"Found {total_groups} unique groups. Processing in parallel...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Configuration dictionary to pass to each process
    config = {
        'date_column': date_column,
        'sales_column': sales_column,
        'grouping_column': grouping_column,
        'time_series_frequency': time_series_frequency,
        'test_size_days': test_size_days,
        'lstm_look_back': lstm_look_back,
        'lstm_epochs': lstm_epochs,
        'lstm_batch_size': lstm_batch_size,
        'use_arima_model': use_arima_model,
        'use_ma_model': use_ma_model,
        'ma_q_order': ma_q_order,
        'use_nn_model': use_nn_model,
        'nn_layers': nn_layers,
        'nn_units': nn_units,
        'use_lstm_model': use_lstm_model,
        'use_bidirectional_lstm': use_bidirectional_lstm,
        'use_hybrid_model': use_hybrid_model, # Pass hybrid flag
        'add_lagged_features': add_lagged_features,
        'output_dir': output_dir # Pass output directory to processes
    }

    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for group_id in group_ids:
            group_df = df[df[grouping_column] == group_id].copy()
            # Convert group_df to JSON string to pass to process
            # Need to convert datetime index to string before JSON serialization
            group_df_for_json = group_df.copy()
            # Keep original date column for split, then set index inside the process after feature engineering
            # group_df_for_json.reset_index(inplace=True) # Reset index before JSON to keep date as column
            group_df_for_json[date_column] = group_df_for_json[date_column].astype(str) # Convert date column to string
            group_df_json = group_df_for_json.to_json(orient='split') # Use 'split' for easier reconstruction


            future = executor.submit(train_and_forecast_group, group_id, group_df_json, config)
            futures.append(future)

        # Use Streamlit progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()

        completed_tasks = 0
        for future in as_completed(futures):
            completed_tasks += 1
            progress = completed_tasks / total_groups
            progress_bar.progress(progress)
            status_text.text(f"Processing group {completed_tasks}/{total_groups}...")

            group_id, group_forecasts, test_data_json, metrics, error_message, residual_data, lstm_history, nn_history = future.result()

            if error_message:
                st.warning(f"Error processing group {group_id}: {error_message}")
                all_errors_list.append({'group_id': group_id, 'error': error_message})

            if metrics:
                all_evaluation_metrics_list.append(metrics)

            if group_forecasts:
                all_forecasts_dict[group_id] = group_forecasts

            if test_data_json:
                 # Reconstruct test_data DataFrame from JSON string
                 test_data_group = pd.read_json(test_data_json, orient='split')
                 test_data_group[date_column] = pd.to_datetime(test_data_group[date_column]) # Convert date back to datetime
                 test_data_group.set_index(date_column, inplace=True) # Set index back
                 all_test_data_dict[group_id] = test_data_group


            if residual_data:
                 all_residual_plots_data[group_id] = residual_data # Store residual data

            if lstm_history:
                 all_lstm_history_data[group_id] = lstm_history # Store LSTM history

            if nn_history:
                 all_nn_history_data[group_id] = nn_history # Store NN history


        status_text.text("Parallel processing complete.")
        progress_bar.empty() # Hide progress bar after completion

    st.success("Step 2: Model training and forecasting complete for all groups.")

    # --- Step 3: Consolidate and Evaluate Results ---
    st.subheader("Step 3: Consolidate and Evaluate Results")

    # Consolidate evaluation metrics
    if all_evaluation_metrics_list:
        evaluation_df = pd.DataFrame(all_evaluation_metrics_list)
        st.write("Evaluation Metrics (RMSE and MAE per group):", evaluation_df)

        # Calculate overall average metrics
        avg_metrics = evaluation_df.mean(numeric_only=True).to_dict()
        st.write("Overall Average Metrics:")
        for metric, value in avg_metrics.items():
             if not np.isnan(value):
                 st.write(f"- {metric}: {value:.4f}")
             else:
                 st.write(f"- {metric}: N/A (Model not selected or failed)")

        # Option to download evaluation metrics
        csv_metrics = evaluation_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Evaluation Metrics as CSV",
            data=csv_metrics,
            file_name='evaluation_metrics.csv',
            mime='text/csv',
        )
    else:
        st.warning("No evaluation metrics were collected.")

    # Consolidate forecasts and actual test data
    consolidated_forecasts = []
    consolidated_test_data = []
    sample_group_ids = [] # To store a few group IDs for plotting

    # Check if there are any successful forecasts before consolidation
    has_successful_forecasts = any(all_forecasts_dict.values())

    if all_forecasts_dict and all_test_data_dict:
         for group_id in all_forecasts_dict.keys(): # Iterate through groups with forecasts
             if group_id in all_test_data_dict: # Ensure test data exists for this group
                 forecasts = all_forecasts_dict[group_id]
                 test_data_group = all_test_data_dict[group_id].copy() # Use copied test data

                 # Create a DataFrame for forecasts for this group
                 # Ensure the index aligns with the test_data_group
                 forecast_df_group = pd.DataFrame(test_data_group.index)
                 forecast_df_group.columns = [date_column] # Set date column name
                 forecast_df_group.set_index(date_column, inplace=True) # Set date as index

                 # Add actual sales from test data
                 forecast_df_group['Actual'] = test_data_group[sales_column]

                 # Add forecast columns for each model that ran successfully for this group
                 for model_name, forecast_values in forecasts.items():
                      if forecast_values is not None and len(forecast_values) == len(test_data_group):
                           forecast_df_group[f'{model_name}_Forecast'] = forecast_values
                      else:
                           forecast_df_group[f'{model_name}_Forecast'] = np.nan # Mark as NaN if forecast missing or wrong length


                 forecast_df_group[grouping_column] = group_id # Add grouping column
                 consolidated_forecasts.append(forecast_df_group)

                 # Also consolidate test data separately if needed later, but consolidating forecasts is key for plotting
                 # consolidated_test_data.append(test_data_group.reset_index().assign(**{grouping_column: group_id})) # Append test data with grouping column


         if consolidated_forecasts:
             consolidated_forecasts_df = pd.concat(consolidated_forecasts)
             st.write("Consolidated Forecasts and Actuals (Sample Head):", consolidated_forecasts_df.head())

             # Save consolidated forecasts
             try:
                 consolidated_forecasts_path = os.path.join(output_dir, 'consolidated_forecasts.csv')
                 consolidated_forecasts_df.to_csv(consolidated_forecasts_path)
                 st.success(f"Consolidated forecasts saved to '{consolidated_forecasts_path}'")

                 # Option to download consolidated forecasts
                 csv_forecasts = consolidated_forecasts_df.to_csv().encode('utf-8')
                 st.download_button(
                     label="Download Consolidated Forecasts as CSV",
                     data=csv_forecasts,
                     file_name='consolidated_forecasts.csv',
                     mime='text/csv',
                 )

             except Exception as e:
                 st.error(f"Error saving consolidated forecasts: {e}")
                 st.exception(e)
         else:
             st.warning("No consolidated forecasts DataFrame could be created.")
             consolidated_forecasts_df = pd.DataFrame() # Initialize empty DataFrame


    else:
        st.warning("No forecasts or test data were collected from group processing.")
        consolidated_forecasts_df = pd.DataFrame() # Initialize empty DataFrame


    # --- Step 4: Save Models and Results ---
    st.subheader("Step 4: Saving Results")

    # Save models are handled within train_and_forecast_group process
    # Scalers are also saved within the process

    # Save evaluation metrics (already handled in Step 3)
    # Save consolidated forecasts (already handled in Step 3)

    # Save residual data for plotting
    if all_residual_plots_data:
        try:
            residual_plots_data_path = os.path.join(output_dir, 'residual_plots_data.joblib')
            joblib.dump(all_residual_plots_data, residual_plots_data_path)
            st.success(f"Residuals data for plotting saved to '{residual_plots_data_path}'")
        except Exception as e:
            st.error(f"Error saving residual plots data: {e}")
            st.exception(e)

    # Save LSTM training history
    if all_lstm_history_data:
        try:
            lstm_history_path = os.path.join(output_dir, 'lstm_training_history.joblib')
            joblib.dump(all_lstm_history_data, lstm_history_path)
            st.success(f"LSTM training history saved to '{lstm_history_path}'")
        except Exception as e:
            st.error(f"Error saving LSTM training history: {e}")
            st.exception(e)

    # Save NN training history
    if all_nn_history_data:
        try:
            nn_history_path = os.path.join(output_dir, 'nn_training_history.joblib')
            joblib.dump(all_nn_history_data, nn_history_path)
            st.success(f"NN training history saved to '{nn_history_path}'")
        except Exception as e:
            st.error(f"Error saving NN training history: {e}")
            st.exception(e)


    # Report errors encountered during processing
    if all_errors_list:
        st.subheader("Processing Errors")
        st.error("The following errors occurred during group processing:")
        for err in all_errors_list:
            st.write(f"- Group ID {err['group_id']}: {err['error']}")


    st.success("Step 4: Saving results complete.")


    # --- Step 5: Visualize Results ---
    st.subheader("Step 5: Visualize Results")

    # Function to display individual model performance
    def display_model_performance(group_id, consolidated_df, evaluation_df, residuals_data, date_col, sales_col, model_name, output_dir):
        st.subheader(f"{model_name} Performance for Group {group_id}")

        # Filter data for the selected group
        group_plot_df = consolidated_df[consolidated_df[grouping_column] == group_id].copy()

        if not group_plot_df.empty:
            # Display Metrics for the specific model
            model_metrics = evaluation_df[evaluation_df['group_id'] == group_id]
            if not model_metrics.empty:
                 st.write(f"Metrics for {model_name}:")
                 rmse_col = f'{model_name}_RMSE'
                 mae_col = f'{model_name}_MAE'
                 rmse = model_metrics.iloc[0].get(rmse_col, np.nan)
                 mae = model_metrics.iloc[0].get(mae_col, np.nan)

                 if not np.isnan(rmse):
                     st.write(f"- RMSE: {rmse:.4f}")
                 else:
                     st.write(f"- RMSE: N/A (Model failed or not selected)")

                 if not np.isnan(mae):
                     st.write(f"- MAE: {mae:.4f}")
                 else:
                     st.write(f"- MAE: N/A (Model failed or not selected)")
            else:
                 st.info(f"No evaluation metrics found for {model_name} for this group.")


            # Actual vs. Forecasted Plot for the specific model
            forecast_col = f'{model_name}_Forecast'
            if forecast_col in group_plot_df.columns and not group_plot_df[forecast_col].isnull().all():
                 plot_data = group_plot_df.reset_index().melt(
                    id_vars=[date_col, grouping_column],
                    value_vars=['Actual', forecast_col],
                    var_name='Series',
                    value_name='Sales'
                 )
                 plot_data['Series'] = plot_data['Series'].replace('Actual', 'Actual Sales')

                 fig = px.bar(plot_data, x=date_col, y='Sales', color='Series', barmode='group',
                               title=f'Actual vs. {model_name} Forecasted Sales for Group {group_id}')
                 st.plotly_chart(fig, use_container_width=True)

                 # Save the plot
                 try:
                    group_plot_dir = os.path.join(output_dir, f"plots_{group_id}")
                    os.makedirs(group_plot_dir, exist_ok=True)
                    plot_html_path = os.path.join(group_plot_dir, f'{model_name.lower()}_forecast_plot_{group_id}.html')
                    fig.write_html(plot_html_path)
                    st.success(f"{model_name} forecast plot saved as HTML to '{plot_html_path}'")
                    try:
                        plot_png_path = os.path.join(group_plot_dir, f'{model_name.lower()}_forecast_plot_{group_id}.png')
                        fig.write_image(plot_png_path)
                        st.success(f"{model_name} forecast plot saved as PNG to '{plot_png_path}'")
                    except Exception as img_e:
                        st.warning(f"Could not save {model_name} plot as PNG (requires kaleido engine): {img_e}")
                 except Exception as e:
                    st.error(f"Error saving {model_name} forecast plot for group {group_id}: {e}")
                    st.exception(e)

            else:
                 st.info(f"No valid {model_name} forecast data available for plotting.")

            # Residuals Plot for the specific model
            if group_id in residuals_data and model_name in residuals_data[group_id] and residuals_data[group_id][model_name] is not None:
                 residual_data_group = residuals_data[group_id]
                 residual_df = pd.DataFrame({
                     'Date': group_plot_df.index,
                     f'{model_name}_Residuals': residual_data_group[model_name]
                 })
                 residual_df.set_index('Date', inplace=True)


                 if not residual_df.empty:
                    fig_residuals = px.line(residual_df.reset_index(), x='Date', y=f'{model_name}_Residuals',
                                            title=f'{model_name} Residuals over Time for Group {group_id}')
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Residuals")
                    st.plotly_chart(fig_residuals, use_container_width=True)

                    # Save the residual plot
                    try:
                        group_plot_dir = os.path.join(output_dir, f"plots_{group_id}")
                        os.makedirs(group_plot_dir, exist_ok=True)
                        residual_plot_html_path = os.path.join(group_plot_dir, f'{model_name.lower()}_residuals_plot_{group_id}.html')
                        fig_residuals.write_html(residual_plot_html_path)
                        st.success(f"{model_name} residuals plot saved as HTML to '{residual_plot_html_path}'")
                        try:
                            residual_plot_png_path = os.path.join(group_plot_dir, f'{model_name.lower()}_residuals_plot_{group_id}.png')
                            fig_residuals.write_image(residual_plot_png_path)
                            st.success(f"{model_name} residuals plot saved as PNG to '{residual_plot_png_path}'")
                        except Exception as img_e:
                            st.warning(f"Could not save {model_name} residuals plot as PNG (requires kaleido engine): {img_e}")
                    except Exception as e:
                        st.error(f"Error saving {model_name} residuals plot for group {group_id}: {e}")
                        st.exception(e)

                 else:
                    st.info(f"No valid {model_name} residual data available for plotting.")
            else:
                 st.info(f"No residual data available for {model_name} for this group.")


    # This 'try' block encompasses the entire visualization step
    try:
        # Check if there is data to plot
        if not consolidated_forecasts_df.empty and all_evaluation_metrics_list: # Ensure both are available
            st.write("Select a group to visualize forecasts and residuals.")

            # Get unique group IDs from the consolidated forecasts DataFrame
            sample_group_ids = consolidated_forecasts_df[grouping_column].unique().tolist()

            if sample_group_ids:
                selected_group_plot = st.selectbox("Choose a Group ID:", sample_group_ids)

                # Filter data for the selected group
                group_plot_df = consolidated_forecasts_df[consolidated_forecasts_df[grouping_column] == selected_group_plot].copy()

                if not group_plot_df.empty:

                    # --- Model Visualization Selection ---
                    st.subheader(f"Model Visualization Options for Group {selected_group_plot}")
                    visualization_mode = st.radio(
                        "Select Visualization Mode:",
                        ('Compare All Models', 'View Individual Model Performance', 'Compare Model Metrics') # Added 'Compare Model Metrics'
                    )

                    if visualization_mode == 'Compare All Models':
                        # --- Actual vs. Forecasted Sales Plot (Comparison) ---
                        st.subheader(f"Actual vs. Forecasted Sales Comparison for Group {selected_group_plot}")
                        plot_data = group_plot_df.reset_index().melt(
                            id_vars=[date_column, grouping_column],
                            value_vars=[col for col in group_plot_df.columns if 'Forecast' in col or col == 'Actual'],
                            var_name='Series',
                            value_name='Sales'
                        )
                        plot_data['Series'] = plot_data['Series'].replace('Actual', 'Actual Sales')
                        # Changed px.line to px.bar
                        fig = px.bar(plot_data, x=date_column, y='Sales', color='Series', barmode='group',
                                      title=f'Actual vs. Forecasted Sales for Group {selected_group_plot}')
                        st.plotly_chart(fig, use_container_width=True)

                        # Save the comparison plot
                        try:
                            group_plot_dir = os.path.join(output_dir, f"plots_{selected_group_plot}")
                            os.makedirs(group_plot_dir, exist_ok=True)
                            plot_html_path = os.path.join(group_plot_dir, f'forecast_comparison_plot_{selected_group_plot}.html')
                            fig.write_html(plot_html_path)
                            st.success(f"Comparison forecast plot saved as HTML to '{plot_html_path}'")
                            try:
                                plot_png_path = os.path.join(group_plot_dir, f'forecast_comparison_plot_{selected_group_plot}.png')
                                fig.write_image(plot_png_path)
                                st.success(f"Comparison forecast plot saved as PNG to '{plot_png_path}'")
                            except Exception as img_e:
                                st.warning(f"Could not save comparison plot as PNG (requires kaleido engine): {img_e}")
                        except Exception as e:
                            st.error(f"Error saving comparison forecast plot for group {selected_group_plot}: {e}")
                            st.exception(e)


                    elif visualization_mode == 'View Individual Model Performance':
                        # Identify available models based on columns in the consolidated dataframe
                        available_models = [col.replace('_Forecast', '') for col in group_plot_df.columns if '_Forecast' in col]
                        if available_models:
                             selected_model_plot = st.radio("Choose a Model:", available_models)
                             # Pass the full evaluation_df to the display function
                             display_model_performance(selected_group_plot, consolidated_forecasts_df, pd.DataFrame(all_evaluation_metrics_list), all_residual_plots_data, date_column, sales_column, selected_model_plot, output_dir)

                             # --- Training Loss Plot (for Sequence Models - if selected model is NN or LSTM) ---
                             if selected_model_plot in ['LSTM', 'NN', 'Hybrid']: # Include Hybrid for potential future residual model loss
                                  st.subheader(f"Training Loss History for Group {selected_group_plot} ({selected_model_plot})")
                                  if selected_model_plot == 'LSTM':
                                       loss_history = all_lstm_history_data.get(selected_group_plot)
                                       model_label = 'LSTM'
                                  elif selected_model_plot == 'NN': # selected_model_plot == 'NN'
                                       loss_history = all_nn_history_data.get(selected_group_plot)
                                       model_label = 'NN'
                                  elif selected_model_plot == 'Hybrid':
                                      # Assuming you might want to show the loss of the residual model within Hybrid
                                      # Currently, the script doesn't store hybrid residual model loss history explicitly
                                      # This part would need modification if you store hybrid residual loss
                                      loss_history = None # Placeholder - need to store hybrid residual loss if desired
                                      model_label = 'Hybrid Residual LSTM'
                                      st.info("Training loss history for the Hybrid model's residual component is not currently collected.") # Inform user


                                  if loss_history:
                                       loss_df = pd.DataFrame({'Epoch': range(1, len(loss_history) + 1), 'Loss': loss_history, 'Model': model_label})
                                       fig_loss = px.line(loss_df, x='Epoch', y='Loss',
                                                          title=f'Training Loss History for Group {selected_group_plot} ({model_label})')
                                       st.plotly_chart(fig_loss, use_container_width=True)

                                       # Save the loss plot
                                       try:
                                            group_plot_dir = os.path.join(output_dir, f"plots_{selected_group_plot}")
                                            os.makedirs(group_plot_dir, exist_ok=True)
                                            loss_plot_html_path = os.path.join(group_plot_dir, f'{model_label.lower().replace(" ", "_")}_training_loss_plot_{selected_group_plot}.html') # Adjust filename for hybrid
                                            fig_loss.write_html(loss_plot_html_path)
                                            st.success(f"{model_label} training loss plot saved as HTML to '{loss_plot_html_path}'")
                                            try:
                                                loss_plot_png_path = os.path.join(group_plot_dir, f'{model_label.lower().replace(" ", "_")}_training_loss_plot_{selected_group_plot}.png') # Adjust filename for hybrid
                                                fig_loss.write_image(loss_plot_png_path)
                                                st.success(f"{model_label} training loss plot saved as PNG to '{loss_plot_png_path}'")
                                            except Exception as img_e:
                                                st.warning(f"Could not save {model_label} training loss plot as PNG (requires kaleido engine): {img_e}")
                                       except Exception as e:
                                            st.error(f"Error saving {model_label} training loss plot for group {selected_group_plot}: {e}")
                                            st.exception(e)

                                  else:
                                       # Only show this if it wasn't the placeholder message for Hybrid
                                       if selected_model_plot != 'Hybrid':
                                            st.info(f"No training history available for {selected_model_plot} for this group.")


                        else:
                             st.info("No individual models available to visualize for this group.")

                    elif visualization_mode == 'Compare Model Metrics':
                         st.subheader(f"Model Performance Metrics Comparison for Group {selected_group_plot}")
                         # Filter metrics for the selected group
                         group_metrics_df = pd.DataFrame(all_evaluation_metrics_list)
                         group_metrics_df = group_metrics_df[group_metrics_df['group_id'] == selected_group_plot]

                         if not group_metrics_df.empty:
                             # Prepare data for the metrics bar chart
                             metrics_data = []
                             for col in group_metrics_df.columns:
                                 if col.endswith('_RMSE'):
                                     model_name = col.replace('_RMSE', '')
                                     rmse_value = group_metrics_df.iloc[0][col]
                                     mae_value = group_metrics_df.iloc[0].get(f'{model_name}_MAE', np.nan) # Get MAE if exists
                                     if not np.isnan(rmse_value):
                                         metrics_data.append({'Model': model_name, 'Metric Type': 'RMSE', 'Value': rmse_value})
                                     if not np.isnan(mae_value):
                                         metrics_data.append({'Model': model_name, 'Metric Type': 'MAE', 'Value': mae_value})

                             if metrics_data:
                                 metrics_plot_df = pd.DataFrame(metrics_data)
                                 fig_metrics = px.bar(metrics_plot_df, x='Model', y='Value', color='Metric Type', barmode='group',
                                                      title=f'RMSE and MAE Comparison for Group {selected_group_plot}')
                                 st.plotly_chart(fig_metrics, use_container_width=True)

                                 # Save the metrics comparison plot
                                 try:
                                    group_plot_dir = os.path.join(output_dir, f"plots_{selected_group_plot}")
                                    os.makedirs(group_plot_dir, exist_ok=True)
                                    metrics_plot_html_path = os.path.join(group_plot_dir, f'metrics_comparison_plot_{selected_group_plot}.html')
                                    fig_metrics.write_html(metrics_plot_html_path)
                                    st.success(f"Metrics comparison plot saved as HTML to '{metrics_plot_html_path}'")
                                    try:
                                        metrics_plot_png_path = os.path.join(group_plot_dir, f'metrics_comparison_plot_{selected_group_plot}.png')
                                        fig_metrics.write_image(metrics_plot_png_path)
                                        st.success(f"Metrics comparison plot saved as PNG to '{metrics_plot_png_path}'")
                                    except Exception as img_e:
                                        st.warning(f"Could not save metrics comparison plot as PNG (requires kaleido engine): {img_e}")
                                 except Exception as e:
                                    st.error(f"Error saving metrics comparison plot for group {selected_group_plot}: {e}")
                                    st.exception(e)


                             else:
                                 st.info("No valid metrics available for plotting for this group.")
                         else:
                             st.info(f"No evaluation metrics found for group {selected_group_plot}.")


                else:
                    st.info(f"No data found for the sample group ID: {selected_group_plot} for plotting forecasts/residuals.")
            else:
                st.info("No groups available to visualize forecasts.")
        else:
            st.info("Consolidated forecasts DataFrame or Evaluation Metrics are empty, cannot plot Actual vs. Forecasted Sales or compare metrics.")

    # This 'except' block catches errors from the visualization 'try' block above
    except Exception as e:
        st.warning(f"Could not plot actual vs. forecasted sales or residuals: {e}")
        st.exception(e)


    # Note: Saving results (metrics, forecasts, models) was moved to Step 4 and happens after parallel processing finishes but before plotting.
    # This ensures results are saved even if plotting fails for a specific group.


# --- Main App Execution ---
if start_button:
    # Check if at least one model is selected
    if not (use_arima_model or use_ma_model or use_nn_model or use_lstm_model or use_hybrid_model):
        st.warning("Please select at least one model to run.")
    else:
        # Check for Hybrid model dependencies
        if use_hybrid_model and (not use_arima_model or not use_lstm_model):
             st.warning("Hybrid ARIMA-LSTM model requires both ARIMA and LSTM models to be selected.")
        else:
             # Run the forecasting process
             run_forecasting(data_full_file_path, date_column, sales_column, grouping_column,
                             time_series_frequency, test_size_days, lstm_look_back, lstm_epochs,
                             lstm_batch_size, use_arima_model, use_ma_model, ma_q_order, use_nn_model,
                             nn_layers, nn_units, use_lstm_model, use_bidirectional_lstm, use_hybrid_model,
                             add_lagged_features, max_workers, output_dir)


st.markdown("---") # Separator
st.write("Script execution finished.")
