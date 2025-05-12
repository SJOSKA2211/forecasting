import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, Bidirectional, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import holidays


def create_sequences(data, n_steps_in, n_steps_out=1):
    """
    Create input/output sequences for time series forecasting.

    Args:
        data: Array of time series data
        n_steps_in: Number of time steps for input (lookback window)
        n_steps_out: Number of time steps to predict (forecast horizon)

    Returns:
        X: Input sequences
        y: Target sequences
    """
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in)])
        y.append(data[(i + n_steps_in):(i + n_steps_in + n_steps_out)])

    return np.array(X), np.array(y)


def create_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, bidirectional=False):
    """
    Create an LSTM model for time series forecasting.

    Args:
        input_shape: Shape of input data (timesteps, features)
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        bidirectional: Whether to use Bidirectional LSTM

    Returns:
        model: Compiled LSTM model
    """
    model = Sequential()

    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape))
    else:
        model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))

    model.add(Dropout(dropout_rate))

    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units // 2)))
    else:
        model.add(LSTM(lstm_units // 2))

    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_nn_model(input_shape, units=64, dropout_rate=0.2):
    """
    Create a simple neural network model for time series forecasting.

    Args:
        input_shape: Shape of input data
        units: Number of neurons in hidden layers
        dropout_rate: Dropout rate for regularization

    Returns:
        model: Compiled neural network model
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units // 2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def generate_rolling_forecast(model, initial_sequence, steps, scaler=None, is_lstm=True):
    """
    Generate a rolling forecast where each prediction is fed back
    as input for the next prediction.

    Args:
        model: Trained model (LSTM or NN)
        initial_sequence: Initial input sequence
        steps: Number of steps to forecast
        scaler: Scaler object for inverse transformation
        is_lstm: Whether the model is LSTM (affects input reshaping)

    Returns:
        forecast: Array of forecasted values
    """
    forecast = []
    curr_seq = initial_sequence.copy()

    for _ in range(steps):
        # Reshape for prediction
        if is_lstm:
            input_seq = curr_seq.reshape(1, curr_seq.shape[0], curr_seq.shape[1])
        else:
            input_seq = curr_seq.reshape(1, -1)

        # Predict next value and append to forecast
        pred = model.predict(input_seq, verbose=0)[0]
        forecast.append(pred[0])

        # Update sequence for next prediction
        if is_lstm:
            curr_seq = np.roll(curr_seq, -1, axis=0)
            curr_seq[-1] = pred
        else:
            curr_seq = np.roll(curr_seq, -1)
            curr_seq[-1] = pred

    # Inverse transform if scaler is provided
    if scaler:
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    return np.array(forecast)


def evaluate_forecast(y_true, y_pred):
    """
    Calculate evaluation metrics for forecasts.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if not np.any(y_true == 0) else np.nan,
        'r2': r2_score(y_true, y_pred)
    }
    return metrics

def run_all_forecasts(df_processed, current_app_config, models_to_run):
    """
    Run all selected forecasting models and store results.

    Args:
        df_processed: Processed DataFrame
        current_app_config: Dictionary of app configuration
        models_to_run: List of models to run

    Returns:
        None (stores results in session state)
    """
    st.info(f"Starting forecasting process for models: {', '.join(models_to_run)}...")

    metrics_list = []
    forecast_dfs_list = []
    residual_plots_data = {}
    lstm_histories = {}
    nn_histories = {}
    arima_pi_data = {}

    date_col = current_app_config['date_column']
    target_col = current_app_config['target_column']
    group_col = current_app_config['group_column']
    forecast_horizon = current_app_config['forecast_horizon']
    test_size_percentage = current_app_config['test_size_percentage'] / 100.0  # Convert to fraction

    # Configure neural network parameters
    lookback_window = current_app_config.get('lookback_window', 12)  # Default to 12 time steps
    nn_epochs = current_app_config.get('nn_epochs', 50)
    nn_batch_size = current_app_config.get('nn_batch_size', 32)

    # Determine groups for iteration
    groups = [None]  # For non-grouped data (overall)
    if group_col and group_col in df_processed.columns:
        groups = df_processed[group_col].unique()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, group_id in enumerate(groups):
        group_id_str = group_id if group_id is not None else 'Overall'
        status_text.text(f"Processing group {i+1}/{len(groups)}: {group_id_str}")
        progress_bar.progress((i + 1) / len(groups))

        # Filter data for current group
        group_df = df_processed.copy()
        if group_id is not None:
            group_df = df_processed[df_processed[group_col] == group_id].copy()

        group_metrics = {'group_id': group_id_str}

        # Sort by date and set index for time series analysis
        group_df = group_df.sort_values(by=date_col).set_index(date_col)
        series = group_df[target_col]

        # Check if we have enough data
        min_required_points = forecast_horizon + lookback_window + 1
        if len(series) < min_required_points:
            st.warning(f"Skipping group '{group_id_str}': insufficient data ({len(series)} points, need {min_required_points})")
            metrics_list.append(group_metrics)
            continue

        # Calculate test size in actual points
        test_size_points = max(int(len(series) * test_size_percentage), forecast_horizon)
        test_size_points = min(test_size_points, len(series) - lookback_window - 1)

        # Split into train and test sets
        train_series = series[:-test_size_points]
        test_series = series[-test_size_points:]

        # Create forecast DataFrame
        group_forecast_df = pd.DataFrame(index=test_series.index)
        group_forecast_df['Actual'] = test_series

        # Add group column if applicable
        if group_col:
            group_forecast_df[group_col] = group_id

        # ======================= ARIMA FORECASTING =======================
        if "ARIMA" in models_to_run:
            try:
                status_text.text(f"Running ARIMA for group: {group_id_str}")

                # Auto ARIMA to find optimal parameters
                auto_model = auto_arima(
                    train_series,
                    seasonal=True,
                    m=12,  # Monthly seasonality by default
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False,
                    max_order=10,
                    max_d=2,
                    max_p=5,
                    max_q=5
                )

                # Get the best order and seasonal order
                order = auto_model.order
                seasonal_order = auto_model.seasonal_order

                # Fit ARIMA model with the best parameters
                arima_model = ARIMA(
                    train_series,
                    order=order,
                    seasonal_order=seasonal_order
                )
                arima_results = arima_model.fit()

                # Generate forecasts
                arima_forecast = arima_results.forecast(steps=len(test_series))
                group_forecast_df['ARIMA_Forecast'] = arima_forecast

                # Calculate prediction intervals
                pred_interval = arima_results.get_forecast(steps=len(test_series)).conf_int()
                lower_bound = pred_interval.iloc[:, 0]
                upper_bound = pred_interval.iloc[:, 1]

                # Store prediction intervals
                group_forecast_df['ARIMA_Lower'] = lower_bound.values
                group_forecast_df['ARIMA_Upper'] = upper_bound.values

                # Store for later use
                arima_pi_data[group_id_str] = {
                    'lower': lower_bound.values,
                    'upper': upper_bound.values
                }

                # Calculate metrics
                arima_metrics = evaluate_forecast(test_series.values, arima_forecast)
                for metric_name, metric_value in arima_metrics.items():
                    group_metrics[f'ARIMA_{metric_name.upper()}'] = metric_value

                # Store residuals for plotting
                residual_plots_data[f"{group_id_str}_ARIMA"] = {
                    'residuals': test_series.values - arima_forecast,
                    'fitted': arima_forecast
                }

            except Exception as e:
                st.warning(f"ARIMA failed for group {group_id_str}: {str(e)}")
                group_metrics['ARIMA_RMSE'] = np.nan
                group_metrics['ARIMA_MAE'] = np.nan

        # ======================= DATA PREPARATION FOR NN/LSTM =======================
        if "LSTM" in models_to_run or "Simple NN" in models_to_run:
            try:
                # Scale the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                series_values = series.values.reshape(-1, 1)
                scaled_data = scaler.fit_transform(series_values)

                # Split scaled data
                train_data = scaled_data[:-test_size_points]
                test_data = scaled_data[-test_size_points:]

                # Create sequences
                X_train, y_train = create_sequences(train_data, lookback_window, 1)
                X_test, y_test = create_sequences(test_data, lookback_window, 1)

                # Ensure we have enough sequences
                if len(X_train) == 0 or len(X_test) == 0:
                    raise ValueError(f"Not enough data to create sequences with lookback {lookback_window}")

            except Exception as e:
                st.error(f"NN/LSTM data preparation failed for group {group_id_str}: {str(e)}")
                X_train = np.array([])
                y_train = np.array([])
                X_test = np.array([])
                y_test = np.array([])

        # ======================= LSTM FORECASTING =======================
        if "LSTM" in models_to_run and len(X_train) > 0:
            try:
                status_text.text(f"Running LSTM for group: {group_id_str}")

                # Create LSTM model
                lstm_model = create_lstm_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    lstm_units=64,
                    dropout_rate=0.2,
                    bidirectional=True
                )

                # Callbacks for training
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
                ]

                # Train model
                lstm_history = lstm_model.fit(
                    X_train, y_train,
                    epochs=nn_epochs,
                    batch_size=nn_batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )

                # Store training history
                lstm_histories[group_id_str] = lstm_history

                # Generate rolling forecast
                initial_sequence = scaled_data[-test_size_points-lookback_window:-test_size_points]

                lstm_forecast = []
                last_sequence = initial_sequence.copy()

                for step in range(len(test_series)):
                    # Reshape sequence for prediction
                    current_seq = last_sequence.reshape(1, lookback_window, 1)
                    # Predict next value
                    next_pred = lstm_model.predict(current_seq, verbose=0)[0, 0]
                    # Add to forecast
                    lstm_forecast.append(next_pred)
                    # Update sequence for next prediction
                    last_sequence = np.append(last_sequence[1:], next_pred).reshape(-1, 1)

                # Inverse scale forecasts
                lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()

                # Add to forecast DataFrame
                group_forecast_df['LSTM_Forecast'] = lstm_forecast

                # Calculate metrics
                lstm_metrics = evaluate_forecast(test_series.values, lstm_forecast)
                for metric_name, metric_value in lstm_metrics.items():
                    group_metrics[f'LSTM_{metric_name.upper()}'] = metric_value

                # Store residuals for plotting
                residual_plots_data[f"{group_id_str}_LSTM"] = {
                    'residuals': test_series.values - lstm_forecast,
                    'fitted': lstm_forecast
                }

            except Exception as e:
                st.warning(f"LSTM failed for group {group_id_str}: {str(e)}")
                group_metrics['LSTM_RMSE'] = np.nan
                group_metrics['LSTM_MAE'] = np.nan

        # ======================= SIMPLE NN FORECASTING =======================
        if "Simple NN" in models_to_run and len(X_train) > 0:
            try:
                status_text.text(f"Running Simple NN for group: {group_id_str}")

                # Reshape data for simple NN (flatten sequences)
                X_train_nn = X_train.reshape(X_train.shape[0], -1)
                X_test_nn = X_test.reshape(X_test.shape[0], -1)

                # Create Simple NN model
                nn_model = create_nn_model(
                    input_shape=(X_train_nn.shape[1],),
                    units=64,
                    dropout_rate=0.2
                )

                # Callbacks for training
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
                ]

                # Train model
                nn_history = nn_model.fit(
                    X_train_nn, y_train,
                    epochs=nn_epochs,
                    batch_size=nn_batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )

                # Store training history
                nn_histories[group_id_str] = nn_history

                # Generate rolling forecast
                initial_sequence = scaled_data[-test_size_points-lookback_window:-test_size_points]

                nn_forecast = []
                last_sequence = initial_sequence.copy()

                for step in range(len(test_series)):
                    # Flatten sequence for prediction
                    current_seq = last_sequence.reshape(1, -1)
                    # Predict next value
                    next_pred = nn_model.predict(current_seq, verbose=0)[0, 0]
                    # Add to forecast
                    nn_forecast.append(next_pred)
                    # Update sequence for next prediction
                    last_sequence = np.append(last_sequence[1:], next_pred).reshape(-1, 1)

                # Inverse scale forecasts
                nn_forecast = scaler.inverse_transform(np.array(nn_forecast).reshape(-1, 1)).flatten()

                # Add to forecast DataFrame
                group_forecast_df['NN_Forecast'] = nn_forecast

                # Calculate metrics
                nn_metrics = evaluate_forecast(test_series.values, nn_forecast)
                for metric_name, metric_value in nn_metrics.items():
                    group_metrics[f'NN_{metric_name.upper()}'] = metric_value

                # Store residuals for plotting
                residual_plots_data[f"{group_id_str}_NN"] = {
                    'residuals': test_series.values - nn_forecast,
                    'fitted': nn_forecast
                }

            except Exception as e:
                st.warning(f"Simple NN failed for group {group_id_str}: {str(e)}")
                group_metrics['NN_RMSE'] = np.nan
                group_metrics['NN_MAE'] = np.nan

        # Append results to lists
        metrics_list.append(group_metrics)
        forecast_dfs_list.append(group_forecast_df.reset_index())

    # Combine results
    evaluation_df = pd.DataFrame(metrics_list)
    consolidated_forecast_df = pd.concat(forecast_dfs_list) if forecast_dfs_list else pd.DataFrame()

    # Store in session state
    st.session_state.evaluation_df = evaluation_df
    st.session_state.consolidated_forecast_df = consolidated_forecast_df
    st.session_state.all_residual_plots_data = residual_plots_data
    st.session_state.all_lstm_history_data = lstm_histories
    st.session_state.all_nn_history_data = nn_histories
    st.session_state.all_arima_pi_data = arima_pi_data
    st.session_state.forecast_results_ready = True

    # Clear progress indicators
    status_text.empty()
    progress_bar.empty()

    st.success("Forecasting completed successfully!")


def enhanced_display_model_performance(group_id, consolidated_df, evaluation_summary_df,
                                       residual_plots_data, lstm_history_data, nn_history_data,
                                       arima_pi_data, group_column_name, date_column_name,
                                       target_column_name, plot_df_eda_source):
    """
    Enhanced function to display model performance charts and metrics.

    Args:
        group_id: ID of the group to display
        consolidated_df: DataFrame with all forecasts
        evaluation_summary_df: DataFrame with evaluation metrics
        residual_plots_data: Dict with residual plotting data
        lstm_history_data: Dict with LSTM training history
        nn_history_data: Dict with NN training history
        arima_pi_data: Dict with ARIMA prediction intervals
        group_column_name: Name of group column
        date_column_name: Name of date column
        target_column_name: Name of target column
        plot_df_eda_source: The original processed DataFrame for EDA, especially for feature analysis.
    """
    st.write(f"### Performance for Group: {group_id}")

    # Filter data for current group
    plot_df_group = consolidated_df
    if group_column_name and group_column_name in consolidated_df.columns:
        plot_df_group = consolidated_df[consolidated_df[group_column_name] == group_id]

    if plot_df_group.empty:
        st.info(f"No forecast data available for group '{group_id}'")
        return

    # Get the series for the current group from the EDA source for diagnostics
    series_for_diagnostics = plot_df_eda_source.copy()
    if group_column_name and group_column_name in plot_df_eda_source.columns:
        series_for_diagnostics = plot_df_eda_source[plot_df_eda_source[group_column_name] == group_id].copy()
    series_for_diagnostics = series_for_diagnostics.sort_values(by=date_column_name).set_index(date_column_name)[target_column_name]

    if series_for_diagnostics.empty:
        st.warning(f"No data available for diagnostics for group '{group_id}'")
        return

    # Display evaluation metrics
    if evaluation_summary_df is not None:
        if group_column_name:
            group_metrics = evaluation_summary_df[evaluation_summary_df['group_id'] == group_id]
        else:
            group_metrics = evaluation_summary_df[evaluation_summary_df['group_id'] == 'Overall']

        if not group_metrics.empty:
            st.write("#### Evaluation Metrics:")
            # Drop group_id column for cleaner display
            metrics_display = group_metrics.drop(columns=['group_id'], errors='ignore')
            st.dataframe(metrics_display)

            # Create a bar chart of metrics
            metrics_to_plot = {}
            model_names = []

            for col in metrics_display.columns:
                if col.endswith('_RMSE'):
                    model_name = col.replace('_RMSE', '')
                    if not pd.isna(metrics_display[col].values[0]):
                        model_names.append(model_name)
                        metrics_to_plot[model_name] = {
                            'RMSE': metrics_display[col].values[0],
                            'MAE': metrics_display[f'{model_name}_MAE'].values[0] if f'{model_name}_MAE' in metrics_display.columns else None
                        }

            if model_names:
                # Create metrics comparison bar chart
                fig_metrics = plt.figure(figsize=(10, 6))
                x = np.arange(len(model_names))
                width = 0.35

                rmse_values = [metrics_to_plot[model]['RMSE'] for model in model_names]
                mae_values = [metrics_to_plot[model]['MAE'] for model in model_names]

                plt.bar(x - width/2, rmse_values, width, label='RMSE')
                plt.bar(x + width/2, mae_values, width, label='MAE')

                plt.xlabel('Models')
                plt.ylabel('Error')
                plt.title(f'Model Comparison for {group_id}')
                plt.xticks(x, model_names)
                plt.legend()

                st.pyplot(fig_metrics)

    # Display forecast plot
    if date_column_name in plot_df_group.columns:
        st.write("#### Forecast Plot:")

        actual_col = 'Actual'
        forecast_cols = [col for col in plot_df_group.columns if 'Forecast' in col]

        if actual_col in plot_df_group.columns and forecast_cols:
            import plotly.graph_objects as go

            # Create Plotly figure
            fig = go.Figure()

            # Add actual values
            fig.add_trace(go.Scatter(
                x=plot_df_group[date_column_name],
                y=plot_df_group[actual_col],
                mode='lines+markers',
                name='Actual',
                line=dict(color='black', width=2)
            ))

            # Add forecasts for each model
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            for i, forecast_col in enumerate(forecast_cols):
                color = colors[i % len(colors)]
                model_name = forecast_col.replace('_Forecast', '')

                fig.add_trace(go.Scatter(
                    x=plot_df_group[date_column_name],
                    y=plot_df_group[forecast_col],
                    mode='lines+markers',
                    name=f'{model_name} Forecast',
                    line=dict(color=color)
                ))

                # Add prediction intervals for ARIMA if available
                if model_name == 'ARIMA' and group_id in arima_pi_data:
                    # Check if ARIMA_Upper and ARIMA_Lower columns exist in plot_df_group
                    if 'ARIMA_Upper' in plot_df_group.columns and 'ARIMA_Lower' in plot_df_group.columns:
                        fig.add_trace(go.Scatter(
                            x=plot_df_group[date_column_name],
                            y=plot_df_group['ARIMA_Upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))

                        fig.add_trace(go.Scatter(
                            x=plot_df_group[date_column_name],
                            y=plot_df_group['ARIMA_Lower'],
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(0, 0, 255, 0.2)',
                            fill='tonexty',
                            name='ARIMA 95% CI'
                        ))

            # Update layout
            fig.update_layout(
                title=f'Forecast vs Actuals for {group_id}',
                xaxis_title=date_column_name,
                yaxis_title=target_column_name,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

    # Display residuals and diagnostics
    st.write("#### Residual Analysis:")

    model_options = []
    for key in residual_plots_data.keys():
        if key.startswith(f"{group_id}_"):
            model_name = key.replace(f"{group_id}_", "")
            model_options.append(model_name)

    if model_options:
        selected_model = st.selectbox("Select Model for Residual Analysis:", model_options)

        if selected_model:
            residual_key = f"{group_id}_{selected_model}"
            if residual_key in residual_plots_data:
                residual_data = residual_plots_data[residual_key]
                residuals = residual_data['residuals']
                fitted = residual_data['fitted']

                # Create residual plots
                fig_residuals, axs = plt.subplots(2, 2, figsize=(14, 10))

                # Residuals vs Fitted
                axs[0, 0].scatter(fitted, residuals)
                axs[0, 0].axhline(y=0, color='r', linestyle='-')
                axs[0, 0].set_title('Residuals vs Fitted')
                axs[0, 0].set_xlabel('Fitted values')
                axs[0, 0].set_ylabel('Residuals')

                # Residuals Histogram
                axs[0, 1].hist(residuals, bins=20)
                axs[0, 1].set_title('Residuals Distribution')
                axs[0, 1].set_xlabel('Residuals')
                axs[0, 1].set_ylabel('Frequency')

                # QQ Plot
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=axs[1, 0])
                axs[1, 0].set_title('Normal Q-Q Plot')

                # Residuals Time Plot
                axs[1, 1].plot(range(len(residuals)), residuals)
                axs[1, 1].axhline(y=0, color='r', linestyle='-')
                axs[1, 1].set_title('Residuals over Time')
                axs[1, 1].set_xlabel('Time')
                axs[1, 1].set_ylabel('Residuals')

                plt.tight_layout()
                st.pyplot(fig_residuals)

                # Statistical tests
                st.write("#### Statistical Tests on Residuals:")

                # Ljung-Box test for autocorrelation
                from statsmodels.stats.diagnostic import acorr_ljungbox

                try:
                    lb_test = acorr_ljungbox(series_for_diagnostics, lags=min(10, len(series_for_diagnostics) // 5), return_df=True)
                    st.write("#### Ljung-Box Test for Autocorrelation")
                    st.write("Null hypothesis: Residuals are independently distributed (no autocorrelation)")
                    st.dataframe(lb_test)

                    # Interpret results
                    if any(lb_test['lb_pvalue'] < 0.05):
                        st.warning("There is significant autocorrelation in the time series at some lags (p-value < 0.05).")
                        st.write("This suggests temporal patterns that could be exploited by forecasting models.")
                    else:
                        st.success("No significant autocorrelation detected at the tested lags (p-value >= 0.05).")
                except Exception as e:
                    st.error(f"Error running Ljung-Box test: {e}")

                # ACF and PACF plots
                st.write("#### ACF and PACF Plots")
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    max_lags = min(40, len(series_for_diagnostics) - 1)

                    plot_acf(series_for_diagnostics, lags=max_lags, ax=ax1)
                    ax1.set_title("Autocorrelation Function (ACF)")

                    plot_pacf(series_for_diagnostics, lags=max_lags, ax=ax2)
                    ax2.set_title("Partial Autocorrelation Function (PACF)")

                    plt.tight_layout()
                    st.pyplot(fig)

                    st.write("""
                    **Interpretation Guide:**
                    - ACF shows correlation between a time series and its lagged values
                    - PACF shows direct correlation between a time series and its lagged values, controlling for values at intervening lags
                    - Significant spikes at specific lags suggest potential AR or MA orders for ARIMA models
                    """)
                except Exception as e:
                    st.error(f"Error creating ACF/PACF plots: {e}")

                # Augmented Dickey-Fuller test for stationarity
                from statsmodels.tsa.stattools import adfuller

                st.write("#### Stationarity Test (Augmented Dickey-Fuller)")
                try:
                    adf_result = adfuller(series_for_diagnostics.dropna())

                    adf_output = pd.Series({
                        'Test Statistic': adf_result[0],
                        'p-value': adf_result[1],
                        '1% Critical Value': adf_result[4]['1%'],
                        '5% Critical Value': adf_result[4]['5%'],
                        '10% Critical Value': adf_result[4]['10%']
                    })

                    st.dataframe(adf_output)

                    # Interpret results
                    if adf_result[1] <= 0.05:
                        st.success("Time series is stationary (p-value <= 0.05). This is good for forecasting!")
                    else:
                        st.warning("Time series is likely non-stationary (p-value > 0.05). Differencing may help.")

                        # Offer differencing option
                        if st.checkbox("Apply differencing to make series stationary"):
                            diff_order = st.slider("Differencing order:", 1, 2, 1)
                            diff_series = series_for_diagnostics.diff(diff_order).dropna()

                            # Plot differenced series
                            fig_diff = plt.figure(figsize=(10, 6))
                            plt.plot(diff_series)
                            plt.title(f"Series after {diff_order}-order differencing")
                            plt.xlabel("Time"); plt.ylabel("Differenced Value")
                            st.pyplot(fig_diff)

                            # Re-run ADF test on differenced series
                            adf_diff_result = adfuller(diff_series.dropna())
                            adf_diff_output = pd.Series({
                                'Test Statistic': adf_diff_result[0],
                                'p-value': adf_diff_result[1],
                                '1% Critical Value': adf_diff_result[4]['1%'],
                                '5% Critical Value': adf_diff_result[4]['5%'],
                                '10% Critical Value': adf_diff_result[4]['10%']
                            })

                            st.write("ADF Test on Differenced Series:")
                            st.dataframe(adf_diff_output)

                            if adf_diff_result[1] <= 0.05:
                                st.success("Differenced series is now stationary (p-value <= 0.05).")
                                st.info(f"Suggestion: Consider using ARIMA with d={diff_order} for this time series.")
                            else:
                                st.warning("Series remains non-stationary even after differencing.")

                except Exception as e:
                    st.error(f"Error running stationarity test: {e}")

                # Distribution analysis
                st.write("#### Distribution Analysis")
                try:
                    fig_dist, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12, 5))

                    # Histogram with KDE
                    sns.histplot(series_for_diagnostics, kde=True, ax=ax_hist)
                    ax_hist.set_title("Distribution of Target Variable")
                    ax_hist.set_xlabel(target_column_name)
                    ax_hist.set_ylabel("Frequency")

                    # QQ plot
                    stats.probplot(series_for_diagnostics.dropna(), plot=ax_qq)
                    ax_qq.set_title("Q-Q Plot")

                    plt.tight_layout()
                    st.pyplot(fig_dist)

                    # Skewness and kurtosis
                    skewness = series_for_diagnostics.skew()
                    kurtosis = series_for_diagnostics.kurtosis()

                    st.write(f"**Skewness**: {skewness:.4f} ({'Positively' if skewness > 0 else 'Negatively'} skewed)")
                    st.write(f"**Kurtosis**: {kurtosis:.4f} ({'Heavy' if kurtosis > 0 else 'Light'} tailed compared to normal distribution)")

                    # Normality test
                    shapiro_test = stats.shapiro(series_for_diagnostics.sample(min(5000, len(series_for_diagnostics))) if len(series_for_diagnostics) > 5000 else series_for_diagnostics)
                    st.write(f"**Shapiro-Wilk Test p-value**: {shapiro_test[1]:.6f}")
                    if shapiro_test[1] < 0.05:
                        st.warning("The data is not normally distributed (p-value < 0.05)")

                        # Box-Cox transformation suggestion
                        if st.checkbox("Apply Box-Cox transformation") and (series_for_diagnostics > 0).all():
                            from scipy import stats
                            transformed_data, lambda_value = stats.boxcox(series_for_diagnostics)

                            # Plot transformed series
                            fig_transform = plt.figure(figsize=(10, 6))
                            plt.hist(transformed_data, bins=30, alpha=0.7, density=True)
                            plt.title(f"Box-Cox Transformed Data (λ = {lambda_value:.4f})")
                            plt.xlabel("Transformed Value")
                            st.pyplot(fig_transform)

                            # Re-test normality
                            shapiro_transform = stats.shapiro(transformed_data[:5000] if len(transformed_data) > 5000 else transformed_data)
                            st.write(f"**Shapiro-Wilk Test p-value after transformation**: {shapiro_transform[1]:.6f}")
                            if shapiro_transform[1] < 0.05:
                                st.warning("Data remains non-normal even after transformation.")
                            else:
                                st.success("Box-Cox transformation successfully normalized the data.")
                                st.info("Consider using this transformation before modeling for better results.")
                    else:
                        st.success("The data follows a normal distribution (p-value >= 0.05)")
                except Exception as e:
                    st.error(f"Error in distribution analysis: {e}")

                # Outlier detection
                st.write("#### Outlier Detection")
                try:
                    q1 = series_for_diagnostics.quantile(0.25)
                    q3 = series_for_diagnostics.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = series_for_diagnostics[(series_for_diagnostics < lower_bound) | (series_for_diagnostics > upper_bound)]
                    outlier_percentage = (len(outliers) / len(series_for_diagnostics)) * 100

                    st.write(f"**Number of outliers detected**: {len(outliers)} ({outlier_percentage:.2f}% of the data)")

                    # Box plot for outliers
                    fig_box = plt.figure(figsize=(10, 6))
                    sns.boxplot(x=series_for_diagnostics)
                    plt.title("Box Plot with Outliers")
                    plt.xlabel(target_column_name)
                    st.pyplot(fig_box)

                    if len(outliers) > 0:
                        st.write("**Sample of outliers:**")
                        st.dataframe(outliers.head(10))

                        if st.checkbox("Show outliers in time series"):
                            fig_outlier_ts = plt.figure(figsize=(12, 6))
                            plt.plot(series_for_diagnostics.index, series_for_diagnostics, 'b-', label='Original Series')
                            plt.scatter(outliers.index, outliers, color='red', label='Outliers')
                            plt.title("Time Series with Outliers Highlighted")
                            plt.xlabel("Date")
                            plt.ylabel(target_column_name)
                            plt.legend()
                            st.pyplot(fig_outlier_ts)

                        # Option to handle outliers
                        if st.checkbox("Handle outliers"):
                            outlier_method = st.radio(
                                "Select outlier handling method:",
                                ("Cap at bounds", "Remove outliers", "Replace with median")
                            )

                            if outlier_method == "Cap at bounds":
                                clean_series = series_for_diagnostics.copy()
                                clean_series[clean_series < lower_bound] = lower_bound
                                clean_series[clean_series > upper_bound] = upper_bound
                                st.info("Outliers capped at IQR boundaries")
                            elif outlier_method == "Remove outliers":
                                clean_series = series_for_diagnostics[(series_for_diagnostics >= lower_bound) & (series_for_diagnostics <= upper_bound)]
                                st.info(f"Removed {len(series_for_diagnostics) - len(clean_series)} outliers")
                            else:  # Replace with median
                                clean_series = series_for_diagnostics.copy()
                                clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)] = series_for_diagnostics.median()
                                st.info("Outliers replaced with median value")

                            # Plot cleaned series
                            fig_clean = plt.figure(figsize=(12, 6))
                            plt.plot(series_for_diagnostics.index, series_for_diagnostics, 'b-', alpha=0.5, label='Original Series')
                            plt.plot(clean_series.index, clean_series, 'g-', label='Cleaned Series')
                            plt.title("Original vs Cleaned Time Series")
                            plt.xlabel("Date")
                            plt.ylabel(target_column_name)
                            plt.legend()
                            st.pyplot(fig_clean)
                    else:
                        st.success("No outliers detected.")
                except Exception as e:
                    st.error(f"Error in outlier detection: {e}")

            # Feature importance analysis for additional features if available
            if st.checkbox("Analyze feature importance"):
                feature_cols = [col for col in plot_df_eda_source.columns if col not in [date_column_name, target_column_name, group_column_name]
                               and pd.api.types.is_numeric_dtype(plot_df_eda_source[col])]

                if feature_cols:
                    st.write("#### Feature Importance Analysis")

                    # Correlation matrix
                    st.write("**Correlation with Target Variable:**")
                    corr_with_target = plot_df_eda_source[[target_column_name] + feature_cols].corr()[target_column_name].drop(target_column_name).sort_values(ascending=False)

                    # Bar chart of correlations
                    fig_corr = px.bar(
                        x=corr_with_target.index,
                        y=corr_with_target.values,
                        labels={'x': 'Feature', 'y': f'Correlation with {target_column_name}'},
                        title=f"Feature Correlations with {target_column_name}"
                    )
                    st.plotly_chart(fig_corr)

                    # Full correlation heatmap
                    st.write("**Full Correlation Matrix:**")
                    corr_matrix = plot_df_eda_source[[target_column_name] + feature_cols].corr()
                    fig_heatmap = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Heatmap"
                    )
                    st.plotly_chart(fig_heatmap)

                    # Feature scatter plots
                    st.write("**Feature Scatter Plots:**")
                    selected_feature = st.selectbox("Select feature to plot against target:", feature_cols)

                    fig_scatter = px.scatter(
                        plot_df_eda_source,
                        x=selected_feature,
                        y=target_column_name,
                        trendline="ols",
                        title=f"{selected_feature} vs {target_column_name}"
                    )
                    st.plotly_chart(fig_scatter)

                    # Random Forest feature importance
                    if st.checkbox("Run Random Forest feature importance analysis"):
                        from sklearn.ensemble import RandomForestRegressor
                        from sklearn.model_selection import train_test_split

                        try:
                            X = plot_df_eda_source[feature_cols].fillna(plot_df_eda_source[feature_cols].mean())
                            y = plot_df_eda_source[target_column_name]

                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(X, y)

                            importance_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': rf.feature_importances_
                            }).sort_values('Importance', ascending=False)

                            fig_imp = px.bar(
                                importance_df,
                                x='Feature',
                                y='Importance',
                                title="Random Forest Feature Importance"
                            )
                            st.plotly_chart(fig_imp)

                        except Exception as e:
                            st.error(f"Error in Random Forest feature importance: {e}")
                else:
                    st.info("No additional numeric features available for importance analysis.")

            # Add calendar visualization for seasonality analysis
            if st.checkbox("Visualize calendar patterns"):
                st.write("#### Calendar Pattern Analysis")

                # Ensure date_column_name is datetime
                plot_df_eda_source[date_column_name] = pd.to_datetime(plot_df_eda_source[date_column_name])

                # Extract date components
                date_features = pd.DataFrame({
                    'date': plot_df_eda_source[date_column_name],
                    'value': plot_df_eda_source[target_column_name],
                    'year': plot_df_eda_source[date_column_name].dt.year,
                    'month': plot_df_eda_source[date_column_name].dt.month,
                    'day': plot_df_eda_source[date_column_name].dt.day,
                    'dayofweek': plot_df_eda_source[date_column_name].dt.dayofweek,
                    'quarter': plot_df_eda_source[date_column_name].dt.quarter
                })

                # Select pattern to visualize
                pattern_type = st.radio(
                    "Select pattern to analyze:",
                    ("Monthly", "Day of Week", "Quarterly", "Yearly")
                )

                if pattern_type == "Monthly":
                    monthly_avg = date_features.groupby('month')['value'].mean().reset_index()
                    monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: pd.Timestamp(2020, x, 1).strftime('%b'))

                    fig_monthly = px.bar(
                        monthly_avg,
                        x='month_name',
                        y='value',
                        title="Average Target Value by Month",
                        labels={'value': target_column_name, 'month_name': 'Month'}
                    )
                    st.plotly_chart(fig_monthly)

                elif pattern_type == "Day of Week":
                    dow_avg = date_features.groupby('dayofweek')['value'].mean().reset_index()
                    dow_avg['day_name'] = dow_avg['dayofweek'].apply(lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])

                    fig_dow = px.bar(
                        dow_avg,
                        x='day_name',
                        y='value',
                        title="Average Target Value by Day of Week",
                        labels={'value': target_column_name, 'day_name': 'Day of Week'}
                    )
                    st.plotly_chart(fig_dow)

                elif pattern_type == "Quarterly":
                    q_avg = date_features.groupby('quarter')['value'].mean().reset_index()
                    q_avg['quarter_name'] = 'Q' + q_avg['quarter'].astype(str)

                    fig_q = px.bar(
                        q_avg,
                        x='quarter_name',
                        y='value',
                        title="Average Target Value by Quarter",
                        labels={'value': target_column_name, 'quarter_name': 'Quarter'}
                    )
                    st.plotly_chart(fig_q)

                elif pattern_type == "Yearly":
                    if date_features['year'].nunique() > 1:
                        y_avg = date_features.groupby('year')['value'].mean().reset_index()

                        fig_y = px.bar(
                            y_avg,
                            x='year',
                            y='value',
                            title="Average Target Value by Year",
                            labels={'value': target_column_name, 'year': 'Year'}
                        )
                        st.plotly_chart(fig_y)

                        # Year-over-year comparison
                        if st.checkbox("Show year-over-year comparison"):
                            yoy_data = date_features.pivot_table(
                                index=['month', 'day'],
                                columns='year',
                                values='value',
                                aggfunc='mean'
                            ).reset_index()

                            yoy_melted = yoy_data.melt(
                                id_vars=['month', 'day'],
                                var_name='year',
                                value_name='value'
                            )
                            yoy_melted['date'] = yoy_melted.apply(
                                lambda x: pd.Timestamp(2000, x['month'], x['day']), axis=1
                            )

                            fig_yoy = px.line(
                                yoy_melted.sort_values('date'),
                                x='date',
                                y='value',
                                color='year',
                                title="Year-over-Year Comparison",
                                labels={'value': target_column_name, 'date': 'Month and Day'}
                            )
                            fig_yoy.update_xaxes(
                                tickformat="%b %d",
                                tickmode='array',
                                tickvals=[pd.Timestamp(2000, m, 1) for m in range(1, 13)]
                            )
                            st.plotly_chart(fig_yoy)
                    else:
                        st.info("Not enough years in the data for yearly comparison.")

            # Add holiday impact analysis
            if st.checkbox("Analyze holiday impact"):
                st.write("#### Holiday Impact Analysis")

                country_code = st.selectbox(
                    "Select country for holidays:",
                    ["US", "UK", "CA", "AU", "DE", "FR", "JP", "CN", "BR", "IN"]
                )

                # Ensure date_column_name is datetime
                plot_df_eda_source[date_column_name] = pd.to_datetime(plot_df_eda_source[date_column_name])

                year_range = plot_df_eda_source[date_column_name].dt.year.unique()
                if len(year_range) > 0:
                    country_holidays = holidays.country_holidays(
                        country_code,
                        years=list(year_range)
                    )

                    # Find holiday dates in the data
                    plot_df_eda_holiday = plot_df_eda_source.copy().reset_index()
                    plot_df_eda_holiday['is_holiday'] = plot_df_eda_holiday[date_column_name].dt.date.apply(
                        lambda x: x in country_holidays
                    )

                    if plot_df_eda_holiday['is_holiday'].any():
                        # Add holiday names
                        plot_df_eda_holiday['holiday_name'] = plot_df_eda_holiday[date_column_name].dt.date.apply(
                            lambda x: country_holidays.get(x, "")
                        )

                        # Compare holiday vs non-holiday
                        holiday_avg = plot_df_eda_holiday.groupby('is_holiday')[target_column_name].mean()

                        fig_hol = px.bar(
                            x=['Non-Holiday', 'Holiday'],
                            y=holiday_avg.values,
                            title="Average Target Value: Holiday vs Non-Holiday",
                            labels={'x': 'Day Type', 'y': target_column_name}
                        )
                        st.plotly_chart(fig_hol)

                        # Table of holidays and their values
                        holiday_df = plot_df_eda_holiday[plot_df_eda_holiday['is_holiday']].copy()
                        if not holiday_df.empty:
                            st.write("**Holidays in the data:**")
                            holiday_summary = holiday_df.groupby('holiday_name').agg({
                                target_column_name: ['mean', 'count']
                            }).reset_index()
                            holiday_summary.columns = ['Holiday', 'Average Value', 'Count']
                            st.dataframe(holiday_summary)

                            # Plot specific holidays
                            st.write("**Impact of specific holidays:**")
                            top_holidays = holiday_df['holiday_name'].value_counts().nlargest(10).index.tolist()
                            selected_holiday = st.selectbox("Select holiday:", top_holidays)

                            if selected_holiday:
                                holiday_dates = holiday_df[holiday_df['holiday_name'] == selected_holiday][date_column_name].dt.date

                                # Find days around the holiday
                                holiday_impact = pd.DataFrame()
                                for hdate in holiday_dates:
                                    # Get 3 days before and after
                                    start_date = pd.Timestamp(hdate) - pd.Timedelta(days=3)
                                    end_date = pd.Timestamp(hdate) + pd.Timedelta(days=3)
                                    date_range = pd.date_range(start_date, end_date)

                                    # Find data for these dates
                                    period_data = plot_df_eda_source[
                                        (plot_df_eda_source[date_column_name] >= start_date) &
                                        (plot_df_eda_source[date_column_name] <= end_date)
                                    ].copy()

                                    if not period_data.empty:
                                        period_data['days_from_holiday'] = (period_data[date_column_name] - pd.Timestamp(hdate)).dt.days
                                        period_data['holiday_date'] = hdate
                                        holiday_impact = pd.concat([holiday_impact, period_data])

                                if not holiday_impact.empty:
                                    impact_summary = holiday_impact.groupby('days_from_holiday')[target_column_name].mean().reset_index()
                                    impact_summary['day_label'] = impact_summary['days_from_holiday'].apply(
                                        lambda x: f"H{x:+d}" if x != 0 else "Holiday"
                                    )

                                    fig_impact = px.bar(
                                        impact_summary.sort_values('days_from_holiday'),
                                        x='day_label',
                                        y=target_column_name,
                                        title=f"Impact of {selected_holiday} (Days Before and After)",
                                        labels={'day_label': 'Day', target_column_name: target_column_name}
                                    )
                                    st.plotly_chart(fig_impact)
                    else:
                        st.info(f"No holidays from {country_code} found in the data range.")
                else:
                    st.error("Cannot determine year range from the data.")


# ======================================================================
# Main Streamlit Application Logic
# ======================================================================

st.set_page_config(layout="wide", page_title="Advanced Demand Forecasting App")

st.title("📈 Advanced Demand Forecasting Application")
st.markdown("Upload your time series data, configure forecasting parameters, and evaluate multiple models.")

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'forecast_results_ready' not in st.session_state:
    st.session_state.forecast_results_ready = False
if 'evaluation_df' not in st.session_state:
    st.session_state.evaluation_df = None
if 'consolidated_forecast_df' not in st.session_state:
    st.session_state.consolidated_forecast_df = None
if 'all_residual_plots_data' not in st.session_state:
    st.session_state.all_residual_plots_data = None
if 'all_lstm_history_data' not in st.session_state:
    st.session_state.all_lstm_history_data = None
if 'all_nn_history_data' not in st.session_state:
    st.session_state.all_nn_history_data = None
if 'all_arima_pi_data' not in st.session_state:
    st.session_state.all_arima_pi_data = None

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.forecast_results_ready = False # Reset results if new file uploaded

        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else: # .xlsx
                df = pd.read_excel(uploaded_file)

            st.session_state.df_original = df.copy()
            st.session_state.df_processed = df.copy() # df_processed will be used for actual processing
            st.success("File uploaded successfully!")
            st.write("First 5 rows of your data:")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.session_state.df_original = None
            st.session_state.df_processed = None
            st.session_state.uploaded_file = None # Clear uploaded file to avoid re-processing on rerun
else:
    st.session_state.df_original = None
    st.session_state.df_processed = None
    st.session_state.uploaded_file = None
    st.session_state.forecast_results_ready = False


if st.session_state.df_original is not None:
    df = st.session_state.df_original.copy()
    available_columns = df.columns.tolist()

    st.sidebar.header("Configuration")

    # Column selection
    date_column = st.sidebar.selectbox("Select Date Column", available_columns)
    target_column = st.sidebar.selectbox("Select Target Column", available_columns)

    # Optional group column
    group_column_options = [None] + available_columns
    group_column = st.sidebar.selectbox("Select Group Column (Optional)", group_column_options, format_func=lambda x: x if x is not None else "None")

    # Forecasting parameters
    forecast_horizon = st.sidebar.number_input("Forecast Horizon (steps)", min_value=1, value=12)
    test_size_percentage = st.sidebar.slider("Test Data Size (% of total data)", min_value=10, max_value=50, value=20)
    lookback_window = st.sidebar.number_input("Lookback Window (for NN/LSTM)", min_value=1, value=12)

    # Model selection
    st.sidebar.subheader("Select Models to Run")
    models_to_run = []
    if st.sidebar.checkbox("ARIMA", value=True):
        models_to_run.append("ARIMA")
    if st.sidebar.checkbox("LSTM", value=True):
        models_to_run.append("LSTM")
    if st.sidebar.checkbox("Simple NN", value=False):
        models_to_run.append("Simple NN")

    # NN/LSTM training parameters
    if "LSTM" in models_to_run or "Simple NN" in models_to_run:
        st.sidebar.subheader("Neural Network Parameters")
        nn_epochs = st.sidebar.number_input("Epochs", min_value=1, value=50)
        nn_batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=32)
    else:
        nn_epochs = 50 # Default if not selected
        nn_batch_size = 32 # Default if not selected

    current_app_config = {
        'date_column': date_column,
        'target_column': target_column,
        'group_column': group_column,
        'forecast_horizon': forecast_horizon,
        'test_size_percentage': test_size_percentage,
        'lookback_window': lookback_window,
        'nn_epochs': nn_epochs,
        'nn_batch_size': nn_batch_size
    }

    if st.sidebar.button("Run Forecasts"):
        if not models_to_run:
            st.warning("Please select at least one model to run.")
        else:
            with st.spinner("Processing data and running forecasts... This may take a while."):
                # Data preprocessing steps
                try:
                    # Convert date column to datetime objects
                    st.session_state.df_processed[date_column] = pd.to_datetime(st.session_state.df_processed[date_column])
                    # Handle missing values in target column (e.g., fill with median or mean)
                    if st.session_state.df_processed[target_column].isnull().any():
                        st.warning(f"Missing values found in '{target_column}'. Filling with median.")
                        st.session_state.df_processed[target_column].fillna(st.session_state.df_processed[target_column].median(), inplace=True)

                    # Ensure target column is numeric
                    st.session_state.df_processed[target_column] = pd.to_numeric(st.session_state.df_processed[target_column], errors='coerce')
                    if st.session_state.df_processed[target_column].isnull().any():
                        st.error(f"Target column '{target_column}' contains non-numeric values after conversion. Please clean your data.")
                        st.stop()

                    # Ensure sorted by date for time series analysis
                    st.session_state.df_processed = st.session_state.df_processed.sort_values(by=date_column)

                    run_all_forecasts(st.session_state.df_processed, current_app_config, models_to_run)
                except Exception as e:
                    st.error(f"Error during data processing or forecasting setup: {e}")
                    st.session_state.forecast_results_ready = False # Reset flag
            st.rerun() # Replaced st.experimental_rerun with st.rerun

    if st.session_state.forecast_results_ready:
        st.subheader("Forecasting Results")

        if st.session_state.evaluation_df is not None and not st.session_state.evaluation_df.empty:
            # Determine groups for selection
            display_groups = ['Overall']
            if group_column and group_column in st.session_state.df_processed.columns:
                display_groups.extend(st.session_state.df_processed[group_column].unique().tolist())

            selected_group = st.selectbox(
                "Select a group to view detailed performance:",
                display_groups
            )

            group_id_for_display = selected_group if selected_group != 'Overall' else None

            enhanced_display_model_performance(
                group_id=selected_group,
                consolidated_df=st.session_state.consolidated_forecast_df,
                evaluation_summary_df=st.session_state.evaluation_df,
                residual_plots_data=st.session_state.all_residual_plots_data,
                lstm_history_data=st.session_state.all_lstm_history_data,
                nn_history_data=st.session_state.all_nn_history_data,
                arima_pi_data=st.session_state.all_arima_pi_data,
                group_column_name=group_column,
                date_column_name=date_column,
                target_column_name=target_column,
                plot_df_eda_source=st.session_state.df_processed # Pass processed DF for EDA
            )
        else:
            st.warning("No forecasting results available. Please upload data and run forecasts.")

else:
    st.info("Please upload a CSV or Excel file to get started with demand forecasting.")
