# Demand Forecasting Application

This repository contains a Streamlit application for forecasting demand using various time series models. The app allows users to upload their sales data, select different forecasting models, configure training parameters, and visualize the results. It is designed to handle forecasting for multiple distinct groups (e.g., different stores, products) in parallel.

## Features

-   **Multiple Models:** Implementations of ARIMA, Moving Average (MA), Feedforward Neural Network (NN), Long Short-Term Memory (LSTM), and a Hybrid ARIMA-LSTM model.
-   **Parallel Processing:** Forecasts for different groups are processed in parallel to improve performance.
-   **Configurable Parameters:** Easily adjust test set size, LSTM look-back period, training epochs, batch size, model architecture (NN layers/units), MA order, and more via the Streamlit sidebar.
-   **Data Handling:** Reads data from CSV or Excel files. Requires columns for Date, Sales/Demand, and a Grouping identifier.
-   **Feature Engineering:** Includes options for adding simple lagged features and automatically extracts time-based features (Year, Month, Day, DayOfWeek, etc.).
-   **Checkpointing:** NN and LSTM models can attempt to resume training from previously saved checkpoints, saving time on subsequent runs.
-   **Comprehensive Visualization:** Interactive plots using Plotly showing actual vs. forecasted sales (as a bar chart for comparison), model residuals, and training loss history for neural network models.
-   **Evaluation Metrics:** Calculates and displays RMSE and MAE for each model per group, and overall averages.
-   **Results Saving:** Saves consolidated forecasts, evaluation metrics, trained models, scalers, and plots (as HTML and PNG) to a specified output directory.

## Models Implemented

-   **ARIMA (AutoRegressive Integrated Moving Average):** Uses `pmdarima` for automatic selection of optimal ARIMA orders.
-   **Moving Average (MA):** Implemented using `statsmodels.tsa.arima.model.ARIMA` with appropriate orders.
-   **Neural Network (NN):** A simple Feedforward Neural Network (MLP) trained on time series sequences with configurable layers and units.
-   **LSTM (Long Short-Term Memory):** A type of recurrent neural network (RNN) well-suited for sequence data, with an option for Bidirectional LSTM.
-   **Hybrid ARIMA-LSTM:** A hybrid approach where an LSTM model is trained on the residuals from the ARIMA model's predictions.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SJOSKA2211/forecasting.git](https://github.com/SJOSKA2211/forecasting.git)
    cd forecasting
    ```
2.  **Install Dependencies:** Make sure you have Python installed (preferably Python 3.7+). Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have an NVIDIA GPU and a CUDA setup, you might need to install `tensorflow-gpu` instead of `tensorflow` for accelerated training, depending on your specific environment and TensorFlow version compatibility.*
3.  **Prepare your data:** Ensure you have a sales data file in CSV or Excel format with columns for Date, Sales/Demand, and a Grouping identifier (e.g., 'Date', 'Sales', 'StoreID').

## How to Run

1.  Open your terminal or command prompt.
2.  Navigate to the repository directory if you are not already there:
    ```bash
    cd /path/to/your/forecasting/repo
    ```
3.  Run the Streamlit application:
    ```bash
    streamlit run demand_forecast_app.py
    ```
4.  The application will open in your web browser. Use the sidebar to configure the settings and click "Start Forecasting".

## Configuration

In the application's sidebar, you can configure the following:

-   **Data File Path:** Path to your input CSV or Excel file.
-   **Column Names:** Exact names for your Date, Sales, and Grouping columns.
-   **Time Series Frequency:** Specify the frequency (e.g., 'D', 'W', 'M') for ARIMA/MA seasonality. Leave empty for auto-detection.
-   **Test Size:** Number of days to reserve for the test set.
