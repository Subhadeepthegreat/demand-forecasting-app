import streamlit as st
import pandas as pd
import numpy as np
import time
import asyncio
import plotly.express as px

from darts.models import NHiTSModel
from darts import TimeSeries
import plotly.graph_objects as go

# Ensure an event loop is available (to avoid "no running event loop" errors)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set page configuration for dark mode and wide layout
st.set_page_config(
    page_title="Demand Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for dark mode and adjust sidebar styling
st.markdown(
    """
    <style>
    .reportview-container, .main {
        background-color: #2e2e2e;
        color: #f0f0f0;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
    }
    .streamlit-expanderHeader {
        color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Demand Forecasting")
st.markdown("""
            
This app lets you upload your Demand and External Variables Excel files, choose which external variables to include, 
and run a forecasting model.

""")

# Sidebar: File uploaders and selection options
st.sidebar.header("Input Files & Options")

demand_file = st.sidebar.file_uploader("Upload Demand Data (Excel)", type=["xlsx"])
ext_var_file = st.sidebar.file_uploader("Upload External Variables (Excel)", type=["xlsx"])

# To hold processed results to avoid heavy recomputation on every widget change
if "sku_graph_data" not in st.session_state:
    st.session_state["sku_graph_data"] = None
if "final_output_df" not in st.session_state:
    st.session_state["final_output_df"] = None
if "average_mape" not in st.session_state:
    st.session_state["average_mape"] = None

if demand_file is not None and ext_var_file is not None:
    # Load Demand Data
    try:
        demand_data = pd.read_excel(demand_file)
    except Exception as e:
        st.error(f"Error loading Demand Data: {e}")
    
    # Load External Variables
    try:
        ext_var = pd.read_excel(ext_var_file)
    except Exception as e:
        st.error(f"Error loading External Variables: {e}")
    
    # Preprocess Demand Data
    if 'Date' not in demand_data.columns:
        st.error("Demand Data must contain a 'Date' column.")
    else:
        demand_data['Date'] = pd.to_datetime(demand_data['Date'])
        demand_data.set_index('Date', inplace=True)
        demand_data.fillna(0, inplace=True)
    
    # Arrange previews side by side using columns
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Demand Data Preview"):
            st.dataframe(demand_data.head())
    with col2:
        with st.expander("External Variables Preview"):
            if 'Date' not in ext_var.columns:
                st.error("External Variables must contain a 'Date' column.")
            else:
                ext_var['Date'] = pd.to_datetime(ext_var['Date'])
                ext_var_cols = ext_var.columns.tolist()
                ext_var_cols.remove('Date')
                chosen_vars = st.sidebar.multiselect("Select External Variables", 
                                                      options=ext_var_cols,
                                                      default=ext_var_cols)
                ext_var = ext_var[['Date'] + chosen_vars]
                st.dataframe(ext_var.head())
    
    # Split demand data into training and testing parts
    train_data = demand_data[:48]
    test_data = demand_data[48:]
    
    # Let user select which SKU columns (demand columns) to use for forecasting
    sku_options = demand_data.columns.tolist()
    sample_skus = st.sidebar.multiselect("Select SKU Columns for Forecasting", 
                                         options=sku_options,
                                         default=sku_options[:2])
    if len(sample_skus) < 1:
        st.error("Please select at least one SKU column for forecasting.")
    else:
        train_data_sample = train_data[sample_skus]
        test_data_sample = test_data[sample_skus]
    
    st.sidebar.markdown("---")
    st.sidebar.info("Press the button below to start processing.")
    
    progress_text = st.empty()  # For status updates
    progress_bar = st.progress(0)
    
    # Dictionary to store merged prediction dataframes for each SKU for plotting later
    sku_graph_data = {}
    
    if st.button("Start Processing"):
        overall_results = []
        dates = pd.date_range(start='2019-01-01', end='2024-06-01', freq='MS')
        num_skus = len(train_data_sample.columns)
        
        # Process each SKU one by one
        for i, sku in enumerate(train_data_sample.columns, start=1):
            progress_text.text(f"Processing SKU: {sku} ({i}/{num_skus})")
            
            # Extract train and test data for the current SKU and reset index
            train_sku_data = pd.DataFrame(train_data_sample[sku]).reset_index()
            test_sku_data = pd.DataFrame(test_data_sample[sku]).reset_index()
            
            # Extend test data with 6 future months
            last_date = test_sku_data['Date'].max()
            new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
            new_data = pd.DataFrame({'Date': new_dates, sku: [None] * 6})
            test_sku_extended = pd.concat([test_sku_data, new_data], ignore_index=True)
            
            # Merge training data with external variables
            merged_train_sku_data = pd.merge(train_sku_data, ext_var, on='Date', how='left')
            merged_train_sku_data['Date'] = pd.to_datetime(merged_train_sku_data['Date'])
            
            # Create a TimeSeries object using the target SKU column
            ts = TimeSeries.from_dataframe(merged_train_sku_data, time_col='Date', value_cols=sku)
            
            # Train the NHiTS model
            progress_text.text(f"Training model for SKU: {sku}")
            model = NHiTSModel(input_chunk_length=2, output_chunk_length=6, n_epochs=100)
            model.fit(ts)
            progress_bar.progress(10 + int((i / num_skus) * 40))
            
            # Generate forecast
            progress_text.text(f"Generating forecast for SKU: {sku}")
            pred = model.predict(18)
            predictions_valid = pred.pd_dataframe().reset_index().rename(columns={"index": "Date"})
            predictions_valid['Date'] = pd.to_datetime(predictions_valid['Date'])
            # Rename the predictions column for clarity
            predictions_valid.rename(columns={sku: "Predicted"}, inplace=True)
            
            # Merge predictions with test data
            test_sku_extended['Date'] = pd.to_datetime(test_sku_extended['Date'])
            test_sku_extended.rename(columns={sku: "Actuals"}, inplace=True)
            merged_prediction_df = pd.merge(predictions_valid, test_sku_extended, on='Date', how='left')
            
            # Convert Predicted and Actuals to numeric types for consistency
            merged_prediction_df["Predicted"] = pd.to_numeric(merged_prediction_df["Predicted"], errors="coerce")
            merged_prediction_df["Actuals"] = pd.to_numeric(merged_prediction_df["Actuals"], errors="coerce")
            
            # Calculate MAPE on validation period (dates <= 2023-12-01 and non-zero predicted values)
            validation_df = merged_prediction_df[(merged_prediction_df['Date'] <= '2023-12-01') & (merged_prediction_df["Predicted"] != 0)]
            if len(validation_df) > 0:
                validation_df['APE'] = abs((validation_df["Predicted"] - validation_df['Actuals']) / validation_df["Predicted"]) * 100
                mape = validation_df['APE'].mean()
            else:
                mape = np.nan
            
            # Prepare output row with unified date range
            result_row = {'SKU': sku, 'MAPE': mape, 'Model': 'NHiTSModel'}
            for date in dates:
                if date in train_sku_data['Date'].values:
                    result_row[date] = train_sku_data.loc[train_sku_data['Date'] == date, sku].values[0]
                elif date in test_sku_data['Date'].values:
                    result_row[date] = test_sku_data.loc[test_sku_data['Date'] == date, sku].values[0]
                elif date in predictions_valid['Date'].values:
                    result_row[date] = predictions_valid.loc[predictions_valid['Date'] == date, "Predicted"].values[0]
                else:
                    result_row[date] = None
            overall_results.append(result_row)
            
            # Save merged prediction data for plotting
            sku_graph_data[sku] = merged_prediction_df.copy()
            
            progress_bar.progress(50 + int((i / num_skus) * 40))
            time.sleep(0.5)
        
        # Create final output DataFrame and calculate average MAPE
        final_output_df = pd.DataFrame(overall_results)
        final_output_df.set_index('SKU', inplace=True)
        average_mape = final_output_df['MAPE'].mean()
        
        progress_text.text("Finalizing output...")
        progress_bar.progress(100)
        time.sleep(1)
        st.success("Processing complete!")
        
        # Save the processed results in session_state for later use
        st.session_state["sku_graph_data"] = sku_graph_data
        st.session_state["final_output_df"] = final_output_df
        st.session_state["average_mape"] = average_mape

    # If processing has been completed, show final output and interactive graph selection
    if st.session_state["final_output_df"] is not None:
        with st.expander("Final Output DataFrame"):
            st.dataframe(st.session_state["final_output_df"])
            st.write(f"**Average MAPE:** {st.session_state['average_mape']}")
        
        st.subheader("Interactive Forecast vs Actual Graphs")
        # Use a form to prevent immediate re-run when selecting a SKU
        with st.form("sku_graph_form"):
            sku_choice = st.selectbox("Select SKU to view graph", options=list(st.session_state["sku_graph_data"].keys()))
            submit_graph = st.form_submit_button("View Graph")
        if submit_graph:
            df_plot = st.session_state["sku_graph_data"][sku_choice].copy()
            # Ensure numeric conversion for plotting
            df_plot["Predicted"] = pd.to_numeric(df_plot["Predicted"], errors="coerce")
            df_plot["Actuals"] = pd.to_numeric(df_plot["Actuals"], errors="coerce")

            

            # Define the actual series (all rows except the last 6)
            actual_dates = df_plot["Date"].iloc[:-6]
            actual_values = df_plot["Actuals"].iloc[:-6]

            # Get the last actual point to join with the forecast
            last_actual_date = actual_dates.iloc[-1]
            last_actual_value = actual_values.iloc[-1]

            # Define the forecast series: join last actual point with the last 6 predicted values
            forecast_dates = pd.concat([pd.Series([last_actual_date]), df_plot["Date"].iloc[-6:]]).tolist()
            forecast_values = pd.concat([pd.Series([last_actual_value]), df_plot["Predicted"].iloc[-6:]]).tolist()

            fig = go.Figure()

            # Plot actual values as a solid line
            fig.add_trace(go.Scatter(
                x=actual_dates,
                y=actual_values,
                mode='lines',
                name='Actuals',
                line=dict(color='cyan', width=2)
            ))

            # Plot forecast as a dotted line
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines',
                name='Predicted',
                line=dict(color='magenta', dash='dot', width=2)
            ))

            fig.update_layout(
                title=f"Forecast vs Actual for SKU: {sku_choice}",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload both the Demand Data and External Variables Excel files using the sidebar.")
