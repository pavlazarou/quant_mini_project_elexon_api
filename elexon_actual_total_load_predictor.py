import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# API Configuration
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

# Dictionary of datasets to track
DATASETS = {
            "Actual Total Load": "demand/actual/total",
            }
# Looking at historical data from a specific week
HISORICAL_DATES_FROM = "from=2025-12-10"
HISORICAL_DATES_TO = "to=2025-12-17"

# Looking at settlement periods for the whole day
SETTLEMENT_PERIODS_FROM = "fromSettlementPeriod=1"
SETTLEMENT_PERIODS_TO = "toSettlementPeriod=48"

def fetch_historical_data(endpoint):
    """Fetch historical data for a given endpoint from Elexon API"""
    headers = {
        "accept": "application/json"
    }

    url = f"{BASE_URL}/{endpoint}?{HISORICAL_DATES_FROM}&{HISORICAL_DATES_TO}&{SETTLEMENT_PERIODS_FROM}&{SETTLEMENT_PERIODS_TO}"

    try:
        response = requests.get(url,headers=headers, params={"format": "json"})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error Fetching {endpoint}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching {endpoint}: {e}")
        return None
    

def analyse_historical_data():
    """Fetch data and perform analysis"""
    analysis_results = {}
    dataframes = {}

    for dataset_name, endpoint in DATASETS.items():
        data = fetch_historical_data(endpoint)

        if data and 'data' in data:
            df = pd.DataFrame(data['data'])

            # Convert startTime to datetime
            if 'startTime' in df.columns:
                df['startTime'] = pd.to_datetime(df['startTime'])
                df = df.sort_values('startTime')
            
            # Convert relevant columns to numeric
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df['settlementPeriod'] = pd.to_numeric(df['settlementPeriod'], errors='coerce')

            df = df.dropna()

            dataframes[dataset_name] = df

            # Basic stats
            analysis_results[dataset_name] = {
                'quantity_mean': df['quantity'].mean(),
                'quantity_std': df['quantity'].std(),
                'n_samples': len(df)
            }
        else:
            analysis_results[dataset_name] = "No data available"
            dataframes[dataset_name] = None

    # Predict next 48 settlement periods using Random Forest
    if 'Actual Total Load' in dataframes and dataframes['Actual Total Load'] is not None:
        df = dataframes['Actual Total Load'].copy()

        # Create lagged feature
        df = df.sort_values('startTime')
        df['lag1'] = df['quantity'].shift(1)
        df = df.dropna()

        if len(df) > 0:
            # Features: settlementPeriod and lag1
            X = df[['settlementPeriod', 'lag1']].values  # Use .values to get numpy array
            y = df['quantity']

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)

            # Predict next 48 settlement periods
            predictions = []
            current_lag = df['quantity'].iloc[-1]  # Last quantity
            current_sp = df['settlementPeriod'].iloc[-1]

            for i in range(48):
                next_sp = (current_sp % 48) + 1  # Cycle 1 to 48
                next_X = scaler.transform(np.array([[next_sp, current_lag]]))
                pred = model.predict(next_X)[0]
                predictions.append(float(pred))  # Ensure float
                current_lag = pred  # Use prediction as next lag
                current_sp = next_sp

            # Create predictions dataframe
            last_time = df['startTime'].max()
            next_times = [last_time + pd.Timedelta(minutes=30 * i) for i in range(1, 49)]
            predictions_df = pd.DataFrame({
                'startTime': next_times,
                'settlementPeriod': list(range(1, 49)),
                'quantity': predictions,  # Already floats
                'is_predicted': True
            })
            dataframes['Predictions'] = predictions_df

            print(f"Predicted quantities for next 48 settlement periods:")
            print(predictions_df.to_string(index=False))

            analysis_results['Predictions'] = {
                'settlement_periods': list(range(1, 49)),
                'predicted_quantities': predictions
            }
        else:
            analysis_results['Predictions'] = "Not enough data for prediction"

    return analysis_results, dataframes

if __name__ == "__main__":
    results, dfs = analyse_historical_data()
    for dataset, result in results.items():
        print(f"\nAnalysis Results for {dataset}:")
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                elif isinstance(value, list):
                    print(f"  {key}: {value[:5]}...")  # Print first 5
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {result}")
    
    # Plotting
    if 'Actual Total Load' in dfs and dfs['Actual Total Load'] is not None:
        df = dfs['Actual Total Load'].sort_values('startTime')
        plt.figure(figsize=(12,6))
        plt.plot(df['startTime'], df['quantity'], label='Actual Total Load', color='blue')
        
        # Plot predictions on the same graph
        if 'Predictions' in results and isinstance(results['Predictions'], dict):
            sps = results['Predictions']['settlement_periods']
            preds = results['Predictions']['predicted_quantities']
            # Create future times
            last_time = df['startTime'].max()
            next_times = [last_time + pd.Timedelta(minutes=30 * i) for i in range(1, 49)]
            plt.plot(next_times, preds, label='Predicted Load', color='orange', linestyle='--')
        
        plt.xlabel('Time')
        plt.ylabel('Quantity (MW)')
        plt.title('Actual Total Load and Predictions over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
