import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Default dates: Looking at historical data from a specific week
historical_dates_from = "from=2025-12-10"
historical_dates_to = "to=2025-12-17"
#Default: Looking at settlement periods for the whole day
settlement_periods_from = "fromSettlementPeriod=1"
settlement_periods_to = "toSettlementPeriod=48"    


class ElexonClient:
    def __init__(self, base_url="https://data.elexon.co.uk/bmrs/api/v1", historical_dates_from=historical_dates_from, historical_dates_to=historical_dates_to, settlement_periods_from=settlement_periods_from, settlement_periods_to=settlement_periods_to):
        self.base_url = base_url
        self.historical_dates_from = historical_dates_from
        self.historical_dates_to = historical_dates_to
        self.settlement_periods_from = settlement_periods_from
        self.settlement_periods_to = settlement_periods_to
        # Dictionary of datasets to track
        self.datasets = {
            "Actual Total Load": "demand/actual/total",
            }
        self.headers = {"accept": "application/json"} 
        print("[Elexon] Client initialised")

    def test_connection(self, endpoint, historical_dates_from=None, historical_dates_to=None, settlement_periods_from=None, settlement_periods_to=None):
        if historical_dates_from is None:
            historical_dates_from = self.historical_dates_from
        if historical_dates_to is None:
            historical_dates_to = self.historical_dates_to
        if settlement_periods_from is None:
            settlement_periods_from = self.settlement_periods_from
        if settlement_periods_to is None:
            settlement_periods_to = self.settlement_periods_to

        url = f"{self.base_url}/{endpoint}?{historical_dates_from}&{historical_dates_to}&{settlement_periods_from}&{settlement_periods_to}"

        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                print("[Elexon] Connection Successful!")
                return True
            else:
                print(f"[Elexon] Connection Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"[Elexon] Connection Error: {e}")
            return False


    def fetch_historical_data(self, endpoint, historical_dates_from=None, historical_dates_to=None, settlement_periods_from=None, settlement_periods_to=None):
        """Fetch historical data for a given endpoint from Elexon API"""
        if historical_dates_from is None:
            historical_dates_from = self.historical_dates_from
        if historical_dates_to is None:
            historical_dates_to = self.historical_dates_to
        if settlement_periods_from is None:
            settlement_periods_from = self.settlement_periods_from
        if settlement_periods_to is None:
            settlement_periods_to = self.settlement_periods_to

        url = f"{self.base_url}/{endpoint}?{historical_dates_from}&{historical_dates_to}&{settlement_periods_from}&{settlement_periods_to}"

        try:
            response = requests.get(url,headers=self.headers, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[Elexon] Error Fetching {endpoint}: {response.status_code}")
                return None
        except Exception as e:
            print(f"[Elexon] Exception fetching {endpoint}: {e}")
            return None
    
class LoadPredictor:
    def __init__(self):
        self.analysis_results = {}
        self.dataframes = {}
        self.client = ElexonClient()

    def analyse_historical_data(self, historical_dates_from=historical_dates_from, historical_dates_to=historical_dates_to, settlement_periods_from=settlement_periods_from, settlement_periods_to=settlement_periods_to):
        """Fetch data and perform analysis"""

        for dataset_name, endpoint in self.client.datasets.items():
            data = self.client.fetch_historical_data(endpoint, historical_dates_from, historical_dates_to, settlement_periods_from, settlement_periods_to)

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

                self.dataframes[dataset_name] = df

                # Basic stats
                self.analysis_results[dataset_name] = {
                    'quantity_mean': df['quantity'].mean(),
                    'quantity_std': df['quantity'].std(),
                    'n_samples': len(df)
                }
            else:
                self.analysis_results[dataset_name] = "No data available"
                self.dataframes[dataset_name] = None

        # Predict next 48 settlement periods using Random Forest
        if 'Actual Total Load' in self.dataframes and self.dataframes['Actual Total Load'] is not None:
            df = self.dataframes['Actual Total Load'].copy()

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

                # Baseline evaluation on historical sample
                y_true = y.values

                # persistence baseline: y(t) = lag1
                y_pred_persist = df['lag1'].values

                def mae(a, b):
                    # Mean Absolute Error
                    return float(np.mean(np.abs(a-b)))
                
                def rmse(a, b):
                    # Root Mean Squared Error
                    return float(np.sqrt(np.mean((a - b) ** 2)))
                
                def bias(true, pred):
                    # Mean Bias Error - average of (pred - true)
                    return float(np.mean(pred - true))
                
                y_pred_model = model.predict(X_scaled)

                self.analysis_results['Baseline Metrics'] = {
                    "model_mae": mae(y_true, y_pred_model),
                    "model_rmse": rmse(y_true, y_pred_model),
                    "model_bias": bias(y_true, y_pred_model),

                    "persistence_mae": mae(y_true, y_pred_persist),
                    "persistence_rmse": rmse(y_true, y_pred_persist),
                    "persistence_bias": bias(y_true, y_pred_persist),

                    "n_samples": int(len(df))
                }

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
                next_times = pd.date_range(start=last_time + pd.Timedelta(minutes=30), periods=48, freq="30min")
                predictions_df = pd.DataFrame({
                    'startTime': next_times,
                    'settlementPeriod': list(range(1, 49)),
                    'quantity': predictions,  # Already floats
                    'is_predicted': True
                })
                self.dataframes['Predictions'] = predictions_df

                print(f"\nPredicted quantities for next 48 settlement periods:")
                print(predictions_df.to_string(index=False))

                self.analysis_results['Predictions'] = {
                    'settlement_periods': list(range(1, 49)),
                    'predicted_quantities': predictions
                }
            else:
                self.analysis_results['Predictions'] = "Not enough data for prediction"

        return self.analysis_results, self.dataframes

    def plot_data(self,actual_df, predictions_df=None):
        """Plot actual load data and predictions if available"""
        if actual_df is None:
            print("No actual data available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot actual data
        actual_df = actual_df.sort_values('startTime')
        plt.plot(actual_df['startTime'], actual_df['quantity'], 
                label='Actual Total Load', color='blue')
        
        # Plot predictions if available
        if predictions_df is not None:
            plt.plot(predictions_df['startTime'], predictions_df['quantity'], 
                    label='Predicted Load', color='orange', linestyle='--')
        
        plt.xlabel('Time')
        plt.ylabel('Quantity (MW)')
        plt.title('Actual Total Load and Predictions over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

def main(historical_dates_from=historical_dates_from, historical_dates_to=historical_dates_to, settlement_periods_from=settlement_periods_from, settlement_periods_to=settlement_periods_to):
    print("\n" + "="*80)
    print("ACTUAL LOAD PREDICTOR")
    print("="*80)

    # Create Elexon Client
    elexon = ElexonClient()

    # Test connection for each dataset
    for dataset_name, endpoint in elexon.datasets.items():
        if not elexon.test_connection(endpoint, historical_dates_from=historical_dates_from, historical_dates_to=historical_dates_to, settlement_periods_from=settlement_periods_from, settlement_periods_to=settlement_periods_to):
            print(f"[Main] Failed to connect to Elexon for {dataset_name}. Check Inputs.")
            return

    # Create Load Predictor
    predictor = LoadPredictor()
    results, dfs = predictor.analyse_historical_data(historical_dates_from=historical_dates_from, historical_dates_to=historical_dates_to, settlement_periods_from=settlement_periods_from, settlement_periods_to=settlement_periods_to)

    # Add datsets
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
    
    # Plotting Data
    predictor.plot_data(dfs.get('Actual Total Load'), dfs.get('Predictions'))
    print("[Main] Data Plot Complete!")
    print("[Main] Total Load Prediction Complete!")

if __name__ == "__main__":
    main(historical_dates_from="from=2026-01-01", historical_dates_to="to=2026-01-07", settlement_periods_from="fromSettlementPeriod=1", settlement_periods_to="toSettlementPeriod=48")
