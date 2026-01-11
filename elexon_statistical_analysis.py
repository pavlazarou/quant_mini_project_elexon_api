import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ElexonClient:
    def __init__(self, base_url="https://data.elexon.co.uk/bmrs/api/v1"):
        self.base_url = base_url
        # Dictionary of datasets to track
        self.datasets = {
            "Demand Forecast (Day-ahead)" :"forecast/demand/day-ahead?format=json", 
            "Indicated Forecast (Day-ahead)" :"forecast/indicated/day-ahead?format=json"
            }
        self.headers = {"accept": "application/json"} 

        print("[Elexon] Client initialised")

    def test_connection(self, endpoint):

        url = f"{self.base_url}/{endpoint}"

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


    def fetch_forecast_data(self, endpoint):
        """Fetch historical data for a given endpoint from Elexon API"""

        url = f"{self.base_url}/{endpoint}"

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
class StatisticalAnalysis:
    def __init__(self):
        self.client = ElexonClient()
        self.analysis_results = {}
        self.dataframes = {}

    def analyse_forecast_data(self):
        """Fetch data for all endpoints and perform statistical analysis."""

        for dataset_name, endpoint in self.client.datasets.items():
            data = self.client.fetch_forecast_data(endpoint)

            if data and 'data' in data:
                df = pd.DataFrame(data['data'])

                # Convert startTime to datetime if it exists
                if 'startTime' in df.columns:
                    df['startTime'] = pd.to_datetime(df['startTime'])
                    df = df.sort_values('startTime')

                # Convert relevant columns to numeric, errors='coerce' will turn non-numeric to NaN
                if dataset_name == "Demand Forecast (Day-ahead)":
                    df['transmissionSystemDemand'] = pd.to_numeric(df['transmissionSystemDemand'], errors='coerce')
                    df['nationalDemand'] = pd.to_numeric(df['nationalDemand'], errors='coerce')

                    # Calculate moving averages (24-hour window)
                    df['transmissionSystemDemand_MA'] = df['transmissionSystemDemand'].rolling(window=24).mean()
                    df['nationalDemand_MA'] = df['nationalDemand'].rolling(window=24).mean()

                    self.analysis_results[dataset_name] = {
                        'transmissionSystemDemand_mean': df['transmissionSystemDemand'].mean(),
                        'transmissionSystemDemand_std': df['transmissionSystemDemand'].std(),
                        'transmissionSystemDemand_median': df['transmissionSystemDemand'].median(),
                        'transmissionSystemDemand_min': df['transmissionSystemDemand'].min(),
                        'transmissionSystemDemand_max': df['transmissionSystemDemand'].max(), 
                        'nationalDemand_mean': df['nationalDemand'].mean(),
                        'nationalDemand_std': df['nationalDemand'].std(),
                        'nationalDemand_median': df['nationalDemand'].median(),
                        'nationalDemand_min': df['nationalDemand'].min(),
                        'nationalDemand_max': df['nationalDemand'].max()                        
                    }
                elif dataset_name == "Indicated Forecast (Day-ahead)":
                    df['indicatedGeneration'] = pd.to_numeric(df['indicatedGeneration'], errors='coerce')
                    df['indicatedDemand'] = pd.to_numeric(df['indicatedDemand'], errors='coerce')
                    
                    # Calculate moving averages
                    df['indicatedGeneration_MA'] = df['indicatedGeneration'].rolling(window=24).mean()
                    df['indicatedDemand_MA'] = df['indicatedDemand'].rolling(window=24).mean()

                    self.analysis_results[dataset_name] = {
                        'indicatedGeneration_mean': df['indicatedGeneration'].mean(),
                        'indicatedGeneration_std': df['indicatedGeneration'].std(),
                        'indicatedGeneration_median': df['indicatedGeneration'].median(),
                        'indicatedGeneration_min': df['indicatedGeneration'].min(),
                        'indicatedGeneration_max': df['indicatedGeneration'].max(),
                        'indicatedDemand_mean': df['indicatedDemand'].mean(),
                        'indicatedDemand_std': df['indicatedDemand'].std(),
                        'indicatedDemand_median': df['indicatedDemand'].median(),
                        'indicatedDemand_min': df['indicatedDemand'].min(),
                        'indicatedDemand_max': df['indicatedDemand'].max()
                    }

                
                self.dataframes[dataset_name] = df
            else:
                self.analysis_results[dataset_name] = "Error fetching data"
                self.dataframes[dataset_name] = None
        
        return self.analysis_results, self.dataframes
    
    @staticmethod 
    def plot_forecast_data(dataframes):
        """Plot the forecast data using Plotly with moving averages."""
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Demand Forecast (Day-ahead)', 'Indicated Forecast (Day-ahead)'))
        
        # Plot Demand Forecast
        if "Demand Forecast (Day-ahead)" in dataframes and dataframes["Demand Forecast (Day-ahead)"] is not None:
            df_demand = dataframes["Demand Forecast (Day-ahead)"]
            
            # Transmission System Demand
            fig.add_trace(
                go.Scatter(
                    x=df_demand['startTime'], 
                    y=df_demand['transmissionSystemDemand'], 
                    mode='lines', 
                    name='Transmission System Demand',
                    line=dict(color='blue', width=2)
                )
            )

            # TSD Moving Average
            fig.add_trace(
                go.Scatter(
                    x=df_demand['startTime'], 
                    y=df_demand['transmissionSystemDemand_MA'], 
                    mode='lines', 
                    name='TSD Moving Average (24h)',
                    line=dict(color='blue', width=2, dash='dash')
                )
            )
            # National Demand
            fig.add_trace(
                go.Scatter(
                    x=df_demand['startTime'], 
                    y=df_demand['nationalDemand'], 
                    mode='lines', 
                    name='National Demand',
                    line=dict(color='green', width=2)
                )
            )
            # National Demand MA        
            fig.add_trace(
                go.Scatter(
                    x=df_demand['startTime'], 
                    y=df_demand['nationalDemand_MA'], 
                    mode='lines', 
                    name='National Demand MA (24h)',
                    line=dict(color='green', width=2, dash='dash')
                )
            )           
        # Plot Indicated Forecast
        if "Indicated Forecast (Day-ahead)" in dataframes and dataframes["Indicated Forecast (Day-ahead)"] is not None:
            df_indicated = dataframes["Indicated Forecast (Day-ahead)"] 
            # Indicated Generation
            fig.add_trace(
                go.Scatter(
                    x=df_indicated['startTime'], 
                    y=df_indicated['indicatedGeneration'], 
                    mode='lines', 
                    name='Indicated Generation',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            # Generation MA
            fig.add_trace(
                go.Scatter(
                    x=df_indicated['startTime'], 
                    y=df_indicated['indicatedGeneration_MA'], 
                    mode='lines',
                    name='Generation MA (24h)',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=2, col=1
            )
            # Indicated Demand
            fig.add_trace(
                go.Scatter(
                    x=df_indicated['startTime'],
                    y=df_indicated['indicatedDemand'],
                    mode='lines',
                    name='Indicated Demand',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            # Demand MA
            fig.add_trace(
                go.Scatter(
                    x=df_indicated['startTime'],
                    y=df_indicated['indicatedDemand_MA'],
                    mode='lines',
                    name='Demand MA (24h)',
                    line=dict(color='orange', width=2, dash='dash')
                ),
                row=2, col=1
            )
        # Update layout    
        fig.update_layout(height=900, width=1200, title_text="Elexon Forecast Data Analysis", showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Demand (MW)", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Generation/Demand (MW)", row=2, col=1) 
        fig.show()
        
def main():
        analysis = StatisticalAnalysis()
        results, dataframes = analysis.analyse_forecast_data()
        for dataset, stats in results.items():
            print(f"\nAnalysis for {dataset}:")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {stats}")
            print()
        
        # Plot the data
        StatisticalAnalysis.plot_forecast_data(dataframes) 
   
if __name__ == "__main__":
    main()
