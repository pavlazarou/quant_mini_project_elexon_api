import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

class ElexonClient:
    def __init__(self, base_url="https://data.elexon.co.uk/bmrs/api/v1"):
        self.base_url = base_url
        # Dictionary of datasets to track
        self.datasets = {
            "Historical Latest Demand Forecast (Day-ahead)" :"forecast/demand/day-ahead/latest",
            "Actual Total Load": "demand/actual/total",
            }
        # Looking at historical data from a specific week
        self.historical_dates_from = "from=2025-11-01"
        self.historical_dates_to = "to=2025-11-07"
        # Looking at settlement periods for the whole day
        self.settlement_periods_from = "fromSettlementPeriod=1"
        self.settlement_periods_to = "toSettlementPeriod=48"    
        self.headers = {"accept": "application/json"} 

        print("[Elexon] Client initialised")

    def test_connection(self, endpoint):

        url = f"{self.base_url}/{endpoint}?{self.historical_dates_from}&{self.historical_dates_to}&{self.settlement_periods_from}&{self.settlement_periods_to}"

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


    def fetch_historical_data(self, endpoint):
        """Fetch historical data for a given endpoint from Elexon API"""

        url = f"{self.base_url}/{endpoint}?{self.historical_dates_from}&{self.historical_dates_to}&{self.settlement_periods_from}&{self.settlement_periods_to}"

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
class linear_regression_model:
    def __init__(self):
        self.client = ElexonClient()
        self.analysis_results = {}
        self.dataframes = {}

    def analyse_historical_data(self):
        """Fetch historical data for all endpoints and perform Linear Regression"""

        for dataset_name, endpoint in self.client.datasets.items():
            data = self.client.fetch_historical_data(endpoint)

            if data and 'data' in data:
                df = pd.DataFrame(data['data'])

                # Convert startTime to datetime if it exists
                if 'startTime' in df.columns:
                    df['startTime'] = pd.to_datetime(df['startTime'], utc=True, errors='coerce')
                    df = df.sort_values('startTime')
                
                #Convert relevant columns to numeric, errors='coerce' will turn non-numeric to NaN
                if dataset_name == "Historical Latest Demand Forecast (Day-ahead)":
                    df['transmissionSystemDemand'] = pd.to_numeric(df['transmissionSystemDemand'], errors='coerce')
                    df['settlementPeriod'] = pd.to_numeric(df['settlementPeriod'], errors='coerce')

                    df = df.dropna()

                    self.analysis_results[dataset_name] = {
                        'transmissionSystemDemand_mean': df['transmissionSystemDemand'].mean(),
                        'transmissionSystemDemand_std': df['transmissionSystemDemand'].std(),
                    }

                elif dataset_name == "Actual Total Load":
                    df['settlementPeriod'] = pd.to_numeric(df['settlementPeriod'], errors='coerce')
                    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

                    df = df.dropna()

                    self.analysis_results[dataset_name] = {
                        'quantity_mean': df['quantity'].mean(),
                        'quantity_std': df['quantity'].std(),
                    }
                

                self.dataframes[dataset_name] = df
            else:
                self.analysis_results[dataset_name] = "No data available"
                self.dataframes[dataset_name] = None

        # Determine relationship between Transmission System Demand and quantity using linear regression
        if self.dataframes["Historical Latest Demand Forecast (Day-ahead)"] is not None and self.dataframes["Actual Total Load"] is not None:
            df_demand = self.dataframes["Historical Latest Demand Forecast (Day-ahead)"]
            df_actual = self.dataframes["Actual Total Load"]

            join_keys = []
            if"startTime" in df_demand.columns and "startTime" in df_actual.columns:
                join_keys = ["startTime", "settlementPeriod"]
            else:
                # Fallback to settlementPeriod only
                join_keys = ["settlementPeriod"]

            # Merge on settlementPeriod
            df_merged = pd.merge(df_demand, df_actual, on=join_keys, how='inner', suffixes=('_demand', '_actual'))
            # Only drop rows where the columns we actually use are missing
            df_merged = df_merged.dropna(subset=["transmissionSystemDemand", "quantity"])

            if not df_merged.empty:
                df_merged["spread"] = df_merged["quantity"] - df_merged["transmissionSystemDemand"] 
                df_merged["abs_spread"] = df_merged["spread"].abs()

                self.analysis_results["Spread Analysis"] = {
                    "mean_spread": float(df_merged["spread"].mean()),
                    "std_spread": float(df_merged["spread"].std()),
                    "p05_spread": float(df_merged["spread"].quantile(0.05)),
                    "p95_spread": float(df_merged["spread"].quantile(0.95)),
                    "min_abs_spread": float(df_merged["abs_spread"].min()),
                    "max_abs_spread": float(df_merged["abs_spread"].max()),
                    "mean_abs_spread": float(df_merged["abs_spread"].mean()),
                    "n_samples": int(len(df_merged))
                }

                X = df_merged[['transmissionSystemDemand']]
                y = df_merged['quantity']

                model = LinearRegression()
                model.fit(X, y)
                slopes = model.coef_[0]
                intercept = model.intercept_
                scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                r2_mean = float(scores.mean())
                print(f"Linear Regression R2 for Transmission System Demand -> Actual Quantity: {r2_mean:.3f}")

                relationship_commentary = self.interpret_relationship(r2_mean)

                self.analysis_results['Relationship Analysis'] = {
                    'slope': float(slopes),
                    'intercept': float(intercept),
                    'r2_score': r2_mean,
                    'n_samples': int(len(df_merged)),
                    'interpretation': relationship_commentary
                }
                self.dataframes['Merged'] = df_merged
            else:
                self.analysis_results['Relationship Analysis'] = "No matching data to merge"
                self.dataframes['Merged'] = None
        return self.analysis_results, self.dataframes

    def interpret_relationship(self,r2_score):
        """Interpret R^2 for relationship between: Actual Total Load and Transmission System Demand"""
        if r2_score <0:
            return (
                "R^2 < 0: The transmission demand forecast has no meaningful relationship with actual load in this setup "
                "and performs worse than a naive mean-based estimate."
                )
        elif r2_score < 0.1:
            return (
                "0 <= R^2 < 0.1: The transmission demand forecast contains almost no information about actual load. "
                "Knowing the forecast barely improves estimation of realised load."
                )
        elif r2_score < 0.3:
            return (
                "0.1 <= R^2 <0.3 : The transmission demand forecast contains a weak but real signal about actual load. "
                "Most of the variation is driven by other factors."
            )
        elif r2_score < 0.6:
            return (
                "0.3 <= R^2 <0.6 : The transmission demand forecast explains a meaningful"
                " portion of the variation in actual load, but a large amount of variability"
                " \nremains unexplained due to system noise, weather, and operational effects."
            )
        elif r2_score < 0.8:
            return (
                "0.6 <= R^2 <0.8 : The transmission demand forecast has a strong relationship with actual load and explains most "
                "of the observed variation."
            )
        else:
            return (
                "R^2 >= 0.8 : The transmission demand forecast and actual load are very tightly coupled and behave almost like "
                "the same quantity. Either the forecast is extremely accurate or the two series are nearly identical." 
            )
        
    def plot_relationship(self, results, dfs):
            # Plotting useful visualizations
        # Plot relationship between Transmission System Demand and Actual Quantity
        if 'Merged' in dfs and dfs['Merged'] is not None:
            df = dfs['Merged']
            fig = plt.figure(figsize=(30,35))
            # create subplopt for scatter plot and histogram
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            # scatter plot
            ax0 = fig.add_subplot(gs[0, 0])

            # Histogram
            ax1 = fig.add_subplot(gs[1, 0])

            fig.suptitle('Transmission System Demand vs Actual Quantity Analysis', fontsize=16)
            
            # Plot scatter plot
            ax0.scatter(df['transmissionSystemDemand'], df['quantity'], alpha=0.5)
            ax0.set_xlabel('Transmission System Demand (MW)')
            ax0.set_ylabel('Actual Quantity (MW)')
            ax0.set_title('Scatter Plot: Transmission System Demand vs Actual Quantity')
            # Add regression line
            if 'Relationship Analysis' in results and isinstance(results['Relationship Analysis'], dict):
                slope = results['Relationship Analysis']['slope']
                intercept = results['Relationship Analysis']['intercept']
                x_vals = np.linspace(df['transmissionSystemDemand'].min(), df['transmissionSystemDemand'].max(), 100)
                y_vals = slope * x_vals + intercept
                ax0.plot(x_vals, y_vals, color='red', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')
                ax0.legend()
            ax0.grid(True)

            # Plot spread histogram
            ax1.hist(df['spread'], bins=50, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Spread (Actual - Forecast)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Histogram of Actual - Forecast Spreads')
            ax1.axvline(df['spread'].mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {df["spread"].mean():.2f}')
            ax1.legend()
            ax1.grid(True)
        plt.show()

def main():
    print("\n" + "="*80)
    print("ELEXON LINEAR REGRESSION ANALYSIS")
    print("="*80)

    # Create Elexon Client
    elexon = ElexonClient()

    # Test connection for each dataset
    for dataset_name, endpoint in elexon.datasets.items():
        if not elexon.test_connection(endpoint):
            print(f"[Main] Failed to connect to Elexon for {dataset_name}. Check Inputs.")
            return

    # Create Linear Regression Model
    model = linear_regression_model()
     # Add datsets
    results, dfs = model.analyse_historical_data()
    for dataset, result in results.items():
        print(f"\nAnalysis Results for {dataset}:")
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                elif isinstance(value, list):
                    print(f"  {key}: {value}")
                elif key == "interpretation":
                    print(f"  interpretation: {value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {result}")
    # Plot the data
    model.plot_relationship(results, dfs)

if __name__ == "__main__":
    main()
