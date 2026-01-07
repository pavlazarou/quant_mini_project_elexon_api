import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# API Configuration
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

# Dictionary of datasets to track
DATASETS = {
            "Historical Latest Demand Forecast (Day-ahead)" :"forecast/demand/day-ahead/latest",
            "Actual Total Load": "demand/actual/total",
            }
# Looking at historical data from a specific week
HISORICAL_DATES_FROM = "from=2025-11-01"
HISORICAL_DATES_TO = "to=2025-11-07"

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
    """Fetch historical data for all endpoints and perform Linear Regression"""
    analysis_results = {}
    dataframes = {}

    for dataset_name, endpoint in DATASETS.items():
        data = fetch_historical_data(endpoint)

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

                analysis_results[dataset_name] = {
                    'transmissionSystemDemand_mean': df['transmissionSystemDemand'].mean(),
                    'transmissionSystemDemand_std': df['transmissionSystemDemand'].std(),
                }

            elif dataset_name == "Actual Total Load":
                df['settlementPeriod'] = pd.to_numeric(df['settlementPeriod'], errors='coerce')
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

                df = df.dropna()

                analysis_results[dataset_name] = {
                    'quantity_mean': df['quantity'].mean(),
                    'quantity_std': df['quantity'].std(),
                }
            

            dataframes[dataset_name] = df
        else:
            analysis_results[dataset_name] = "No data available"
            dataframes[dataset_name] = None

    # Determine relationship between Transmission System Demand and quantity using linear regression
    if dataframes["Historical Latest Demand Forecast (Day-ahead)"] is not None and dataframes["Actual Total Load"] is not None:
        df_demand = dataframes["Historical Latest Demand Forecast (Day-ahead)"]
        df_actual = dataframes["Actual Total Load"]

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

            analysis_results["Spread Analysis"] = {
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

            relationship_commentary = interpret_relationship(r2_mean)

            analysis_results['Relationship Analysis'] = {
                'slope': float(slopes),
                'intercept': float(intercept),
                'r2_score': r2_mean,
                'n_samples': int(len(df_merged)),
                'interpretation': relationship_commentary
            }
            dataframes['Merged'] = df_merged
        else:
            analysis_results['Relationship Analysis'] = "No matching data to merge"
            dataframes['Merged'] = None
    return analysis_results, dataframes

def interpret_relationship(r2_score):
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
    
def plot_relationship(results, dfs):
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
if __name__ == "__main__":
    results, dfs = analyse_historical_data()
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
    plot_relationship(results, dfs)
