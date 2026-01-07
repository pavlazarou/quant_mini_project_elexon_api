import requests
import pandas as pd
import matplotlib.pyplot as plt

#API Configuration
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

# Dictionary of Datasets to track
DATASETS = {"Demand Forecast (Day-ahead)" :"forecast/demand/day-ahead/?format=json", 
            "Indicated Forecast (Day-ahead)" :"forecast/indicated/day-ahead/?format=json"
            }

def fetch_forecast_data(endpoint):
    """Fetch data for a given endpoint from Elexon API"""
    headers = {
        "accept": "application/json"
    }

    url = f"{BASE_URL}/{endpoint}"

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error Fetching {endpoint}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching {endpoint}: {e}")
        return None

def analyse_forecast_data():
    """Fetch data for all endpoints and perform statistical analysis."""
    analysis_results = {}
    dataframes = {}

    for dataset_name, endpoint in DATASETS.items():
        data = fetch_forecast_data(endpoint)

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

                # Calculate moving averages (48-hour window assuming hourly data)
                df['transmissionSystemDemand_MA'] = df['transmissionSystemDemand'].rolling(window=24).mean()
                df['nationalDemand_MA'] = df['nationalDemand'].rolling(window=24).mean()

                analysis_results[dataset_name] = {
                    'transmissionSystemDemand_mean': df['transmissionSystemDemand'].mean(),
                    'transmissionSystemDemand_std': df['transmissionSystemDemand'].std(),
                    'nationalDemand_mean': df['nationalDemand'].mean(),
                    'nationalDemand_std': df['nationalDemand'].std()
                }
            elif dataset_name == "Indicated Forecast (Day-ahead)":
                df['indicatedGeneration'] = pd.to_numeric(df['indicatedGeneration'], errors='coerce')
                df['indicatedDemand'] = pd.to_numeric(df['indicatedDemand'], errors='coerce')
                
                # Calculate moving averages
                df['indicatedGeneration_MA'] = df['indicatedGeneration'].rolling(window=24).mean()
                df['indicatedDemand_MA'] = df['indicatedDemand'].rolling(window=24).mean()

                analysis_results[dataset_name] = {
                    'indicatedGeneration_mean': df['indicatedGeneration'].mean(),
                    'indicatedGeneration_std': df['indicatedGeneration'].std(),
                    'indicatedDemand_mean': df['indicatedDemand'].mean(),
                    'indicatedDemand_std': df['indicatedDemand'].std(),
                }

            
            dataframes[dataset_name] = df
        else:
            analysis_results[dataset_name] = "Error fetching data"
            dataframes[dataset_name] = None
    
    # Calculate spread between demand forecast and indicated demand
    if ("Demand Forecast (Day-ahead)" in dataframes and dataframes["Demand Forecast (Day-ahead)"] is not None and
        "Indicated Forecast (Day-ahead)" in dataframes and dataframes["Indicated Forecast (Day-ahead)"] is not None):
        
        df_demand = dataframes["Demand Forecast (Day-ahead)"]
        df_indicated = dataframes["Indicated Forecast (Day-ahead)"]
        
        # Merge on startTime to align the data
        merged_df = pd.merge(df_demand[['startTime', 'nationalDemand', 'transmissionSystemDemand']], 
                           df_indicated[['startTime', 'indicatedDemand']], 
                           on='startTime', how='inner')
        
        # Calculate spreads
        merged_df['demand_spread_national'] = merged_df['indicatedDemand'] - merged_df['nationalDemand']
        merged_df['demand_spread_transmission'] = merged_df['indicatedDemand'] - merged_df['transmissionSystemDemand']
        merged_df['abs_spread_national'] = merged_df['demand_spread_national'].abs()
        merged_df['abs_spread_transmission'] = merged_df['demand_spread_transmission'].abs()
        
        # Calculate moving averages for spreads
        merged_df['demand_spread_national_MA'] = merged_df['demand_spread_national'].rolling(window=24).mean()
        merged_df['demand_spread_transmission_MA'] = merged_df['demand_spread_transmission'].rolling(window=24).mean()
        
        # Extract hour of day for segmentation
        merged_df['hour_of_day'] = merged_df['startTime'].dt.hour
        
        dataframes['Demand Spread Analysis'] = merged_df
        
        # Calculate correlation between indicatedDemand and (nationalDemand, transmissionSystemDemand)
        national_correlation = merged_df['indicatedDemand'].corr(merged_df['nationalDemand'])
        transmission_correlation = merged_df['indicatedDemand'].corr(merged_df['transmissionSystemDemand'])
        
        # Get descriptive statistics for the spread
        spread_describe_national = merged_df['demand_spread_national'].describe()
        spread_describe_transmission = merged_df['demand_spread_transmission'].describe()
        
        # Calculate mean absolute spread by hour of day
        hourly_abs_spread_national = merged_df.groupby('hour_of_day')['abs_spread_national'].mean()
        hourly_abs_spread_transmission = merged_df.groupby('hour_of_day')['abs_spread_transmission'].mean()
        
        # Add spread statistics to analysis results
        analysis_results['Demand Spread Analysis'] = {
            'demand_spread_national_mean': merged_df['demand_spread_national'].mean(),
            'demand_spread_national_std': merged_df['demand_spread_national'].std(),
            'abs_spread_national_mean': merged_df['abs_spread_national'].mean(),
            'correlation_indicated_vs_national': national_correlation,
            'spread_describe_national': spread_describe_national.to_dict(),
            'hourly_abs_spread_mean_national': hourly_abs_spread_national.to_dict(),
            'demand_spread_transmission_mean': merged_df['demand_spread_transmission'].mean(),
            'demand_spread_transmission_std': merged_df['demand_spread_transmission'].std(),
            'abs_spread_transmission_mean': merged_df['abs_spread_transmission'].mean(),
            'correlation_indicated_vs_transmission': transmission_correlation,
            'spread_describe_transmission': spread_describe_transmission.to_dict(),
            'hourly_abs_spread_mean_transmission': hourly_abs_spread_transmission.to_dict()
        }
    
    return analysis_results, dataframes
   
def plot_forecast_data(dataframes):
    """Plot the forecast data using matplotlib with moving averages and spread analysis."""
    
    fig = plt.figure(figsize=(30, 35))
    
    # Create subplots: 2 rows for time series, 2 rows for histogram and hourly analysis
    gs = fig.add_gridspec(7, 2, hspace=0.5, wspace=0.3)
    
    # Time series plots
    ax1 = fig.add_subplot(gs[0, 0])  # Demand Forecast - spans both columns - Transmission and National
    ax2 = fig.add_subplot(gs[0, 1])  # Indicated Forecast - spans both columns  - Generation and Demand
    ax3 = fig.add_subplot(gs[2, 0])  # Spread Analysis - spans both columns - Transmission
    ax4 = fig.add_subplot(gs[2, 1])  # Spread Analysis - spans both columns - National
    
    # Analysis plots
    ax5 = fig.add_subplot(gs[4, 0])  # Histogram - National
    ax6 = fig.add_subplot(gs[4, 1])  # Histogram - Transmission
    ax7 = fig.add_subplot(gs[6, 0])  # Hourly segmentation - Transmission
    ax8 = fig.add_subplot(gs[6, 1])  # Hourly segmentation - National
    
    fig.suptitle('Elexon Forecast Data Analysis with Spread Calculations', fontsize=16)
    
    # Plot Demand Forecast
    if "Demand Forecast (Day-ahead)" in dataframes and dataframes["Demand Forecast (Day-ahead)"] is not None:
        df_demand = dataframes["Demand Forecast (Day-ahead)"]
        
        # Transmission System Demand
        ax1.plot(df_demand['startTime'], df_demand['transmissionSystemDemand'], 
                label='Transmission System Demand', color='blue', alpha=0.7)
        ax1.plot(df_demand['startTime'], df_demand['transmissionSystemDemand_MA'], 
                label='TSD Moving Average (24h)', color='blue', linestyle='--', linewidth=2)
        
        # National Demand
        ax1.plot(df_demand['startTime'], df_demand['nationalDemand'], 
                label='National Demand', color='green', alpha=0.7)
        ax1.plot(df_demand['startTime'], df_demand['nationalDemand_MA'], 
                label='National Demand MA (24h)', color='green', linestyle='--', linewidth=2)
        
        ax1.set_title('Demand Forecast (Day-ahead)')
        ax1.set_ylabel('Demand (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot Indicated Forecast
    if "Indicated Forecast (Day-ahead)" in dataframes and dataframes["Indicated Forecast (Day-ahead)"] is not None:
        df_indicated = dataframes["Indicated Forecast (Day-ahead)"]
        
        # Indicated Generation
        ax2.plot(df_indicated['startTime'], df_indicated['indicatedGeneration'], 
                label='Indicated Generation', color='red', alpha=0.7)
        ax2.plot(df_indicated['startTime'], df_indicated['indicatedGeneration_MA'], 
                label='Generation MA (24h)', color='red', linestyle='--', linewidth=2)
        
        # Indicated Demand
        ax2.plot(df_indicated['startTime'], df_indicated['indicatedDemand'], 
                label='Indicated Demand', color='orange', alpha=0.7)
        ax2.plot(df_indicated['startTime'], df_indicated['indicatedDemand_MA'], 
                label='Demand MA (24h)', color='orange', linestyle='--', linewidth=2)
        
        
        ax2.set_title('Indicated Forecast (Day-ahead)')
        ax2.set_ylabel('Generation/Demand (MW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot Demand Spread Analysis
    if "Demand Spread Analysis" in dataframes and dataframes["Demand Spread Analysis"] is not None:
        df_spread = dataframes["Demand Spread Analysis"]
        
        # Demand Spread vs Transmission System Demand
        ax3.plot(df_spread['startTime'], df_spread['demand_spread_transmission'], 
                label='Spread (Indicated - Transmission)', color='magenta', alpha=0.7)
        ax3.plot(df_spread['startTime'], df_spread['demand_spread_transmission_MA'], 
                label='Spread MA (24h)', color='magenta', linestyle='--', linewidth=2)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Zero Line')
        
        ax3.set_title('Demand Spread Analysis: Indicated vs Transmission System Demand')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Spread (MW)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Demand Spread vs National Demand
        ax4.plot(df_spread['startTime'], df_spread['demand_spread_national'], 
                label='Spread (Indicated - National)', color='cyan', alpha=0.7)
        ax4.plot(df_spread['startTime'], df_spread['demand_spread_national_MA'], 
                label='Spread MA (24h)', color='cyan', linestyle='--', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        ax4.set_title('Demand Spread Analysis: Indicated vs National Demand')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Spread (MW)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Histogram of National spread
        ax5.hist(df_spread['demand_spread_national'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_title('Histogram of National Demand Spread')
        ax5.set_xlabel('Spread (MW)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)

        # Histogram of Transmission spread
        ax6.hist(df_spread['demand_spread_transmission'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.set_title('Histogram of Transmission Demand Spread')
        ax6.set_xlabel('Spread (MW)')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        # Hourly segmentation of mean absolute transmission spread
        hourly_data_transmission = df_spread.groupby('hour_of_day')['abs_spread_transmission'].mean()
        ax7.bar(hourly_data_transmission.index, hourly_data_transmission.values, alpha=0.7, color='lightcoral', edgecolor='black')
        ax7.set_title('Mean Absolute Transmission Spread by Hour of Day')
        ax7.set_xlabel('Hour of Day')
        ax7.set_ylabel('Mean Absolute Spread (MW)')
        ax7.grid(True, alpha=0.3)

        # Hourly segmentation of mean absolute national spread
        hourly_data_national = df_spread.groupby('hour_of_day')['abs_spread_national'].mean()
        ax8.bar(hourly_data_national.index, hourly_data_national.values, alpha=0.7, color='lightcoral', edgecolor='black')
        ax8.set_title('Mean Absolute National Spread by Hour of Day')
        ax8.set_xlabel('Hour of Day')
        ax8.set_ylabel('Mean Absolute Spread (MW)')
        ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  
   
if __name__ == "__main__":
    results, dataframes = analyse_forecast_data()
    for dataset, stats in results.items():
        print(f"\nAnalysis for {dataset}:")
        if isinstance(stats, dict):
            for stat_name, value in stats.items():
                if 'spread_describe' in stat_name:
                    print(f"  Spread Descriptive Statistics for {stat_name.replace('spread_describe_', '').replace('_', ' ').title()}:")
                    for desc_stat, desc_value in value.items():
                        print(f"    {desc_stat}: {desc_value:.2f}")
                elif 'hourly_abs_spread_mean' in stat_name:
                    print(f"  Mean Absolute Spread by Hour of Day for {stat_name.replace('hourly_abs_spread_mean_', '').replace('_', ' ').title()}:")
                    for hour, mean_spread in sorted(value.items()):
                        print(f"    Hour {hour}: {mean_spread:.2f} MW")
                else:
                    print(f"  {stat_name}: {value:.2f}")
        else:
            print(f"  {stats}")
        print()
    
    # Plot the data
    plot_forecast_data(dataframes) 
