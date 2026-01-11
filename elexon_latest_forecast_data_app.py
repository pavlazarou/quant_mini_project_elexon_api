import requests
import pandas as pd
import tkinter as tk
from tkinter import ttk



class ElexonClient:
    def __init__(self, base_url="https://data.elexon.co.uk/bmrs/api/v1"):
        self.base_url = base_url
        # Dictionary of Datasets to track
        self.datasets = {
            "Demand Forecast (Day-ahead)" :"forecast/demand/day-ahead?format=json", 
            "Indicated Forecast (Day-ahead)" :"forecast/indicated/day-ahead?format=json"
        }
        self.headers = {
            "accept": "application/json"
        }
        print("[Elexon] Client Initialised.")

    def test_connection(self, endpoint):
        """Testing the connection to Elexon API"""
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
        """Fetch the latest data for a given endpoint from Elexon API"""

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
    
class ForecastDataApp:
    def __init__(self):
        self.client = ElexonClient()

    def update_table(self):
        """Fetch data for all endpoints and update the table display."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Fetch and display data for each dataset
        for dataset_name, endpoint in self.client.datasets.items():
            data = self.client.fetch_forecast_data(endpoint)

            if data and 'data' in data:
                entry = data['data'][0]  # latest record

                self.tree.insert('', 'end', values=(
                    dataset_name,
                    entry.get('settlementDate', 'N/A'),
                    entry.get('settlementPeriod', 'N/A'),
                    entry.get('publishTime', 'N/A'),

                    # Demand forecast fields
                    entry.get('transmissionSystemDemand', 'N/A'),
                    entry.get('nationalDemand', 'N/A'),

                    # Indicated forecast fields
                    entry.get('indicatedGeneration', 'N/A'),
                    entry.get('indicatedDemand', 'N/A'),
                    entry.get('indicatedImbalance', 'N/A'),
                    entry.get('indicatedMargin', 'N/A'),
                ))
            else:
                self.tree.insert('', 'end', values=(
                    endpoint,
                    "Error", "Error", "Error",
                    "Error", "Error",
                    "Error", "Error", "Error", "Error"
                ))
    def setup_gui(self):
        """Setup the GUI components"""
    # Create main window
    root = tk.Tk()
    root.title("Elexon Forecast Data Application")
    root.geometry("800x400")

    # Create title label
    title_label = tk.Label(root, text="Elexon Forecast Data", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=10)

    # Create frame for table
    table_frame = tk.Frame(root)
    table_frame.pack(pady=10, padx=20, fill='both', expand=True)

    # Create Tree Widget for displaying data
    columns = (
        "Dataset",
        "Settlement Date",
        "Settlement Period",
        "Publish Time",
        "Transmission System Demand",
        "National Demand",
        "Indicated Generation",
        "Indicated Demand",
        "Indicated Imbalance",
        "Indicated Margin",
    )
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)

    # Define headings
    for col in columns:
        tree.heading(col, text=col)

    # Define column widths
    tree.column("Dataset", width=100, anchor='center')
    tree.column("Settlement Date", width=200, anchor='center')
    tree.column("Settlement Period", width=200, anchor='center')
    tree.column("Publish Time", width=200, anchor='center')
    tree.column("Transmission System Demand", width=200, anchor='center')   
    tree.column("National Demand", width=200, anchor='center')
    tree.column("Indicated Generation", width=200, anchor='center')
    tree.column("Indicated Demand", width=200, anchor='center')
    tree.column("Indicated Imbalance", width=200, anchor='center')
    tree.column("Indicated Margin", width=200, anchor='center')

    # Add scrollbar
    scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
    tree.configure(yscroll=scrollbar.set)

    # Pack widgets
    tree.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    # Create refresh button
    refresh_button = tk.Button(root, text="Refresh Data", command=update_table,
                            font=("Helvetica", 12), bg="#4CAF50", fg="white")
    refresh_button.pack(pady=10)

def main():
    app = ForecastDataApp()
    app.update_table()
    app.root.mainloop()

if __name__ == "__main__":
    main()