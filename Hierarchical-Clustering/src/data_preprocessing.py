import os
import pandas as pd

def load_data():
    # 1. Get the absolute path of the directory where THIS script lives (the 'src' folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up one level to 'K-mean-Clusttering' and then look inside the 'data' folder
    filepath = os.path.join(current_dir, '..', 'data', 'dataset.csv')
    
    # 3. Clean up the path format safely
    filepath = os.path.abspath(filepath)
    
    # 4. Read the CSV file
    df = pd.read_csv(filepath)
    return df
