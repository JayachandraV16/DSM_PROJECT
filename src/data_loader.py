import pandas as pd

def load_data(path):
    df = pd.read_csv(r'C:\Users\vanda\OneDrive\Desktop\DSM_PROJECT\data\data.csv')
    return df