import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")

def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

if __name__ == "__main__":
    data = load_data()
    print(data.head())
