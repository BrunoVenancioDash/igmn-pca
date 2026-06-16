from scipy.io import arff
import pandas as pd

def read_arff(set_name):
    data, meta = arff.loadarff(set_name)
    df = pd.DataFrame(data)
    # Decodifica colunas de bytes (strings) para str
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.decode('utf-8')
    return df