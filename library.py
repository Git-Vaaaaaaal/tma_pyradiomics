import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



def normalize_mean(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = ["image", "new_patient_id", "old_patient_id", "status"]

    df = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_columns]
    scaler = StandardScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    return df

def normalize_minmax(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = ["image", "new_patient_id", "old_patient_id", "status"]

    df = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_columns]

    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    return df