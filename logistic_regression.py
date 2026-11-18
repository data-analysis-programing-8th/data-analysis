import pandas as pd

DATA_PATH = "battery_labels.csv"

df = pd.read_csv(DATA_PATH)

print("데이터 크기:")
print(df.shape)

print("\n실험 개수:")
print(df["exp_id"].nunique())

print("\n열폭주 단계별 데이터 개수:")
print(df["tr_stage"].value_counts().sort_index())

print("\n결측값이 있는 열:")
missing_values = df.isna().sum()
print(missing_values[missing_values > 0])

print("\n데이터 일부:")
print(df.head())