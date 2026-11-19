import pandas as pd

DATA_PATH = "battery_labels.csv"

df = pd.read_csv(DATA_PATH)

# 데이터 확인
print("데이터 크기:")
print(df.shape)

print("\n실험 개수:")
print(df["exp_id"].nunique())

print("\n열폭주 단계별 데이터 개수:")
print(df["tr_stage"].value_counts().sort_index())

print("\n결측값이 있는 열:")
missing_values = df.isna().sum()    # 결측값 = NaN 또는 빈값
print(missing_values[missing_values > 0])

print("\n데이터 일부:")
print(df.head())


# 실험별 최초 5단계 진입 시각
stage5_time = (
    df[df["tr_stage"] == 5]           # tr_stage가 5인 행 (열폭주 5단계)
    .groupby("exp_id")["rel_time"]    # 실험 ID별 실험 시작 후 경과 시간
    .min()                            # 가장 작은 시간(= 처음 시작 시간)
    .rename("stage5_time")            # 데이터 이름 설정a
)

print("\n실험 별 최초 5단계 진입\n", stage5_time.head())
print("\n5단계가 기록된 실험 개수: ", len(stage5_time))

