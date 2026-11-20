import pandas as pd

DATA_PATH = "battery_labels.csv"

df = pd.read_csv(DATA_PATH)

# 데이터 확인
print("데이터 크기:", df.shape)     # (218389, 29)

print("\n실험 개수:", df["exp_id"].nunique())   # 97

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
    .rename("stage5_time")            # 데이터 이름 설정
)

print("\n실험 별 최초 5단계 진입\n", stage5_time.head())
print("\n5단계가 기록된 실험 개수: ", len(stage5_time))


# 5단계 없는 실험
all_experiments = set(df["exp_id"].unique())   # 중복 제거한 전체 실험
stage5_experiments = set(stage5_time.index)    # 5단계 실험ID
excluded_experiments = sorted(                  # 5단계가 없는 부분 선정
    all_experiments - stage5_experiments
)

print("\n5단계가 없는 실험 개수:")
print(len(excluded_experiments))  # 3개

print("\n5단계가 없는 실험 ID:")    # 20250820003, 20250829002, 20250829004
print(excluded_experiments)


# 각 행에 최초 5단계 진입 시각 추가
model_df = df.merge(
    stage5_time,
    on="exp_id",
    how="inner",
)

print("\n5단계 진입 시각 추가 후 데이터 크기:\n", model_df.shape)

print("\n5단계 진입 시각 추가 후 실험 개수:")
print(model_df["exp_id"].nunique())

print("\n5단계 진입 시각이 추가된 데이터:")
print(
    model_df[
        ["exp_id", "rel_time", "tr_stage", "stage5_time"]
    ].head()
)


# 5단계 진입까지 남은 시간 계산
model_df["seconds_to_stage5"] = (
    model_df["stage5_time"] - model_df["rel_time"]
)

print("\n5단계 진입까지 남은 시간이 추가된 데이터:")
print(
    model_df[
        [
            "exp_id",
            "rel_time",
            "tr_stage",
            "stage5_time",
            "seconds_to_stage5",
        ]
    ].head()
)

print("\n최초 5단계 진입 행 개수:")
print((model_df["seconds_to_stage5"] == 0).sum())
