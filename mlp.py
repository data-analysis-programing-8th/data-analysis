import pandas as pd


DATA_PATH = "battery_labels.csv"
HORIZON_SECONDS = 60     # 위험 구간 시간(초)

SENSOR_COLUMNS = [       # 센서 열 목록(총 18개)
    "meas_v",
    "meas_a",
    "meas_t1",
    "meas_t2",
    "meas_p",
    "meas_ch4",
    "meas_co",
    "meas_co2",
    "meas_hcn",
    "meas_hcl",
    "meas_hf",
    "meas_n2o",
    "meas_no",
    "meas_no2",
    "meas_so2",
    "indu_t",
    "atmp_t",
    "thermal_t",
]

# 전처리된 배터리 데이터 불러오기
df = pd.read_csv(DATA_PATH)


# 실험별 최초 5단계 진입 시각 계산
stage5_time = (
    df[df["tr_stage"] == 5]               # 5단계 행만 선택
    .groupby("exp_id")["rel_time"]        # 실험 ID별로 묶기
    .min()
    .rename("stage5_time")
)

print("\n실험별 최초 5단계 진입 시각:")
print(stage5_time.head())

print("\n5단계가 기록된 실험 개수:")
print(len(stage5_time))


# 5단계 기록이 있는 실험만 남기기 + 진입 시각 추가
model_df = df.merge(
    stage5_time,
    on="exp_id",
    how="inner",
)

print("\n모델 데이터의 실험 개수:")
print(model_df["exp_id"].nunique())


# 각 측정 시점에서 5단계 진입까지 남은 시간 계산
model_df["seconds_to_stage5"] = (
    model_df["stage5_time"] - model_df["rel_time"]
)

print("\n5단계 진입까지 남은 시간이 추가된 데이터:")
print(
    model_df[
        [
            "exp_id",       # 실험id
            "rel_time",     # 실험 경과 시간
            "tr_stage",     # 현재 열폭주 단계
            "stage5_time",  # 5단계 시간
            "seconds_to_stage5",    # 5단계까지 남은 시간
        ]
    ].head()
)


# 최초 5단계 진입 전 데이터만 사용
model_df = model_df[
    model_df["seconds_to_stage5"] > 0       # 5단계 이전만 필터링
].copy()

print("\n5단계 진입 전 모델 데이터 개수:")
print(len(model_df))

print("\n5단계 진입 시점 이후 데이터 개수:")
print((model_df["seconds_to_stage5"] <= 0).sum())
