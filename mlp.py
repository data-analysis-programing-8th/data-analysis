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
