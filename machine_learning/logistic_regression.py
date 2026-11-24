import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold   # 모델 검증용

DATA_PATH = "battery_labels.csv"
HORIZON_SECONDS = 60                # 위험 구간 시간(초)

ALARM_THRESHOLD = 0.5               # 위험으로 판정 할 확률 기준
ALARM_CONSECUTIVE_SECONDS = 5       # 경보에 필요한 연속 위험 판정 시간(초)
WINDOW_SECONDS = [10, 30, 60]       # 센서 흐름을 계산할 최근 시간 구간(초)

# meas_a는 상수, indu_t는 과충전 실험에서 전부 결측이므로 제외
SENSOR_COLUMNS = [                  # 센서 열 목록(총 16개)
    "meas_v",                       # meas_a, indu_t 제거
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
    "atmp_t",
    "thermal_t",
]

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


# 실험별 최초 5단계 이상 진입 시각
stage5_time = (
    df[df["tr_stage"] >= 5]           # tr_stage가 5이상인 행 (열폭주 5, 6단계)
    .groupby("exp_id")["rel_time"]    # 실험 ID별 실험 시작 후 경과 시간
    .min()                            # 가장 작은 시간(= 처음 시작 시간)
    .rename("stage5_time")            # 데이터 이름 설정
)

print("\n실험 별 최초 5단계 이상 진입\n", stage5_time.head())
print("\n5단계 이상 진입시각이 기록된 실험 개수: ", len(stage5_time))


# 각 행에 최초 5단계 이상 진입 시각 추가
model_df = df.merge(
    stage5_time,
    on="exp_id",
    how="inner",
)

print("\n5단계 이상 진입 시각 추가 후 데이터 크기:\n", model_df.shape)

print("\n5단계 이상 진입 시각 추가 후 실험 개수:")
print(model_df["exp_id"].nunique())

print("\n5단계 이상 진입 시각이 추가된 데이터:")
print(
    model_df[
        ["exp_id", "rel_time", "tr_stage", "stage5_time"]
    ].head()
)


# 5단계 이상 진입까지 남은 시간 계산
model_df["seconds_to_stage5"] = (
    model_df["stage5_time"] - model_df["rel_time"]
)

print("\n5단계 이상 진입까지 남은 시간이 추가된 데이터:")
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

print("\n최초 5단계 이상 진입 행 개수:")
print((model_df["seconds_to_stage5"] == 0).sum())


# 5단계 이상 진입 시점과 진입 이후 데이터 제외
rows_before_filter = len(model_df)

model_df = model_df[
    model_df["seconds_to_stage5"] > 0    # 5단계 이상 진입전만 필터링 (양수 시간)
].copy()

excluded_rows = rows_before_filter - len(model_df)
print("\n5단계 이상 진입 이후 제외된 행 개수:\n", excluded_rows)

print("\n5단계 이상 진입 이후 제외 후 데이터 크기:\n", model_df.shape)

print("\n0초 이하 데이터 개수:")
print((model_df["seconds_to_stage5"] <= 0).sum())

print("\n5단계 이상 진입까지 남은 시간의 최솟값:")
print(model_df["seconds_to_stage5"].min())


# 윈도우 피처 계산을 위해 실험별 시간순으로 정렬
model_df = model_df.sort_values(
    ["exp_id", "rel_time"]
).reset_index(drop=True)

# 최근 10·30·60초의 변화량, 평균, 표준편차 생성
window_features = {}

for sensor in SENSOR_COLUMNS:
    sensor_by_experiment = model_df.groupby("exp_id")[sensor]

    # 실험 초반에 과거 값이 없을 때 사용할 실험별 첫 센서값
    first_sensor_value = sensor_by_experiment.transform("first")

    for window in WINDOW_SECONDS:
        # 현재 센서값과 window초 전 센서값의 차이
        past_sensor_value = (
            sensor_by_experiment
            .shift(window)
            .fillna(first_sensor_value)
        )
        window_features[f"{sensor}_diff{window}"] = (
            model_df[sensor] - past_sensor_value
        )

        # 현재 시점을 포함한 최근 window초 평균
        window_features[f"{sensor}_mean{window}"] = (
            sensor_by_experiment.transform(
                lambda values: values.rolling(
                    window,
                    min_periods=1,
                ).mean()
            )
        )

        # 현재 시점을 포함한 최근 window초 표준편차
        window_features[f"{sensor}_std{window}"] = (
            sensor_by_experiment.transform(
                lambda values: values.rolling(
                    window,
                    min_periods=1,
                ).std()
            ).fillna(0)
        )

# 생성한 윈도우 피처를 원본 데이터 오른쪽에 한 번에 추가
model_df = pd.concat(
    [
        model_df,
        pd.DataFrame(window_features),
    ],
    axis=1,
)

# 원본 센서 16개와 윈도우 피처 144개를 모델 입력으로 사용
FEATURE_COLUMNS = (
    SENSOR_COLUMNS
    + list(window_features.keys())
)

print("\n기본 센서 개수:", len(SENSOR_COLUMNS))
print("윈도우 피처 개수:", len(window_features))
print("전체 입력 피처 개수:", len(FEATURE_COLUMNS))



# 향후 HORIZON_SECONDS초 이내 5단계 이상 진입 여부

# 모델이 예측해야할 정답 라벨 생성
# 5단계 이상 진입 전 중 HORIZON_SECONDS 시간 내 위험 여부(1 or 0)
model_df["target"] = (
    model_df["seconds_to_stage5"] <= HORIZON_SECONDS
).astype(int)           # True/False -> 1/0 변환

print("\n정답 라벨별 데이터 개수:")  # 0, 1 개수 확인
print(model_df["target"].value_counts().sort_index())

print("\n정답별 5단계 이상 진입까지 남은 시간:")     # 0, 1 각각의 5단계까지 남은 시각
print(
    model_df.groupby("target")["seconds_to_stage5"]
    .agg(["min", "max"])
)

print("\n위험 데이터 비율:")
print(f"{model_df['target'].mean():.2%}")

print("\n정답 열 추가 후 데이터 크기:")
print(model_df.shape)


# 모델 입력, 정답, 실험 그룹 분리
X = model_df[FEATURE_COLUMNS]    # 모델 입력(원본 센서 + 윈도우 피처)
y = model_df["target"]          # 정답 라벨
groups = model_df["exp_id"]      # 실험 그룹

print("\n입력 데이터 X 크기:")
print(X.shape)

print("\n정답 데이터 y 크기:")
print(y.shape)

print("\n교차검증 그룹의 실험 개수:")
print(groups.nunique())

print("\n입력 센서 열:")
print(X.columns.tolist())

print("\n입력 센서의 결측값:")
sensor_missing_values = X.isna().sum()
print(sensor_missing_values[sensor_missing_values > 0])


# 실험 ID 단위로 학습 데이터와 테스트 데이터 분리
group_split = GroupShuffleSplit(
    n_splits=1,        # 데이터 나누기(학습/테스트)
    test_size=0.2,     # 20% 테스트용 / 나머지(80%) 학습용  
    random_state=42,    # seed
)
# 학습, 테스트의 인덱스
train_index, test_index = next(
    group_split.split(X, y, groups=groups)
)
X_train = X.iloc[train_index]   # 학습 행들 
X_test = X.iloc[test_index]     # 테스트 행들
y_train = y.iloc[train_index]   # 학습 열들
y_test = y.iloc[test_index]     # 테스트 열들

groups_train = groups.iloc[train_index]     # 학습 그룹
groups_test = groups.iloc[test_index]       # 테스트 그룹

# 분리 결과 테스트
print("\n학습 입력 데이터 크기:")   # (146558, 18)
print(X_train.shape)

print("\n테스트 입력 데이터 크기:")  # (35564, 18)
print(X_test.shape)

print("\n학습 실험 개수:")          # 75
print(groups_train.nunique())

print("\n테스트 실험 개수:")        # 19
print(groups_test.nunique())

overlap_experiments = (
    set(groups_train.unique())
    & set(groups_test.unique())
)
print("\n학습과 테스트에 중복된 실험 ID:")  # 중복 확인 -> 없음
print(overlap_experiments)


# 입력 센서 결측값 처리
imputer = SimpleImputer(strategy="median")      # 결측값=중앙값으로

X_train_imputed = pd.DataFrame(         # 학습 데이터
    imputer.fit_transform(X_train),     # 학습 데이터의 결측값 채우기(중앙값)
    columns=X_train.columns,            # " 열=센서
    index=X_train.index,                # " 행=인덱스(번호)
)

X_test_imputed = pd.DataFrame(          # 테스트 데이터
    imputer.transform(X_test),          # 테스트 데이터의 결측값 채우기(중앙값)
    columns=X_test.columns,             # " 열=센서
    index=X_test.index,                 # " 행=인덱스(번호)
)

print("\n결측값 처리 후 학습 데이터의 결측값 확인:")
print(X_train_imputed.isna().sum().sum())

print("\n결측값 처리 후 테스트 데이터의 결측값 확인:")
print(X_test_imputed.isna().sum().sum())


# 센서 데이터 표준화
# 센서별 평균을 0, 표준편차를 1에 가깝게 맞춤
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imputed),  # 학습 데이터 기준 계산, 표준화
    columns=X_train_imputed.columns,
    index=X_train_imputed.index,
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imputed),       # 학습 데이터 기준 테스트 데이터 표준화
    columns=X_test_imputed.columns,
    index=X_test_imputed.index,
)

print("\n표준화 후 학습 데이터 전체 평균:")
print(f"{X_train_scaled.to_numpy().mean():.6f}")

print("\n표준화 후 학습 데이터 전체 표준편차:")
print(f"{X_train_scaled.to_numpy().std():.6f}")


# 로지스틱 회귀 모델 생성/학습
logistic_model = LogisticRegression(
    max_iter=1000,          # 최대 반복 횟수
)
# 로지스틱 학습
logistic_model.fit(
    X_train_scaled,         # 표준화한 센서 데이터
    y_train,                # 정답 라벨
)

print("\n로지스틱 회귀 모델 학습 완료")
print("모델 학습 반복 횟수:", logistic_model.n_iter_[0])


# 테스트 데이터 위험 여부 확인, 위험 확률 예측
# 예측 형태  [[정상(0) 확률], [위험(1) 확률]
#            [정상(0) 확률], [위험(1) 확률]...]
y_pred = logistic_model.predict(X_test_scaled)      # 예측 (정상=0, 위험=1)

# 위험O인 열의 구체적 확률
y_risk_probability = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

print("\n테스트 데이터 예측 완료")
print("전체 예측 개수:", len(y_pred))
print("위험 확률 개수:", len(y_risk_probability))
print("위험으로 예측한 개수:", y_pred.sum())


# 테스트 데이터 정확도, Loss 평가
# 정확도 = 맞게 예측한 개수 / 전체 테스트 데이터 개수
test_accuracy = accuracy_score(     # 정확도 계산(예측값과 정답값 비교)
    y_test,         # 실제 정답
    y_pred,         # 모델 예측값
)

test_log_loss = log_loss(   # loss 계산 
    y_test,                 # 실제 정답
    y_risk_probability,     # 모델 위험 확률
)

print("\n로지스틱 회귀 테스트 성능")
print(f"정확도: {test_accuracy:.4f}")        # 97.5%
print(f"Log Loss: {test_log_loss:.4f}")     # 0.0856



# 가열·과충전 비율을 유지하는 실험 ID 단위 5분할 교차검증
stratified_group_kfold = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# 각 행이 가열 실험인지 과충전 실험인지 나타내는 값
stratify_labels = model_df.loc[
    X.index,
    "tr_method",
]

cv_accuracies = []  # 5회차 정확도
cv_log_losses = []  # 5회차 Log Loss


# 각 행이 검증 데이터였을 때 예측된 위험 확률 저장
cv_proba = pd.Series(
    index=X.index,
    dtype=float,
)


# 5개 분할한 대로 회귀 돌리기
for fold_number, (fold_train_index, fold_validation_index) in enumerate(
    stratified_group_kfold.split(
        X,
        stratify_labels,
        groups=groups,
    ),
    start=1,
):
    # 현재 회차의 학습 데이터와 검증 데이터 분리
    X_fold_train = X.iloc[fold_train_index]
    X_fold_validation = X.iloc[fold_validation_index]

    y_fold_train = y.iloc[fold_train_index]
    y_fold_validation = y.iloc[fold_validation_index]

    # 현재 회차의 학습 데이터 기준으로 결측값 처리
    fold_imputer = SimpleImputer(
        strategy="median",
    )

    X_fold_train_imputed = fold_imputer.fit_transform(
        X_fold_train,
    )

    X_fold_validation_imputed = fold_imputer.transform(
        X_fold_validation,
    )

    # 현재 회차의 학습 데이터 기준으로 표준화
    fold_scaler = StandardScaler()

    X_fold_train_scaled = fold_scaler.fit_transform(
        X_fold_train_imputed,
    )

    X_fold_validation_scaled = fold_scaler.transform(
        X_fold_validation_imputed,
    )

    # 현재 회차의 로지스틱 회귀 모델 생성 및 학습
    fold_model = LogisticRegression(
        max_iter=1000,
    )

    fold_model.fit(
        X_fold_train_scaled,
        y_fold_train,
    )

    # 현재 회차의 검증 데이터 예측
    fold_pred = fold_model.predict(
        X_fold_validation_scaled,
    )

    fold_risk_probability = fold_model.predict_proba(
        X_fold_validation_scaled,
    )[:, 1]

    # 현재 검증 회차에서 계산한 확률을 원래 행 위치에 저장
    cv_proba.iloc[
        fold_validation_index
    ] = fold_risk_probability

    # 현재 회차의 정확도와 Loss 계산
    fold_accuracy = accuracy_score(
        y_fold_validation,
        fold_pred,
    )

    fold_log_loss = log_loss(
        y_fold_validation,
        fold_risk_probability,
    )

    cv_accuracies.append(fold_accuracy)
    cv_log_losses.append(fold_log_loss)

    # 같은 실험이 학습과 검증에 중복됐는지 확인
    fold_train_groups = set(
        groups.iloc[fold_train_index].unique()
    )

    fold_validation_groups = set(
        groups.iloc[fold_validation_index].unique()
    )

    overlap_groups = (
        fold_train_groups
        & fold_validation_groups
    )

    print(
        f"{fold_number}회차 - "
        f"정확도: {fold_accuracy:.4f}, "
        f"Log Loss: {fold_log_loss:.4f}, "
        f"중복 실험: {len(overlap_groups)}개"
    )

# 모든 행에 교차검증 예측 확률이 저장됐는지 확인
if cv_proba.isna().any():
    raise ValueError("교차검증 예측 확률이 없는 행이 있습니다.")

print("\n교차검증 위험 확률 저장 개수:")
print(cv_proba.notna().sum())


# 평균과 표준편차 계산을 위해 Series로 변환
cv_accuracy_series = pd.Series(cv_accuracies)
cv_log_loss_series = pd.Series(cv_log_losses)

print("\n로지스틱 회귀 5분할 교차검증 결과")
print(f"평균 정확도: {cv_accuracy_series.mean():.4f}")
print(
    "정확도 표준편차: "
    f"{cv_accuracy_series.std(ddof=1):.4f}"
)
print(f"평균 Log Loss: {cv_log_loss_series.mean():.4f}")
print(
    "Log Loss 표준편차: "
    f"{cv_log_loss_series.std(ddof=1):.4f}"
)

# 경보 계산에 필요한 실험 ID, 시간, 정답과 예측 확률을 하나로 정리
alarm_df = model_df.loc[
    X.index,
    [
        "exp_id",       # 실험 ID
        "rel_time",     # 실험 경과 시간
        "stage5_time",  # 최초 5단계 이상 진입 시각
        "target",       # 실제 정답
    ],
].copy()

alarm_df["proba"] = cv_proba

# 실험별 시간 순서로 정렬
alarm_df = alarm_df.sort_values(
    ["exp_id", "rel_time"]
).reset_index(drop=True)

# 위험 확률이 기준값 이상이면 1, 아니면 0
alarm_df["risk_prediction"] = (
    alarm_df["proba"]
    .ge(ALARM_THRESHOLD)
    .astype(int)
)

# 정상 판정이 나올 때마다 새로운 연속 구간 시작
streak_group = (
    alarm_df["risk_prediction"]
    .eq(0)
    .groupby(alarm_df["exp_id"])
    .cumsum()
)

# 실험별로 위험 판정이 몇 초째 연속되는지 계산
alarm_df["consecutive_risk_count"] = (
    alarm_df["risk_prediction"]
    .groupby(
        [
            alarm_df["exp_id"],
            streak_group,
        ]
    )
    .cumsum()
)

# ALARM_CONSECUTIVE_SECONDS 이상 위험 판정이 연속되면 최종 경보
alarm_df["alarm"] = (
    alarm_df["consecutive_risk_count"]
    >= ALARM_CONSECUTIVE_SECONDS
)

print("\n경보 생성 결과")
print(
    f"경보 조건: 위험 확률 {ALARM_THRESHOLD} 이상 "
    f"{ALARM_CONSECUTIVE_SECONDS}초 연속"
)
print(
    "위험으로 판정된 행:",
    alarm_df["risk_prediction"].sum(),
)
print(
    "최종 경보가 켜진 행:",
    alarm_df["alarm"].sum(),
)
print(
    "경보가 발생한 실험:",
    alarm_df.loc[
        alarm_df["alarm"],
        "exp_id",
    ].nunique(),
)

# 최종 경보를 기준으로 분류 성능 계산
# [Recall], [Precision], [f1-score]
recall = recall_score(
    alarm_df["target"],
    alarm_df["alarm"],
    zero_division=0,
)

precision = precision_score(
    alarm_df["target"],
    alarm_df["alarm"],
    zero_division=0,
)

f1 = f1_score(
    alarm_df["target"],
    alarm_df["alarm"],
    zero_division=0,
)


# 실험별 사전 탐지 성공률, 경고 선행 시간, 오경보 횟수
exp_results = []

for exp_id, experiment in alarm_df.groupby("exp_id"):
    # 실제 위험 구간에서 발생한 경보만 선택
    risk_alarm = (
        experiment["alarm"]
        & experiment["target"].eq(1)
    )

    # 정상 구간에서 발생한 경보만 선택
    normal_alarm = (
        experiment["alarm"]
        & experiment["target"].eq(0)
    ).astype(int)

    # 경보가 0에서 1로 바뀌는 시점을 오경보 시작으로 계산
    false_alarm_starts = (
        normal_alarm.eq(1)
        & normal_alarm.shift(fill_value=0).eq(0)
    )

    # 위험 구간에 경보가 하나라도 있으면 탐지 성공
    detection_success = risk_alarm.any()
    
    # 위험 구간에서 처음 경보가 발생한 시각
    first_alarm_time = experiment.loc[
        risk_alarm,
        "rel_time",
    ].min()

    # 탐지에 성공한 경우에만 경고 선행시간 계산
    lead_time = (
        experiment["stage5_time"].iloc[0]
        - first_alarm_time
        if detection_success
        else float("nan")
    )

    exp_results.append(
        {
            "exp_id": exp_id,               # 실험별 id
            "success": detection_success,   # 탐지 성공 여부
            "lead_time": lead_time,         # 경고 선행시간 계산
            "false_alarms": int(false_alarm_starts.sum()),  # 오경보 횟수
        }
    )

# 실험별 결과를 DataFrame으로 변환
exp_results = pd.DataFrame(exp_results)

# [실험별 사전 탐지 성공률] 구현
# 5단계 이상 진입 전 60초 위험 구간에서 경보가 한 번이라도 발생했으면 해당 실험 탐지 성공
early_detection_success_count = int(
    exp_results["success"].sum()
)
early_detection_total_count = len(exp_results)

early_detection_success_rate = (
    early_detection_success_count
    / early_detection_total_count
)


# [경고 선행 시간] 계산
# 경고 선행시간 = 최초 5단계 이상 진입 시각 - 최초 유효 경보 시각
successful_lead_times = exp_results.loc[
    exp_results["success"],
    "lead_time",
]

median_lead_time = successful_lead_times.median()


# [오경보 횟수] 계산
# 60초 위험 구간보다 앞선 정상 구간에서 경보가 발생한 경우
# 전체 실험에서 발생한 오경보 횟수
false_alarm_count = int(
    exp_results["false_alarms"].sum()
)

# 오경보가 한 번 이상 발생한 실험 개수
false_alarm_experiment_count = int(
    exp_results["false_alarms"].gt(0).sum()
)


# 결과 요약
print("\n\n")
print("※ 로지스틱 회귀 결과 요약")
print(f"위험 구간 기준: 5단계 이상 진입 전 {HORIZON_SECONDS}초")
print(f"원본 데이터: {len(df):,}행, {df['exp_id'].nunique()}개 실험")
print(f"모델 데이터: {len(X):,}행, {groups.nunique()}개 실험")
print(f"입력 센서: {len(SENSOR_COLUMNS)}개")
print(f"전체 입력 피처: {len(FEATURE_COLUMNS)}개")
print(f"위험 데이터 비율: {y.mean():.2%}")
print("-" * 50)
print(f"단일 분할 정확도: {test_accuracy:.4f}")
print(f"단일 분할 Log Loss: {test_log_loss:.4f}")
print(f"교차검증 평균 정확도: {cv_accuracy_series.mean():.4f}")
print(f"교차검증 정확도 표준편차: {cv_accuracy_series.std(ddof=1):.4f}")
print(f"교차검증 평균 Log Loss: {cv_log_loss_series.mean():.4f}")
print(f"교차검증 Log Loss 표준편차: {cv_log_loss_series.std(ddof=1):.4f}")
print("-" * 50)
print("\n경보 기준 분류 성능")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")
print("-" * 50)
print("실험별 사전 탐지 성공률")
print(
    "사전 탐지 성공: "
    f"{early_detection_success_count}/"
    f"{early_detection_total_count}개 실험"
)
print(
    "사전 탐지 성공률: "
    f"{early_detection_success_rate:.2%}"
)
print("\n경고 선행시간")
print(
    "경고 선행시간 중앙값: "
    f"{median_lead_time:.1f}초"
)
print("\n오경보 성능")
print(f"오경보 횟수: {false_alarm_count}회")
print(
    "오경보가 발생한 실험: "
    f"{false_alarm_experiment_count}/"
    f"{early_detection_total_count}개"
)
