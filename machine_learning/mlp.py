import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


DATA_PATH = "battery_labels.csv"
HORIZON_SECONDS = 60            # 위험 구간 시간(초)
WINDOW_SECONDS = [10, 30, 60]   # 센서 흐름을 계산할 최근 시간 구간(초)

# 최적의 성능을 나타낼 MLP 설정 (직접 여러번 비교)
# 현재 단일 분할에서 성능이 가장 좋았던 설정
HIDDEN_LAYER_SIZES = (256, )
LEARNING_RATE_INIT = 0.001

# 위험 확률이 0.5 이상인 상태가 5초 연속이면 경보를 발생시킴
ALARM_THRESHOLD = 0.5
ALARM_CONSECUTIVE_SECONDS = 5


# meas_a는 모든 값이 0이고, indu_t는 과충전 실험에서 모두 결측이므로 제외
SENSOR_COLUMNS = [       # 센서 열 목록(총 16개)
    "meas_v",
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

# 전처리된 배터리 데이터 불러오기
df = pd.read_csv(DATA_PATH)


# 실험별 최초 5단계 이상 진입 시각 계산
# 5단계 기록 없이 바로 6단계로 넘어간 실험도 포함
stage5_time = (
    df[df["tr_stage"] >= 5]               # 5단계 이상 행 선택
    .groupby("exp_id")["rel_time"]        # 실험 ID별로 묶기
    .min()
    .rename("stage5_time")
)

# 5단계 이상 진입 시각이 있는 실험만 남기기 + 진입 시각 추가
model_df = df.merge(
    stage5_time,
    on="exp_id",
    how="inner",
)

# 각 측정 시점에서 5단계 이상 진입까지 남은 시간 계산
model_df["seconds_to_stage5"] = (
    model_df["stage5_time"] - model_df["rel_time"]
)

# 최초 5단계 이상 진입 전 데이터만 사용
model_df = model_df[
    model_df["seconds_to_stage5"] > 0       # 5단계 이상 진입 이전만 필터링
].copy()

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
FEATURE_COLUMNS = SENSOR_COLUMNS + list(window_features.keys())


# 향후 60초 이내에 5단계 이상으로 진입하는지 정답 라벨 생성
# 5단계 이상 진입 전 중 HORIZON_SECONDS 시간 내 위험 여부(1 or 0)
model_df["target"] = (
    model_df["seconds_to_stage5"] <= HORIZON_SECONDS
).astype(int)

# 모델 입력, 정답, 실험 그룹 분리
X = model_df[FEATURE_COLUMNS]
y = model_df["target"]
groups = model_df["exp_id"]

# 모든 행을 정상(0)으로 예측했을 때의 비교 기준 정확도
always_normal_accuracy = 1 - y.mean()

# 실험 ID 단위로 학습 데이터와 테스트 데이터 분리
group_split = GroupShuffleSplit(
    n_splits=1,         # 단일 분할
    test_size=0.2,      # 테스트 20%, 학습용 80%
    random_state=42,
)

train_index, test_index = next(
    group_split.split(
        X,
        y,
        groups=groups,
    )
)

X_train = X.iloc[train_index]   # 학습용 입력 데이터
X_test = X.iloc[test_index]     # 테스트용 입력 데이터
y_train = y.iloc[train_index]   # 학습 정답
y_test = y.iloc[test_index]     # 테스트 정답

# 센서 데이터 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # 센서별 평균과 표준편차 계산 후 표준화
X_test_scaled = scaler.transform(X_test)        # 학습 데이터에서 계산한 동일한 기준으로 테스트 데이터 표준화

# MLP 분류 모델 생성
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,        # hidden layer
    activation="relu",      # relu 활성화 함수 적용
    max_iter=1000,          # 최대 1000번 반복
    early_stopping=True,
    alpha=0.1,
    learning_rate_init=LEARNING_RATE_INIT,       # learning_rate - 보폭
    random_state=42,
)

# 표준화된 학습 데이터로 모델 학습
train_sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

mlp_classifier.fit(
    X_train_scaled,
    y_train,
    sample_weight=train_sample_weights,
)

# 테스트 데이터의 위험 여부와 위험 확률 예측
y_pred = mlp_classifier.predict(
    X_test_scaled
)

y_risk_probability = mlp_classifier.predict_proba(
    X_test_scaled
)[:, 1]

# 테스트 데이터 정확도(Accuracy)와 손실값(Log Loss) 계산
test_accuracy = accuracy_score(
    y_test,
    y_pred,
)

test_log_loss = log_loss(
    y_test,
    y_risk_probability,
)

print("\nMLP 성능 테스트")
print("은닉층 구조:", mlp_classifier.hidden_layer_sizes)
print("초기 학습률:", mlp_classifier.learning_rate_init)
print(f"정확도: {test_accuracy:.4f}")
print(f"Log Loss: {test_log_loss:.4f}\n")


# 같은 exp_id를 묶고 tr_method 비율도 고르게 분할
stratified_group_kfold = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

stratify_labels = model_df.loc[
    X.index,
    "tr_method",
]

cv_accuracies = []  # 각각의 정확도
cv_log_losses = []  # 각각의 손실값
cv_pr_aucs = []     # 각 회차의 PR-AUC

# 각 행이 검증 데이터일 때 예측된 위험 확률
# proba는 probability(확률)의 줄임말
# 이후 경보 및 조기경보 지표는 학습 데이터 예측이 아닌 이 확률로 계산함
cv_proba = pd.Series(
    index=X.index,
    dtype=float,
)

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

    # 학습 데이터 기준으로 표준화
    fold_scaler = StandardScaler()

    X_fold_train_scaled = fold_scaler.fit_transform(
        X_fold_train
    )
    X_fold_validation_scaled = fold_scaler.transform(
        X_fold_validation
    )

    # 현재 설정으로 MLP 생성 및 학습
    fold_model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        max_iter=1000,
        early_stopping=True,
        alpha=0.1,
        learning_rate_init=LEARNING_RATE_INIT,
        random_state=42,
    )

    # 현재 회차의 학습 데이터 비율로 위험 행에 더 큰 가중치 적용
    fold_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_fold_train,
    )

    fold_model.fit(
        X_fold_train_scaled,
        y_fold_train,
        sample_weight=fold_sample_weights,
    )

    # 검증 데이터 예측
    fold_pred = fold_model.predict(
        X_fold_validation_scaled
    )
    fold_risk_probability = fold_model.predict_proba(
        X_fold_validation_scaled
    )[:, 1]

    # 현재 검증 회차에 해당하는 원래 행 위치에 확률 저장
    cv_proba.iloc[
        fold_validation_index
    ] = fold_risk_probability

    # 현재 회차의 정확도와 Log Loss 계산
    fold_accuracy = accuracy_score(
        y_fold_validation,
        fold_pred,
    )
    fold_log_loss = log_loss(
        y_fold_validation,
        fold_risk_probability,
    )
    fold_pr_auc = average_precision_score(
        y_fold_validation,
        fold_risk_probability,
    )

    cv_accuracies.append(fold_accuracy)
    cv_log_losses.append(fold_log_loss)
    cv_pr_aucs.append(fold_pr_auc)

    # 같은 실험이 학습과 검증에 중복됐는지 확인
    fold_train_groups = set(
        groups.iloc[fold_train_index].unique()
    )
    fold_validation_groups = set(
        groups.iloc[fold_validation_index].unique()
    )
    overlap_groups = fold_train_groups & fold_validation_groups

    if overlap_groups:
        raise ValueError("학습과 검증에 같은 실험 ID가 포함됐습니다.")

    print(
        f"{fold_number}회차 - "
        f"정확도: {fold_accuracy:.4f}, "
        f"Log Loss: {fold_log_loss:.4f}, "
        f"PR-AUC: {fold_pr_auc:.4f}, "
        f"중복 실험: {len(overlap_groups)}개"
    )


# 교차검증 결과의 평균과 표준편차 계산
cv_accuracy_series = pd.Series(cv_accuracies)
cv_log_loss_series = pd.Series(cv_log_losses)
cv_pr_auc_series = pd.Series(cv_pr_aucs)

# 교차검증 확률이 모든 행에 저장됐는지 확인
if cv_proba.isna().any():
    raise ValueError("교차검증 예측 확률이 없는 행이 있습니다.")


# 5초 연속 위험 경보 생성
alarm_df = model_df.loc[
    X.index,
    [
        "exp_id",
        "rel_time",
        "stage5_time",
        "target",
        "tr_method",
    ],
].copy()
alarm_df["proba"] = cv_proba
alarm_df = alarm_df.sort_values(
    ["exp_id", "rel_time"]
).reset_index(drop=True)

# ge(): 확률이 기준값 이상인지 비교하여 True/False 반환
alarm_df["risk_prediction"] = (
    alarm_df["proba"].ge(ALARM_THRESHOLD).astype(int)
)

# eq(0)과 cumsum(): 정상 판정이 나올 때마다 연속 위험 구간 번호를 변경
streak_group = (
    alarm_df["risk_prediction"]
    .eq(0)
    .groupby(alarm_df["exp_id"])
    .cumsum()
)

# groupby().cumsum(): 실험별 연속 위험 판정 횟수 계산
alarm_df["consecutive_risk_count"] = (
    alarm_df["risk_prediction"]
    .groupby([alarm_df["exp_id"], streak_group])
    .cumsum()
)
alarm_df["alarm"] = (
    alarm_df["consecutive_risk_count"]
    >= ALARM_CONSECUTIVE_SECONDS
)


# 경보 기준 Recall, Precision, F1 계산
# 각 함수는 실제 정답 target과 최종 경보 alarm을 비교함
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

# 가열·과충전 방식별 분류 성능 계산
classification_by_method = {}

for method_value, method_name in ((1, "가열"), (2, "과충전")):
    method_rows = alarm_df["tr_method"].eq(method_value)

    classification_by_method[method_name] = {
        "recall": recall_score(
            alarm_df.loc[method_rows, "target"],
            alarm_df.loc[method_rows, "alarm"],
            zero_division=0,
        ),
        "precision": precision_score(
            alarm_df.loc[method_rows, "target"],
            alarm_df.loc[method_rows, "alarm"],
            zero_division=0,
        ),
        "f1": f1_score(
            alarm_df.loc[method_rows, "target"],
            alarm_df.loc[method_rows, "alarm"],
            zero_division=0,
        ),
    }

# 실험별 사전 탐지 성공, 선행시간, 오경보 횟수 계산
exp_results = []

for exp_id, experiment in alarm_df.groupby("exp_id"):
    risk_alarm = (
        experiment["alarm"]
        & experiment["target"].eq(1)
    )
    normal_alarm = (
        experiment["alarm"]
        & experiment["target"].eq(0)
    ).astype(int)

    # shift(fill_value=0): 직전 행과 비교해 오경보가 0에서 1로 시작한 순간 탐색
    false_alarm_starts = (
        normal_alarm.eq(1)
        & normal_alarm.shift(fill_value=0).eq(0)
    )

    detection_success = risk_alarm.any()
    first_alarm_time = experiment.loc[
        risk_alarm,
        "rel_time",
    ].min()

    lead_time = (
        experiment["stage5_time"].iloc[0] - first_alarm_time
        if detection_success
        else float("nan")
    )

    exp_results.append(
        {
            "exp_id": exp_id,
            "tr_method": experiment["tr_method"].iloc[0],
            "success": detection_success,
            "lead_time": lead_time,
            "false_alarms": int(false_alarm_starts.sum()),
        }
    )

exp_results = pd.DataFrame(exp_results)

early_detection_success_count = int(
    exp_results["success"].sum()
)
early_detection_total_count = len(exp_results)
median_lead_time = exp_results.loc[
    exp_results["success"],
    "lead_time",
].median()
false_alarm_count = int(
    exp_results["false_alarms"].sum()
)
false_alarm_experiment_count = int(
    exp_results["false_alarms"].gt(0).sum()
)

# 가열·과충전 방식별 조기경보 성능 계산
early_warning_by_method = {}

for method_value, method_name in ((1, "가열"), (2, "과충전")):
    method_results = exp_results[
        exp_results["tr_method"].eq(method_value)
    ]

    early_warning_by_method[method_name] = {
        "success_count": int(method_results["success"].sum()),
        "total_count": len(method_results),
        "median_lead_time": method_results.loc[
            method_results["success"],
            "lead_time",
        ].median(),
        "false_alarm_count": int(
            method_results["false_alarms"].sum()
        ),
        "false_alarm_experiment_count": int(
            method_results["false_alarms"].gt(0).sum()
        ),
    }

# 최종 결과 요약
print("\n\n※ MLP 결과 요약")
print(f"위험 구간 기준: 5단계 이상 진입 전 {HORIZON_SECONDS}초")
print(f"은닉층 구조: {HIDDEN_LAYER_SIZES}")
print(f"활성화 함수: {mlp_classifier.activation}")
print(f"초기 learning_rate: {LEARNING_RATE_INIT}")
print(f"원본 데이터: {len(df):,}행, {df['exp_id'].nunique()}개 실험")
print(f"모델 데이터: {len(X):,}행, {groups.nunique()}개 실험")
print(f"입력 센서: {len(SENSOR_COLUMNS)}개")

print(f"전체 입력 피처: {len(FEATURE_COLUMNS)}개")  # 원본 16개 + 윈도우 피처 144개
print(f"위험 데이터 비율: {y.mean():.2%}")
print("-" * 50)
print(f"항상 정상 기준 정확도: {always_normal_accuracy:.4f}")
print(f"단일 분할 정확도: {test_accuracy:.4f}")
print(f"단일 분할 Log Loss: {test_log_loss:.4f}")
print(
    "교차검증 평균 정확도: "
    f"{cv_accuracy_series.mean():.4f}"
)
print(
    "교차검증 정확도 표준편차: "
    f"{cv_accuracy_series.std(ddof=1):.4f}"
)
print(
    "교차검증 평균 Log Loss: "
    f"{cv_log_loss_series.mean():.4f}"
)
print(
    "교차검증 Log Loss 표준편차: "
    f"{cv_log_loss_series.std(ddof=1):.4f}"
)
print(
    "교차검증 평균 PR-AUC: "
    f"{cv_pr_auc_series.mean():.4f}"
)
print(
    "교차검증 PR-AUC 표준편차: "
    f"{cv_pr_auc_series.std(ddof=1):.4f}"
)
print(f"경보 기준 Recall: {recall:.4f}")
print(f"경보 기준 Precision: {precision:.4f}")
print(f"경보 기준 F1-score: {f1:.4f}")
for method_name, method_metrics in classification_by_method.items():
    print(
        f"{method_name}: "
        f"Recall {method_metrics['recall']:.4f}, "
        f"Precision {method_metrics['precision']:.4f}, "
        f"F1-score {method_metrics['f1']:.4f}"
    )
print(
    "사전 탐지 성공: "
    f"{early_detection_success_count}/"
    f"{early_detection_total_count}개 실험"
)
print(f"경고 선행시간 중앙값: {median_lead_time:.1f}초")
print(f"오경보 횟수: {false_alarm_count}회")
print(
    "오경보가 발생한 실험: "
    f"{false_alarm_experiment_count}/"
    f"{early_detection_total_count}개"
)

print("\n실험 방식별 조기경보 성능")
for method_name, method_metrics in early_warning_by_method.items():
    print(
        f"{method_name}: "
        f"사전 탐지 {method_metrics['success_count']}/"
        f"{method_metrics['total_count']}개, "
        f"선행시간 중앙값 "
        f"{method_metrics['median_lead_time']:.1f}초, "
        f"오경보 {method_metrics['false_alarm_count']}회, "
        f"오경보 실험 "
        f"{method_metrics['false_alarm_experiment_count']}/"
        f"{method_metrics['total_count']}개"
    )
