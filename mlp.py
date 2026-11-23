import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

from sklearn.model_selection import GroupKFold  # 교차 검증


DATA_PATH = "battery_labels.csv"
HORIZON_SECONDS = 60     # 위험 구간 시간(초)

# 최적의 성능을 나타낼 MLP 설정 (직접 여러번 비교)
# 현재 단일 분할에서 성능이 가장 좋았던 설정
HIDDEN_LAYER_SIZES = (64, 64)
LEARNING_RATE_INIT = 0.001


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


# 향후 60초 이내에 5단계로 진입하는지 정답 라벨 생성
# 5단계 진입 전 중 HORIZON_SECONDS 시간 내 위험 여부(1 or 0)
model_df["target"] = (
    model_df["seconds_to_stage5"] <= HORIZON_SECONDS
).astype(int)

print("\n정답 라벨별 데이터 개수:")
print(model_df["target"].value_counts().sort_index())


# 모델 입력, 정답, 실험 그룹 분리
X = model_df[SENSOR_COLUMNS]
y = model_df["target"]
groups = model_df["exp_id"]

print("\n입력 데이터 X 크기:")
print(X.shape)

print("\n정답 데이터 y 크기:")
print(y.shape)

print("\n실험 그룹 개수:")
print(groups.nunique())


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

print("\n학습/테스트 입력 데이터 크기:")
print(X_train.shape, X_test.shape)


# 학습 데이터의 중앙값으로 결측값 처리
imputer = SimpleImputer(
    strategy="median",
)

X_train_imputed = imputer.fit_transform(X_train)    # 학습 데이터의 중앙값 계산 및 결측값 처리
X_test_imputed = imputer.transform(X_test)          # 같은 중앙값으로 테스트 데이터 처리

print("\n결측값 처리 후 남은 결측값:")
print(
    pd.isna(X_train_imputed).sum(),
    pd.isna(X_test_imputed).sum(),
)


# 센서 데이터 표준화
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)  # 센서별 평균과 표준편차 계산 후 표준화
X_test_scaled = scaler.transform(X_test_imputed)        # 학습 데이터에서 계산한 동일한 기준으로 테스트 데이터 표준화

print("\n표준화 후 학습 데이터 평균과 표준편차:")
print(
    f"{X_train_scaled.mean():.4f}",
    f"{X_train_scaled.std():.4f}",
)


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
mlp_classifier.fit(
    X_train_scaled,
    y_train,
)

print("MLP 학습 반복 횟수:", mlp_classifier.n_iter_)


# 테스트 데이터의 위험 여부와 위험 확률 예측
y_pred = mlp_classifier.predict(
    X_test_scaled
)

y_risk_probability = mlp_classifier.predict_proba(
    X_test_scaled
)[:, 1]

print("\n테스트 데이터 예측")
print(
    "예측값/위험 확률 개수:",
    len(y_pred),
    len(y_risk_probability),
)


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


# 실험ID 단위 5분할 교차검증
group_kfold = GroupKFold(
    n_splits=5,     # 5분할
)

cv_accuracies = []  # 각각의 정확도
cv_log_losses = []  # 각각의 손실값

for fold_number, (fold_train_index, fold_validation_index) in enumerate(
    group_kfold.split(
        X,
        y,
        groups=groups,
    ),
    start=1,
):
    # 현재 회차의 학습 데이터와 검증 데이터 분리
    X_fold_train = X.iloc[fold_train_index]
    X_fold_validation = X.iloc[fold_validation_index]

    y_fold_train = y.iloc[fold_train_index]
    y_fold_validation = y.iloc[fold_validation_index]

    # 학습 데이터의 중앙값으로 결측값 처리
    fold_imputer = SimpleImputer(
        strategy="median",
    )

    X_fold_train_imputed = fold_imputer.fit_transform(
        X_fold_train
    )
    X_fold_validation_imputed = fold_imputer.transform(
        X_fold_validation
    )

    # 학습 데이터 기준으로 표준화
    fold_scaler = StandardScaler()

    X_fold_train_scaled = fold_scaler.fit_transform(
        X_fold_train_imputed
    )
    X_fold_validation_scaled = fold_scaler.transform(
        X_fold_validation_imputed
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

    fold_model.fit(
        X_fold_train_scaled,
        y_fold_train,
    )

    # 검증 데이터 예측
    fold_pred = fold_model.predict(
        X_fold_validation_scaled
    )
    fold_risk_probability = fold_model.predict_proba(
        X_fold_validation_scaled
    )[:, 1]

    # 현재 회차의 정확도와 Log Loss 계산
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

    print(
        f"{fold_number}회차 - "
        f"정확도: {fold_accuracy:.4f}, "
        f"Log Loss: {fold_log_loss:.4f}"
    )


# 교차검증 결과의 평균과 표준편차 계산
cv_accuracy_series = pd.Series(cv_accuracies)
cv_log_loss_series = pd.Series(cv_log_losses)

print("\nMLP 5분할 교차검증 결과")
print("은닉층 구조:", HIDDEN_LAYER_SIZES)
print("초기 학습률:", LEARNING_RATE_INIT)
print(f"평균 정확도: {cv_accuracy_series.mean():.4f}")
print(
    "정확도 표준편차: "
    f"{cv_accuracy_series.std(ddof=0):.4f}"
)
print(f"평균 Log Loss: {cv_log_loss_series.mean():.4f}")
print(
    "Log Loss 표준편차: "
    f"{cv_log_loss_series.std(ddof=0):.4f}"
)
