import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from sklearn.model_selection import GroupKFold      # 모델 검증용

DATA_PATH = "battery_labels.csv"
HORIZON_SECONDS = 60               # 위험 구간 시간(초)
SENSOR_COLUMNS = [                  # 센서 열 목록(총 18개)
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


# 5단계 진입 시점과 진입 이후 데이터 제외
rows_before_filter = len(model_df)

model_df = model_df[
    model_df["seconds_to_stage5"] > 0    # 5단계 진입전만 필터링 (양수 시간)
].copy()

excluded_rows = rows_before_filter - len(model_df)
print("\n5단계 진입 이후 제외된 행 개수:\n", excluded_rows)

print("\n5단계 진입 이후 제외 후 데이터 크기:\n", model_df.shape)

print("\n0초 이하 데이터 개수:")
print((model_df["seconds_to_stage5"] <= 0).sum())

print("\n5단계 진입까지 남은 시간의 최솟값:")
print(model_df["seconds_to_stage5"].min())



# 향후 HORIZON_SECONDS초 이내 5단계 진입 여부

# 모델이 예측해야할 정답 라벨 생성
# 5단계 진입 전 중 HORIZON_SECONDS 시간 내 위험 여부(1 or 0)
model_df["target"] = (
    model_df["seconds_to_stage5"] <= HORIZON_SECONDS
).astype(int)           # True/False -> 1/0 변환

print("\n정답 라벨별 데이터 개수:")  # 0, 1 개수 확인
print(model_df["target"].value_counts().sort_index())

print("\n정답별 5단계 진입까지 남은 시간:")     # 0, 1 각각의 5단계까지 남은 시각
print(
    model_df.groupby("target")["seconds_to_stage5"]
    .agg(["min", "max"])
)

print("\n위험 데이터 비율:")
print(f"{model_df['target'].mean():.2%}")

print("\n정답 열 추가 후 데이터 크기:")
print(model_df.shape)


# 모델 입력, 정답, 실험 그룹 분리
X = model_df[SENSOR_COLUMNS]    # 모델 입력(센서)
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



# 모델 검증 5회(실험ID 단위)
group_kfold = GroupKFold(
    n_splits=5,     # 5개 분할
)

cv_accuracies = []  # 5개별 정확도
cv_log_losses = []  # 5개별 loss

# 5개 분할한 대로 회귀 돌리기
for fold_number, (fold_train_index, fold_validation_index) in enumerate(
    group_kfold.split(X, y, groups=groups),
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

# 평균과 표준편차 계산을 위해 Series로 변환
cv_accuracy_series = pd.Series(cv_accuracies)
cv_log_loss_series = pd.Series(cv_log_losses)

print("\n로지스틱 회귀 5겹 교차검증 결과")
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
