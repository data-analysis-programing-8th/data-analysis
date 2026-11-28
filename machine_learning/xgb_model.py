import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from xgboost import XGBClassifier

# 0. 설정 (값을 바꿀 때는 여기만 수정)
DATA_PATH = "20.배터리 열폭주 제어 멀티모달 데이터/output/battery_labels.csv"   # preprocessing.py가 만든 병합 CSV

T = 60                     # 위험 구간: 5단계 진입 전 T초 (팀 기준 60초)
WINDOWS = [10, 30, 60]     # 윈도우 피처를 계산할 최근 구간 길이(초)
THRESHOLD = 0.5            # 예측 확률이 이 값 이상이면 '위험'으로 판정
K_CONSEC = 5               # 위험 판정이 K초 연속 이어지면 경보
N_SPLITS = 5               # 교차검증 분할 수
SEED = 42                  # 결과 재현용 시드

SENSORS = [                # 입력으로 쓸 센서 (split·meas_a·indu_t·tr_stage는 누수·무의미라 제외)
    "meas_v", "meas_t1", "meas_t2", "meas_p",                    # 전압, 표면온도, 양극부 온도, 압력
    "meas_ch4", "meas_co", "meas_co2", "meas_hcn", "meas_hcl",   # 가스 1~5
    "meas_hf", "meas_n2o", "meas_no", "meas_no2", "meas_so2",    # 가스 6~10
    "atmp_t", "thermal_t",                                       # 주변 온도, 열화상 온도
]

# 1. 데이터 불러오기
df = pd.read_csv(DATA_PATH, dtype={"exp_id": str}, encoding="utf-8-sig")  # exp_id가 숫자로 바뀌지 않게 문자열로 읽음
df = df.sort_values(["exp_id", "rel_time"]).reset_index(drop=True)        # 실험별 시간순 정렬 (윈도우 계산의 전제)
print(f"[1] 불러온 데이터: {len(df)}행, 실험 {df['exp_id'].nunique()}개")

# 2. 정답 만들기 (5단계 진입 전 T초 이내면 1, 아니면 0)
stage5_rows = df[df["tr_stage"] >= 5]                           # 5단계 이상인 행 (6단계로 바로 건너뛴 실험도 포함)
t5_by_exp = stage5_rows.groupby("exp_id")["rel_time"].min()     # 실험별 5단계 시작 시각
df["t5"] = df["exp_id"].map(t5_by_exp)                          # 그 실험의 모든 행에 같은 값으로 붙임

time_to_t5 = df["t5"] - df["rel_time"]                          # 5단계까지 남은 시간(초)
df["y"] = (time_to_t5 <= T).astype(int)                         # T초 이하면 위험(1), 아니면 정상(0)

stages_by_exp = df.groupby("exp_id")["tr_stage"].unique()       # 실험별로 나온 단계 목록
skipped = [exp for exp, stages in stages_by_exp.items() if 5 not in stages]  # 5단계 없이 6단계로 간 실험

print(f"[2] 5단계 시작 시각이 있는 실험: {len(t5_by_exp)}개")
print(f"    5단계 기록 없이 6단계로 건너뛴 실험: {len(skipped)}개 {skipped}")

# 3. 윈도우 피처 만들기 (실험별로, 과거 값만 사용)
new_features = {}                                               # 새로 만든 피처를 모아 뒀다가 한 번에 붙임

for sensor in SENSORS:
    by_exp = df.groupby("exp_id")[sensor]                       # 실험별로 나눠서 계산 (실험끼리 값이 섞이지 않게)
    first_value = by_exp.transform("first")                     # 실험의 첫 값 (초반에 w초 전 값이 없을 때 대신 사용)

    for w in WINDOWS:
        past_value = by_exp.shift(w).fillna(first_value)        # w초 전 값
        rolling = by_exp.rolling(w, min_periods=1)              # 지금을 포함한 최근 w초 구간

        new_features[f"{sensor}_diff{w}"] = df[sensor] - past_value                                     # 최근 w초 변화량
        new_features[f"{sensor}_mean{w}"] = rolling.mean().reset_index(level=0, drop=True)              # 최근 w초 평균
        new_features[f"{sensor}_std{w}"] = rolling.std().reset_index(level=0, drop=True).fillna(0)      # 최근 w초 흔들림 (첫 행은 0)

df = pd.concat([df, pd.DataFrame(new_features)], axis=1)        # 원래 표 오른쪽에 피처 열 붙이기
FEATURES = SENSORS + list(new_features)                         # 모델 입력으로 쓸 열 목록 (원래 값 + 윈도우 피처)

print(f"[3] 피처 {len(FEATURES)}개 (원래 값 {len(SENSORS)}개 + 윈도우 피처 {len(new_features)}개), "
      f"결측 {int(df[FEATURES].isna().sum().sum())}개")

# 4. 5단계 이후 행 제거하고 입력·정답 나누기
df = df[df["rel_time"] < df["t5"]].reset_index(drop=True)       # 열폭주가 이미 시작된 행은 '미리 경보' 대상이 아님

X = df[FEATURES]                                                # 입력: 피처 160개 (정답 재료인 t5·y·tr_stage는 없음)
y = df["y"]                                                     # 정답: 60초 안에 5단계 진입하면 1

heating = df["tr_method"] == 1                                  # 가열 실험 행
overcharge = df["tr_method"] == 2                               # 과충전 실험 행

print(f"[4] 학습 데이터: {len(df)}행, 실험 {df['exp_id'].nunique()}개, 남은 최대 단계 {df['tr_stage'].max()}")
print(f"    위험 구간 비율: 전체 {y.mean():.3f} / 가열 {y[heating].mean():.3f} / 과충전 {y[overcharge].mean():.3f}")

# 5. 실험 단위 5-fold 교차검증으로 XGBoost 학습·예측
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)   # 같은 실험은 한 fold에만, 가열/과충전 비율 유지
df["proba"] = 0.0                                               # 평가 fold에서 낸 예측 확률을 채울 열
df["fold"] = -1                                                 # 각 행이 몇 번 fold에서 평가됐는지 (-1 = 아직 안 됨)
importances = []                                                # fold별 피처 중요도 (나중에 평균)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, df["tr_method"], df["exp_id"])):
    train_exps = set(df.loc[train_idx, "exp_id"])
    test_exps = set(df.loc[test_idx, "exp_id"])
    assert train_exps.isdisjoint(test_exps)                     # 학습·평가에 같은 실험이 있으면 즉시 중단 (누수 방지)

    y_train = y.iloc[train_idx]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()    # 정상 행 수 ÷ 위험 행 수 (학습 fold로만 계산)

    model = XGBClassifier(
        n_estimators=300,                # 트리 개수
        max_depth=3,                     # 트리 깊이 (얕게 → 실험 97개에서 과적합 방지)
        learning_rate=0.05,              # 트리 하나의 반영 비율 (작을수록 천천히 안정적으로 학습)
        subsample=0.8,                   # 트리마다 학습 행의 80%만 사용
        colsample_bytree=0.8,            # 트리마다 피처의 80%만 사용
        min_child_weight=10,             # 잎에 필요한 최소 가중치 (너무 작은 잎을 막음)
        scale_pos_weight=pos_weight,     # 위험 행에 가중치를 줘서 불균형 보정
        tree_method="hist",              # 값을 구간으로 묶어 빠르게 학습
        n_jobs=-1,                       # CPU 전부 사용
        random_state=SEED,
    )
    model.fit(X.iloc[train_idx], y_train)                       # 학습 fold로만 학습

    df.loc[test_idx, "proba"] = model.predict_proba(X.iloc[test_idx])[:, 1]   # 처음 보는 실험의 위험 확률
    df.loc[test_idx, "fold"] = fold
    importances.append(model.feature_importances_)

    fold_ap = average_precision_score(y.iloc[test_idx], df.loc[test_idx, "proba"])
    print(f"[5] fold {fold}: 학습 실험 {len(train_exps)}개 / 평가 실험 {len(test_exps)}개 | PR-AUC {fold_ap:.3f}")

 # 5-1. 정확도·Log Loss·PR-AUC (fold별로 계산해서 평균 ± 표준편차)
accuracies, losses, pr_aucs = [], [], []

for fold in range(N_SPLITS):
    in_fold = df["fold"] == fold                                # 이 fold에서 평가된 행만
    y_true = y[in_fold]

    accuracies.append(accuracy_score(y_true, df.loc[in_fold, "proba"] >= THRESHOLD))   # 확률 0.5 이상을 위험으로 판정한 뒤 맞은 비율
    losses.append(log_loss(y_true, df.loc[in_fold, "proba"], labels=[0, 1]))       # 확률이 정답에서 벗어난 정도 (낮을수록 좋음)
    pr_aucs.append(average_precision_score(y_true, df.loc[in_fold, "proba"]))      # 위험 순간을 정상보다 위에 줄 세운 정도

always_normal = 1 - y.mean()                                    # 모든 행을 '정상'으로 답했을 때 정확도 (비교 기준)
random_pr_auc = y.mean()                                        # 무작위로 찍었을 때 PR-AUC (비교 기준)

print(f"[5-1] 정확도   : {np.mean(accuracies):.4f} ± {np.std(accuracies, ddof=1):.4f} (항상 정상일 때 {always_normal:.4f})")
print(f"      Log Loss : {np.mean(losses):.4f} ± {np.std(losses, ddof=1):.4f}")
print(f"      PR-AUC   : {np.mean(pr_aucs):.3f} ± {np.std(pr_aucs, ddof=1):.3f} (무작위일 때 {random_pr_auc:.3f})")

# 5-2. 단일 분할 (실험 단위 80% 학습 / 20% 평가, 팀 결과 형식 비교용)
single_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)    # 실험 단위로 20%를 평가용으로 떼어 냄
single_train_idx, single_test_idx = next(single_split.split(X, y, groups=df["exp_id"]))

single_train_exps = set(df.loc[single_train_idx, "exp_id"])
single_test_exps = set(df.loc[single_test_idx, "exp_id"])
assert single_train_exps.isdisjoint(single_test_exps)                            # 학습·평가에 같은 실험이 있으면 즉시 중단

single_y_train = y.iloc[single_train_idx]
single_pos_weight = (single_y_train == 0).sum() / (single_y_train == 1).sum()    # 정상 행 수 ÷ 위험 행 수 (학습 데이터로만 계산)

single_model = XGBClassifier(                                                   # 5번과 같은 설정
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    scale_pos_weight=single_pos_weight,
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
)
single_model.fit(X.iloc[single_train_idx], single_y_train)                       # 학습 실험(80%)으로만 학습

single_y_test = y.iloc[single_test_idx]
single_proba = single_model.predict_proba(X.iloc[single_test_idx])[:, 1]         # 평가 실험(20%)의 위험 확률

print(f"[5-2] 단일 분할: 학습 실험 {len(single_train_exps)}개 / 평가 실험 {len(single_test_exps)}개")
print(f"      정확도   : {accuracy_score(single_y_test, single_proba >= THRESHOLD):.4f} "
      f"(항상 정상일 때 {1 - single_y_test.mean():.4f})")
print(f"      Log Loss : {log_loss(single_y_test, single_proba, labels=[0, 1]):.4f}")
print(f"      PR-AUC   : {average_precision_score(single_y_test, single_proba):.3f}")

# 6. 경보 만들기 (위험 판정이 K_CONSEC초 연속 이어지면 경보)
is_risky = (df["proba"] >= THRESHOLD).astype(int)              # 매초 위험 판정 (확률 0.5 이상 = 1)
streak_id = (is_risky == 0).cumsum()                            # 정상 판정이 나올 때마다 번호가 바뀜 → 연속 구간 구분용
streak = is_risky.groupby([df["exp_id"], streak_id]).cumsum()   # 실험별로, 위험 판정이 지금 몇 초째 이어지는지
df["alarm"] = streak >= K_CONSEC                                # K_CONSEC초 이상 이어졌으면 경보 ON

alarm_exps = df.loc[df["alarm"], "exp_id"].nunique()            # 경보가 한 번이라도 켜진 실험 수
print(f"[6] 위험 판정 행 {is_risky.sum()}개 중 경보로 이어진 행 {df['alarm'].sum()}개, "
      f"경보가 켜진 실험 {alarm_exps}/{df['exp_id'].nunique()}개")

# 7. 분류 성능 (경보 기준 Recall, Precision, F1)
groups = {"가열": heating, "과충전": overcharge, "전체": df["tr_method"] > 0}   # 결과를 나눠 볼 행 묶음

for name, rows in groups.items():
    alarm = df.loc[rows, "alarm"]
    risk = df.loc[rows, "y"] == 1                               # 실제 위험 구간인 행

    tp = (alarm & risk).sum()                                   # 위험 구간을 위험으로 탐지
    fn = (~alarm & risk).sum()                                  # 위험 구간을 놓침
    fp = (alarm & ~risk).sum()                                  # 정상 구간에 오경보

    recall = tp / (tp + fn)                                     # 실제 위험 중 탐지한 비율
    precision = tp / (tp + fp)                                  # 경보 중 실제 위험이었던 비율
    f1 = 2 * precision * recall / (precision + recall)          # 둘의 조화 평균

    print(f"[7] {name:<3}: Recall {recall:.3f} / Precision {precision:.3f} / F1 {f1:.3f}  (TP {tp}, FN {fn}, FP {fp})")

# 8. 조기경보 성능 (실험마다 계산한 뒤 묶어서 요약)
exp_results = []

for exp_id, exp in df.groupby("exp_id"):
    t5 = exp["t5"].iloc[0]                                      # 이 실험의 5단계 시작 시각
    alarm = exp["alarm"]
    risk = exp["y"] == 1

    risk_alarm_times = exp.loc[alarm & risk, "rel_time"]        # 위험 구간(마지막 T초) 안에서 경보가 켜진 시각들
    any_alarm_times = exp.loc[alarm, "rel_time"]                # 5단계 전 아무 때나 경보가 켜진 시각들

    normal_alarm = (alarm & ~risk).astype(int)                  # 정상 구간 경보 (0/1)
    alarm_starts = normal_alarm.diff().fillna(normal_alarm) == 1   # 0→1로 바뀌는 순간 = 오경보 1회 시작

    exp_results.append({
        "exp_id": exp_id,
        "method": "가열" if exp["tr_method"].iloc[0] == 1 else "과충전",
        "success": len(risk_alarm_times) > 0,                   # 위험 구간 안에 경보가 있었나
        "success_any": len(any_alarm_times) > 0,                # (팀 문서 기준) 5단계 전에 경보가 있었나
        "lead_time": t5 - risk_alarm_times.min(),               # 위험 구간 안 첫 경보 → 5단계까지 (최대 T초)
        "lead_time_any": t5 - any_alarm_times.min(),            # (팀 문서 기준) 최초 경보 → 5단계까지
        "false_alarms": alarm_starts.sum(),                     # 정상 구간 오경보 횟수 (연속 경보는 1회)
    })

exp_results = pd.DataFrame(exp_results)

for name in ["가열", "과충전", "전체"]:
    r = exp_results if name == "전체" else exp_results[exp_results["method"] == name]
    print(f"[8] {name:<3}: 사전 탐지 {r['success'].sum()}/{len(r)} "
          f"(문서 기준 {r['success_any'].sum()}/{len(r)}), "
          f"선행시간 중앙값 {r['lead_time'].median():.0f}초 (문서 기준 {r['lead_time_any'].median():.0f}초), "
          f"오경보 {r['false_alarms'].sum()}회 (오경보 있는 실험 {(r['false_alarms'] > 0).sum()}/{len(r)})")

missed = exp_results.loc[~exp_results["success"], "exp_id"].tolist()
print(f"    사전 탐지 실패 실험: {missed}")

# 9. 피처 중요도 (5개 fold 평균)
importance = pd.Series(np.mean(importances, axis=0), index=FEATURES).sort_values(ascending=False)
print("[9] 피처 중요도 상위 10개")
print(importance.head(10).round(4).to_string())

sensor_importance = {}                                          # 센서 하나에서 나온 피처 10개(원래 값 + 윈도우 9개)의 중요도 합
for sensor in SENSORS:
    is_from_sensor = (importance.index == sensor) | importance.index.str.startswith(sensor + "_")
    sensor_importance[sensor] = importance[is_from_sensor].sum()

print("[9] 센서별 중요도 합계 상위 5개")
print(pd.Series(sensor_importance).sort_values(ascending=False).head(5).round(3).to_string())