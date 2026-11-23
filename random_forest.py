import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, log_loss

# 1. 데이터 불러오기
df = pd.read_csv("output/battery_labels.csv", dtype={"exp_id": str}, encoding="utf-8-sig")
df = df.sort_values(["exp_id", "rel_time"]).reset_index(drop=True)

# 2. 정답 만들기 (5단계 시작 시각 t5까지 T초 이내면 1)
T = 60
df["t5"] = df["exp_id"].map(df[df["tr_stage"] >= 5].groupby("exp_id")["rel_time"].min())
df["y"] = ((df["t5"] - df["rel_time"]) <= T).astype(int)

# 3. 윈도우 피처 만들기 (실험별, 과거 값만 사용)
sensors =  ["meas_v",
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
            "thermal_t"]
new_cols = {}
for col in sensors:
    group = df.groupby("exp_id")[col]
    for w in [5, 10, 30, 60, 120, 300]:
        new_cols[f"{col}_diff{w}"] = df[col] - group.shift(w).fillna(group.transform("first"))
        new_cols[f"{col}_mean{w}"] = group.transform(lambda s: s.rolling(w, min_periods=1).mean())
        new_cols[f"{col}_std{w}"] = group.transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
features = sensors + list(new_cols)

# 4. 5단계 이후 행 제거
df = df[df["rel_time"] < df["t5"]].reset_index(drop=True)
print(f"행 {len(df)}개, 실험 {df['exp_id'].nunique()}개, 피처 {len(features)}개, 양성 비율 {df['y'].mean():.3f}")

X = df[features]
y = df["y"]

# 5. 실험 단위 5-fold로 RF 학습 및 예측

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df["proba"] = 0.0
df["fold"] = -1 
for fold, (train_idx, test_idx) in enumerate(cv.split(X, df["tr_method"], df["exp_id"])):
    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=50,
        max_features="sqrt",
        max_samples=0.5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    df.loc[test_idx, "proba"] = rf.predict_proba(X.iloc[test_idx])[:, 1]
    df.loc[test_idx, "fold"] = fold
    print(f"fold {fold}: PR-AUC {average_precision_score(y.iloc[test_idx], df.loc[test_idx, 'proba']):.3f}")

# 5-1. 정확도와 손실값
accs, losses = [], []
for fold in range(5):
    m = df["fold"] == fold                                                     # 이 fold에서 평가된 행만
    accs.append(accuracy_score(y[m], df.loc[m, "proba"] >= 0.5))             # 확률 0.5 이상을 위험으로 판정
    losses.append(log_loss(y[m], df.loc[m, "proba"], labels=[0, 1]))         # 확률이 정답에서 얼마나 벗어났나
print(f"교차검증 정확도: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
print(f"교차검증 Log Loss: {np.mean(losses):.4f} ± {np.std(losses, ddof=1):.4f}")
print(f"(참고) 항상 '정상'으로 답할 때 정확도: {1 - y.mean():.4f}")

# 6. 경보 만들기 (확률이 기준값 이상으로 5초 연속이면 경보)
threshold = 0.5
k = 5
above = (df["proba"] >= threshold).astype(int)
run = above.groupby([df["exp_id"], (above == 0).cumsum()]).cumsum()
df["alarm"] = run >= k

# 7. 분류 성능 (Recall, Precision, F1)
for name, d in [("가열", df[df["tr_method"] == 1]), ("과충전", df[df["tr_method"] == 2])]:
    tp = (d["alarm"] & (d["y"] == 1)).sum()
    fn = (~d["alarm"] & (d["y"] == 1)).sum()
    fp = (d["alarm"] & (d["y"] == 0)).sum()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"{name}: Recall {recall:.3f}, Precision {precision:.3f}, F1 {f1:.3f}")

# 8. 조기경보 성능 (사전 탐지 성공, 선행시간, 오경보 횟수)
results = []
for exp_id, e in df.groupby("exp_id"):
    risk_alarm = e["alarm"] & (e["y"] == 1)
    normal_alarm = (e["alarm"] & (e["y"] == 0)).astype(int)
    results.append({
        "exp_id": exp_id,
        "tr_method": e["tr_method"].iloc[0],
        "success": risk_alarm.any(),
        "lead_time": e["t5"].iloc[0] - e.loc[risk_alarm, "rel_time"].min(),
        "false_alarms": (normal_alarm.diff().fillna(normal_alarm) == 1).sum(),
    })
results = pd.DataFrame(results)

for name, r in [("가열", results[results["tr_method"] == 1]), ("과충전", results[results["tr_method"] == 2])]:
    print(f"{name}: 사전 탐지 {r['success'].sum()}/{len(r)}, 선행시간 중앙값 {r['lead_time'].median():.0f}초, "
          f"오경보 {r['false_alarms'].sum()}회, 오경보 있는 실험 {(r['false_alarms'] > 0).sum()}/{len(r)}")
print(f"사전 탐지 실패 실험: {results.loc[~results['success'], 'exp_id'].tolist()}")

# 9. 피처 중요도 상위 10개 (마지막 fold 모델 기준)
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(importance.head(10))

# 10. 예측 결과 저장
df[["exp_id", "rel_time", "t5", "tr_method", "y", "proba", "alarm"]].to_csv("output/rf_result.csv", index=False)
