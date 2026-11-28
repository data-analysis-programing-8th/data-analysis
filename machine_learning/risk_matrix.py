"""발표에서는 가열 실험의 로지스틱 회귀 표 하나를 먼저 읽으면 된다.

1. 무엇을 위험으로 봤나? 최초 5단계 진입까지 60초 이하로 남은 측정값에 1을 붙였다.
2. 어떻게 검증했나? 한 실험 전체를 학습 또는 검증에 배정하고, 검증 예측만 모았다.
3. 얼마나 놓쳤나? 왼쪽 아래 칸은 실제 위험 측정값 중 모델이 정상으로 판정한 비율이다.
   예를 들어 위험 측정값 100개 중 30개를 정상으로 판정하면 미탐률은 30%다.
   한 실험에서도 여러 번 측정하므로 이 수치만으로 경보를 한 번도 못 낸 실험 수는 알 수 없다.

오른쪽 과충전 표도 같은 방식으로 읽는다. 랜덤 포레스트는 4단계 전 120초를
10초 구간 단위로 예측한다. 두 모델의 성능 순위를 비교하려면 위험 정의와
평가 단위, 학습·검증 실험을 먼저 통일해야 한다.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from .logistic_regression import HORIZON_SECONDS, validation_results
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from config import FOCUS_SENSORS_BY_METHOD, GRAPH_PATH, RANDOM_STATE
from .precursor_ml import HORIZON, features


def add_counts(rows, method, n_experiments, name, y, predicted_labels, stage, horizon, unit):
    # 가로 한 줄을 따라 읽는다. 윗줄은 실제 정상 중 정상 판정과 오경보,
    # 아랫줄은 실제 위험 중 미탐과 위험 감지다. 각 줄의 분모는 그 줄 전체 건수다.
    counts = confusion_matrix(y, predicted_labels, labels=[False, True])
    assert counts.sum() == len(y)
    for actual, predicted in np.ndindex(2, 2):
        total = counts[actual].sum()
        count = counts[actual, predicted]
        rows.append((method, n_experiments, name, actual, predicted, count, total,
                     100 * count / total if total else np.nan, stage, horizon, unit))


def evaluate():
    data = features()
    models = {
        "랜덤 포레스트": RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                           random_state=RANDOM_STATE, n_jobs=-1),
    }
    rows = []
    logistic = validation_results()
    # 로지스틱 회귀는 가열·과충전을 합쳐 학습한다. 여기서 방식을 나누는 목적은
    # 같은 모델이 각 실험 조건에서 어떤 예측을 했는지 따로 집계하는 것이다.
    for code, method in [(1, "가열"), (2, "과충전")]:
        subset = logistic.loc[logistic.tr_method.eq(code)]
        add_counts(rows, method, subset.exp_id.nunique(), "로지스틱 회귀", subset.target,
                   subset.prediction, 5, HORIZON_SECONDS, "측정 행")
    for code, method in [(1, "가열"), (2, "과충전")]:
        group = data.loc[data.method.eq(code)]
        sensors = list(FOCUS_SENSORS_BY_METHOD[method])
        # 트리 모델은 두 가지 변화를 본다. 첫 60초의 기준값에서 얼마나 벗어났는지,
        # 현재 10초 구간의 센서 중앙값이 30초 전 구간에 비해 얼마나 달라졌는지다.
        x = group[sensors + [s + "_change" for s in sensors]]
        y = group.lead.le(HORIZON)
        ids = group.exp.unique()
        folds = list(KFold(5, shuffle=True, random_state=RANDOM_STATE).split(ids))
        for name, estimator in models.items():
            predictions = pd.Series(index=group.index, dtype="float64")
            for train, test in folds:
                fit, check = group.exp.isin(ids[train]), group.exp.isin(ids[test])
                model = make_pipeline(SimpleImputer(strategy="median"), estimator)
                model.fit(x.loc[fit], y.loc[fit])
                predictions.loc[check] = model.predict_proba(x.loc[check])[:, 1]
            assert predictions.notna().all()
            # 위험 확률이 0.5 이상이면 경보로 센다. 이 기준을 낮추면 같은 예측에서
            # 경보가 늘어날 수 있으므로 미탐(왼쪽 아래)과 오경보(오른쪽 위)를 함께 본다.
            add_counts(rows, method, len(ids), name, y, predictions.ge(.5), 4, HORIZON, "10초 구간")
    return pd.DataFrame(rows, columns=["방식", "실험수", "모델", "실제위험", "예측위험",
                                       "구간수", "실제행합계", "행비율", "도달단계", "위험구간초", "평가단위"])


def draw(results):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
        if font in available:
            plt.rcParams["font.family"] = font
            break
    fig = plt.figure(figsize=(12, 7.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def label(x, y, value, size=11, color="#192536", **kwargs):
        ax.text(x, y, value, size=size, color=color, va="center", **kwargs)

    label(.038, .94, "각 모델은 위험 구간을 얼마나 놓쳤을까?", 20, weight="bold")
    label(.039, .879, "빨간 테두리 = 위험 미탐   ·   행마다 예측 대상과 평가 단위를 확인하세요", 12, "#475569")
    palette = LinearSegmentedColormap.from_list("risk", ["#f1f6fc", "#88b8e8"])
    for col, (method, group) in enumerate(results.groupby("방식", sort=False)):
        x0 = .235 + col * .405
        label(x0 + .149, .796, method, 15, ha="center", weight="bold")
        for row, (model, matrix) in enumerate(group.groupby("모델", sort=False)):
            top = .702 - row * .31
            if col == 0:
                label(.028, top - .050, model, 14, weight="bold")
                info = matrix.iloc[0]
                label(.028, top - .095, f"{info.도달단계}단계 전 {info.위험구간초}초", 10)
            label(x0 + .15, top + .056, f"{matrix['실험수'].iloc[0]}개 실험", 9, ha="center")
            unit = "행" if matrix["평가단위"].iloc[0] == "측정 행" else "구간"
            panel = fig.add_axes([x0, top - .148, .30, .148])
            cells = matrix.set_index(["실제위험", "예측위험"]).sort_index()
            rates = cells["행비율"].unstack().to_numpy()
            counts = cells["구간수"].unstack().to_numpy()
            panel.imshow(rates, cmap=palette, vmin=0, vmax=100, aspect="auto",
                         interpolation="nearest", origin="upper")
            panel.set_xticks([0, 1], ["예측 정상", "예측 위험"])
            panel.set_yticks([0, 1], ["실제 정상", "실제 위험"])
            panel.xaxis.tick_top()
            panel.tick_params(length=0, colors="#475569", labelsize=11, pad=5)
            for spine in panel.spines.values():
                spine.set_visible(False)
            panel.axvline(.5, color="white", lw=2)
            panel.axhline(.5, color="white", lw=2)
            panel.add_patch(Rectangle((-.49, .51), .98, .98, fill=False,
                                      edgecolor="#c32232", linewidth=2))
            for actual, predicted in np.ndindex(2, 2):
                rate = rates[actual, predicted]
                value = f"{rate:.1f}%" if np.isfinite(rate) else "—"
                panel.text(predicted, actual - .14, value, ha="center", va="center",
                           color="#192536", fontsize=14, weight="bold")
                panel.text(predicted, actual + .23, f"{counts[actual, predicted]:,}{unit}",
                           ha="center", va="center", color="#253850", fontsize=9)
    label(.039, .074, "실제 정상과 실제 위험은 각각 100%   ·   로지스틱 회귀는 측정 행, 나머지 모델은 10초 구간 기준", color="#475569")
    label(.039, .036, "로지스틱 회귀는 전체 방식 통합 학습 후 방식별 집계   ·   랜덤 포레스트는 방식별 학습", color="#475569")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="창을 열지 않고 결과 저장")
    args = parser.parse_args()
    plt.switch_backend("Agg" if args.no_show else "TkAgg")
    results = evaluate()
    GRAPH_PATH.mkdir(parents=True, exist_ok=True)
    results.to_csv(GRAPH_PATH / "위험구간_혼동행렬.csv", index=False, encoding="utf-8-sig")
    fig = draw(results)
    fig.savefig(GRAPH_PATH / "위험구간_혼동행렬.png", dpi=150, facecolor=fig.get_facecolor())
    if not args.no_show:
        fig.canvas.manager.set_window_title("위험 구간 예측 · 모델별 혼동행렬")
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
