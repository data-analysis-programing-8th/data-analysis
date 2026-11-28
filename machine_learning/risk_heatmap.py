"""같은 실험 분할에서 학습·검증 손실과 경보 성능을 비교한다."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, GRAPH_PATH, RANDOM_STATE
from .logistic_regression import HORIZON_SECONDS, SENSOR_COLUMNS, make_logistic_model


FOLDS = 5
THRESHOLD = 0.5
HORIZONS = (HORIZON_SECONDS, 180, 300)
REFERENCE_LOSS = 0.138
MODEL_NAMES = {"RF": "랜덤 포레스트", "LR": "로지스틱 회귀"}
COLORS = {"RF": "#7854C4", "LR": "#27A9AA", "4단계": "#858B94",
          "기준": "#858B94"}
METRICS = ("사전 탐지율 (%)", "선행 시간 중앙값 (초)", "실험당 오경보 (회)")


def load_samples():
    raw = pd.read_csv(DATA_PATH, usecols=["exp_id", "rel_time", "tr_stage", "tr_method", *SENSOR_COLUMNS])
    first_stage = raw.groupby(["exp_id", "tr_stage"])["rel_time"].min().unstack()
    data = raw.join(first_stage[5].rename("stage5"), on="exp_id")
    data = data.loc[data.rel_time.lt(data.stage5)].sort_values(["exp_id", "rel_time"]).copy()
    # 10초 구간에서 마지막 측정값을 쓰고, 그 측정의 실제 시각에 판정한다.
    data["bin"] = data.rel_time.floordiv(10)
    data = data.drop_duplicates(["exp_id", "bin"], keep="last").copy()
    data["lead"] = data.stage5 - data.rel_time
    stage4_lead = (first_stage[5] - first_stage[4]).reindex(data.exp_id.unique())
    return data, stage4_lead.where(stage4_lead.gt(0))


def alarm_metrics(first_valid_lead, false_alarms):
    return (100 * first_valid_lead.notna().mean(), first_valid_lead.median()
            if first_valid_lead.notna().any() else np.nan, false_alarms.mean())


def evaluate(data, stage4_lead):
    x = data[SENSOR_COLUMNS]
    folds = list(GroupKFold(FOLDS).split(x, groups=data.exp_id))
    losses, alarms = [], []
    for horizon in HORIZONS:
        y = data.lead.le(horizon)
        models = {
            "RF": RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                         random_state=RANDOM_STATE, n_jobs=-1),
            "LR": make_pipeline(StandardScaler(), make_logistic_model()),
        }
        for name, estimator in models.items():
            out_of_fold = np.full(len(data), np.nan)
            for fold, (train, test) in enumerate(folds, start=1):
                model = make_pipeline(SimpleImputer(strategy="median"), estimator)
                model.fit(x.iloc[train], y.iloc[train])
                out_of_fold[test] = model.predict_proba(x.iloc[test])[:, 1]
                if horizon == HORIZON_SECONDS:
                    train_loss = log_loss(y.iloc[train], model.predict_proba(x.iloc[train]))
                    test_loss = log_loss(y.iloc[test], out_of_fold[test], labels=[False, True])
                    losses.append((name, fold, train_loss, test_loss))
            assert np.isfinite(out_of_fold).all()
            alarm = pd.Series(out_of_fold >= THRESHOLD, index=data.index)
            # 연속해서 켜져 있는 경보는 한 번으로 센다. H 밖에서 시작하면 오경보다.
            previous = alarm.groupby(data.exp_id).shift(fill_value=False)
            adjacent = data.groupby("exp_id").bin.diff().eq(1)
            starts = alarm & ~(previous & adjacent)
            valid = data.lead.where(starts & y).groupby(data.exp_id).max()
            false = (starts & ~y).groupby(data.exp_id).sum()
            alarms.append((name, horizon, *alarm_metrics(valid, false)))
        # 센서 모델과 별개로 실제 4단계 최초 진입에서 딱 한 번 발생하는 경보를 집계한다.
        alarms.append(("4단계", horizon, *alarm_metrics(
            stage4_lead.where(stage4_lead.le(horizon)), stage4_lead.gt(horizon))))
    return (pd.DataFrame(losses, columns=["모델", "폴드", "학습", "검증"]),
            pd.DataFrame(alarms, columns=["모델", "H", *METRICS]))


def heatmap(ax, values, rows, columns, models, scale, fmt):
    rgb = np.ones((*values.shape, 3))
    for col, model in enumerate(models):
        strength = np.clip(values[:, col] / scale, 0, 1)
        blend = np.where(np.isfinite(strength), .12 + .48 * strength, 0)
        rgb[:, col] -= blend[:, None] * (1 - np.asarray(to_rgb(COLORS[model])))
    ax.imshow(rgb, aspect="auto")
    headers = [f"{label} ({MODEL_NAMES[label]})" if label in MODEL_NAMES else label for label in columns]
    ax.set_xticks(range(len(columns)), headers, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(rows)), rows)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row, col in np.ndindex(values.shape):
        value = values[row, col]
        ax.text(col, row, format(value, fmt) if np.isfinite(value) else "—",
                ha="center", va="center", color="#192536", fontweight="bold")
    for row in range(len(rows) - 1):
        ax.axhline(row + .5, color="white", linewidth=2)
    for col in range(len(columns) - 1):
        ax.axvline(col + .5, color="white", linewidth=2)


def draw(losses, alarms, data):
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
        if name in fonts:
            plt.rcParams["font.family"] = name
            break
    fig, axes = plt.subplots(4, 1, figsize=(10, 13.5), facecolor="white")
    fig.subplots_adjust(left=.15, right=.95, top=.855, bottom=.10, hspace=1.05)
    experiments = data.drop_duplicates("exp_id").tr_method.value_counts()
    conditions = (f"{FOLDS}겹 실험 단위 검증 · 경보 기준 {THRESHOLD:g} · "
                  f"가열 {experiments.get(1, 0)}건 · 과충전 {experiments.get(2, 0)}건 · 실제 데이터")
    fig.text(.08, .975, "모델의 학습 손실과 사전 경보 성능", fontsize=22, weight="bold")
    fig.text(.08, .947, conditions, fontsize=11, color="#475569")

    names = list(MODEL_NAMES)
    means = losses.groupby("모델")[["학습", "검증"]].mean().reindex(names)
    gap = (means["검증"] - means["학습"]).idxmax()
    loss_table = np.column_stack([means.T.to_numpy(), [REFERENCE_LOSS, REFERENCE_LOSS]])
    axes[0].set_title(f"{MODEL_NAMES[gap]}의 검증 손실과 학습 손실 차이가 가장 큽니다.  (H={HORIZON_SECONDS}초)",
                      loc="left", y=1.43, fontsize=14, weight="bold")
    heatmap(axes[0], loss_table, ["학습", "검증"], [*names, "3.1% 기준"],
            [*names, "기준"], max(REFERENCE_LOSS, means.to_numpy().max()), ".3f")

    names = [*names, "4단계"]
    for ax, metric, scale, fmt in zip(axes[1:], METRICS, [100, max(HORIZONS), None], [".1f", ".0f", ".2f"]):
        values = alarms.pivot(index="H", columns="모델", values=metric).loc[list(HORIZONS), names]
        ax.set_title(metric, loc="left", y=1.39, fontsize=14, weight="bold")
        heatmap(ax, values.to_numpy(), [f"H={h}초" for h in HORIZONS],
                names, names, scale or max(1, values.max().max()), fmt)
    first, last = HORIZONS[0], HORIZONS[-1]
    detection = alarms.pivot(index="H", columns="모델", values=METRICS[0])
    improved = [MODEL_NAMES[name] for name in names[:-1] if detection.loc[last, name] > detection.loc[first, name]]
    message = (f"H를 {first}초에서 {last}초로 늘리면 {', '.join(improved)}의 탐지율이 높아집니다."
               if improved else f"H를 {first}초에서 {last}초로 늘려도 탐지율이 높아진 모델은 없습니다.")
    fig.text(.08, .718, message, fontsize=14, weight="bold")
    fig.text(.08, .050, "H는 최초 5단계 진입 전 위험 시간대입니다. 모델은 10초 구간마다 판정합니다.", fontsize=10)
    fig.text(.08, .030, "선행 시간은 탐지된 실험의 첫 유효 경보 기준입니다. H 이전에 시작한 경보는 오경보로 셉니다.", fontsize=10)
    fig.text(.08, .010, "회색 4단계 열은 실제 4단계 진입 때 실험당 한 번 울리는 규칙입니다. 손실의 0.138은 제공된 기준값입니다.", fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="창을 열지 않고 PNG와 CSV 저장")
    args = parser.parse_args()
    plt.switch_backend("Agg" if args.no_show else "TkAgg")
    data, stage4_lead = load_samples()
    losses, alarms = evaluate(data, stage4_lead)
    GRAPH_PATH.mkdir(parents=True, exist_ok=True)
    losses.to_csv(GRAPH_PATH / "모델_학습검증_손실.csv", index=False, encoding="utf-8-sig")
    alarms.to_csv(GRAPH_PATH / "모델_H별_경보성능.csv", index=False, encoding="utf-8-sig")
    fig = draw(losses, alarms, data)
    fig.savefig(GRAPH_PATH / "모델_평가_히트맵.png", dpi=160, facecolor="white")
    if not args.no_show:
        fig.canvas.manager.set_window_title("모델 평가 히트맵")
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
