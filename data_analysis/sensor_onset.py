import pandas as pd
import numpy as np
import argparse
from config import DATA_PATH, GRAPH_PATH, SENSOR_COLUMNS, FOCUS_SENSORS_BY_METHOD
from .eda import SENSOR_LABELS
import matplotlib.pyplot as plt


def summarize_focus_sensors(results):
    rows = []
    for method, sensors in FOCUS_SENSORS_BY_METHOD.items():
        group = results.loc[results["방식"].eq(method)]
        if group.empty:
            continue
        for sensor in sensors:
            samples = group.loc[group["원본센서"].eq(sensor)]
            total = group["실험"].nunique()
            detected = samples["시작초"].count()
            row = {"방식": method, "원본센서": sensor, "센서": SENSOR_LABELS[sensor],
                   "전체실험수": total, "검출수": detected, "검출률(%)": 100 * detected / total,
                   "선행표본": samples["선행초"].count()}
            for metric in ("시작초", "선행초"):
                for label, q in (("Q1", .25), ("중앙값", .5), ("Q3", .75)):
                    row[metric + label] = samples[metric].quantile(q)
            rows.append(row)
    return pd.DataFrame(rows)


def save_focus_comparison(results, output_dir):
    summary = summarize_focus_sensors(results)
    summary.to_csv(output_dir / "실험별_중점센서_비교.csv", index=False, encoding="utf-8-sig")
    colors = {"과충전": "#195A94", "가열": "#AC4916"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True, layout="constrained")
    labels = []
    for y, row in summary.iterrows():
        method = row["방식"]
        color = colors[method]
        labels.append(f"{method} · {row['센서']}")
        rate = row["검출률(%)"]
        axes[0].barh(y, rate, height=.5, color=color)
        axes[0].text(rate + 2, y, f"{row['검출수']}/{row['전체실험수']}", va="center")
        for ax, metric in zip(axes[1:], ("시작초", "선행초")):
            median = row[metric + "중앙값"]
            if pd.notna(median):
                bounds = row[[metric + "Q1", metric + "Q3"]].to_numpy(dtype=float) / 60
                ax.plot(bounds, [y, y], color=color, linewidth=3)
                ax.scatter(median / 60, y, color=color, zorder=3)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 125)
    axes[0].set_xticks([0, 25, 50, 75, 100])
    for ax, title, unit in zip(axes, ("변화가 검출된 실험", "실험 시작부터 변화까지", "변화부터 5단계 도달까지"),
                               ("검출률 (%)", "경과 시간 (분)", "남은 시간 (분)")):
        ax.set(title=title, xlabel=unit, xlim=(0, ax.get_xlim()[1]))
        ax.grid(axis="y", visible=False)
    fig.suptitle("가열과 과충전에서 센서값은 언제 달라졌는가?\n점은 중앙값, 가로선은 실험의 가운데 50%가 속한 범위")
    fig.savefig(output_dir / "실험별_중점센서_비교.png", dpi=180)
    return fig

def main(show=False):
    if show:
        plt.switch_backend("TkAgg")
    data = pd.read_csv(DATA_PATH)
    rows = []
    for exp, g in data.groupby("exp_id"):
        end = g.loc[g.tr_stage.eq(5), "rel_time"].min()
        g = g[g.tr_stage.between(1, 4)]
        if pd.notna(end):
            g = g[g.rel_time < end]
        base = g.loc[g.rel_time < 60, SENSOR_COLUMNS]
        center = base.median()
        threshold = 3 * base.std()
        levels = g.groupby(g.rel_time.floordiv(10).astype(int))[SENSOR_COLUMNS].median()
        levels = levels.reindex(range(6, int(levels.index.max()) + 1))
        levels = levels[(levels.index + 1) * 10 <= g.rel_time.max() + 1]
        delta = levels - center
        signs = delta.gt(threshold + 1e-10).astype(int) - delta.lt(-threshold - 1e-10).astype(int)
        sustained = signs.ne(0) & signs.eq(signs.shift()) & signs.eq(signs.shift(2))
        for sensor in SENSOR_COLUMNS:
            hits = sustained.index[sustained[sensor]]
            onset = (hits[0] - 2) * 10 if len(hits) else np.nan
            rows.append((exp, g.tr_method.iloc[0], sensor, onset, end - onset))
    r = pd.DataFrame(rows, columns=["실험","방식","원본센서","시작초","선행초"])
    r["최초"] = r.시작초.eq(r.groupby("실험").시작초.transform("min"))
    r["센서"] = r.원본센서.map(SENSOR_LABELS)
    r["방식"] = r.방식.map({1:"가열",2:"과충전"})
    s = r.groupby(["방식","센서"]).agg(최초횟수=("최초","sum"), 검출수=("시작초","count"), 시작초중앙값=("시작초","median"), 선행초중앙값=("선행초","median"), 선행표본=("선행초","count")).sort_values(["방식","최초횟수"],ascending=[True,False])
    GRAPH_PATH.mkdir(exist_ok=True,parents=True)
    focus_figure = save_focus_comparison(r, GRAPH_PATH)
    s.to_csv(GRAPH_PATH/"센서_최초변화_비교.csv",encoding="utf-8-sig")
    r.to_csv(GRAPH_PATH/"실험별_센서_최초변화.csv",index=False,encoding="utf-8-sig")
    fig, axes = plt.subplots(1,2,figsize=(11,5),layout="constrained")
    for ax,(method,group) in zip(axes,s.groupby(level=0)):
        top = group.droplevel(0).head(5).iloc[::-1]
        ax.barh(top.index,top.최초횟수,color="steelblue" if method=="과충전" else "coral")
        ax.set(title=method,xlabel="가장 먼저 변한 실험 수 (동률 포함)",ylabel="센서")
    fig.suptitle("어떤 센서가 먼저 반응했을까?")
    fig.savefig(GRAPH_PATH/"센서_최초변화_비교.png",dpi=160)
    plt.close(fig)
    print(s.groupby(level=0).head(5).to_string())
    if show:
        focus_figure.canvas.manager.set_window_title("실험별 중점 센서 비교")
        plt.show(block=True)
    else:
        plt.close(focus_figure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험별 센서 변화 분석 및 시각화")
    parser.add_argument("--no-show", action="store_true", help="그래프 창을 열지 않고 결과만 저장")
    main(show=not parser.parse_args().no_show)
