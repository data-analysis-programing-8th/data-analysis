import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PATH, GRAPH_PATH


def plot_mean_stage_arrival(data_path=DATA_PATH, output_dir=GRAPH_PATH):
    stages = [1, 2, 3, 4]
    methods = {1: "가열", 2: "과충전"}

    df = pd.read_csv(
        data_path,
        usecols=["exp_id", "tr_method", "tr_stage", "rel_time"],
        dtype={"exp_id": str},
    )
    selected = df.loc[
        df["tr_stage"].isin(stages) & df["tr_method"].isin(methods)
    ]

    # 실험마다 각 단계에 처음 도달한 경과 시간(초)
    first_arrival = (
        selected.groupby(["tr_method", "exp_id", "tr_stage"])["rel_time"]
        .min()
    )

    # 방식과 단계별로 실험당 한 값씩 평균
    summary = (
        first_arrival.groupby(level=["tr_method", "tr_stage"])
        .agg(mean_seconds="mean", experiment_count="count")
    )

    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.36

    for (method, name), offset in zip(methods.items(), [-width / 2, width / 2]):
        positions = []
        heights = []
        labels = []

        for x, stage in enumerate(stages):
            key = (method, stage)
            if key not in summary.index:
                continue

            mean = summary.at[key, "mean_seconds"]
            count = summary.at[key, "experiment_count"]
            if count == 0:
                continue

            positions.append(x + offset)
            heights.append(mean)
            labels.append(f"{mean:,.1f}초\n({count}회)")

        bars = ax.bar(positions, heights, width=width, label=name)
        ax.bar_label(bars, labels=labels, padding=4, fontsize=9)

    ax.set_xticks(range(len(stages)), [f"{stage}단계" for stage in stages])
    ax.set_title("가열·과충전의 열폭주 단계별 평균 최초 도달 시간")
    ax.set_xlabel("열폭주 단계")
    ax.set_ylabel("실험 시작부터 걸린 시간 (초)")
    ax.set_ylim(0, max(1, ax.get_ylim()[1] * 1.15))
    ax.legend(title="실험 방식")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "mean_stage_arrival.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_mean_stage_arrival()